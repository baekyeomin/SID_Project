import torch

from functools import cached_property
from huggingface_hub import PyTorchModelHubMixin
from typing import List, NamedTuple, Optional

from torch import nn
from torch import Tensor

from modules.encoder import MLP
from modules.loss import ReconstructionLoss, RqVaeLoss
from modules.quantize import Quantize, QuantizeForwardMode


torch.set_float32_matmul_precision("high")


# ============================================================
# RQ-VAE Output
# ============================================================

class RqVaeOutput(NamedTuple):
    embeddings: Tensor
    residuals: Tensor
    sem_ids: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor


# ============================================================
# RQ-VAE Loss Output
# ============================================================

class RqVaeComputedLosses(NamedTuple):
    loss: Tensor
    reconstruction_loss: Tensor
    codebook_loss: Tensor
    commitment_loss: Tensor
    rqvae_loss: Tensor
    embs_norm: Tensor
    p_unique_ids: Tensor


# ============================================================
# RQ-VAE
# ============================================================

class RqVae(
    nn.Module,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        hidden_dims: List[int],
        num_categories: int,
        c2_codebook_size: int = 256,
        c3_codebook_size: int = 256,
        codebook_normalize: bool = False,
        codebook_sim_vq: bool = False,
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,
        lambda_rec: float = 1.0,
        lambda_cb: float = 1.0,
        lambda_com: float = 0.25,
    ) -> None:

        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.hidden_dims = hidden_dims
        self.num_categories = num_categories
        self.c2_codebook_size = c2_codebook_size
        self.c3_codebook_size = c3_codebook_size

        # ----------------------------------------------------
        # Model config
        # ----------------------------------------------------

        self._config = {
            "input_dim": input_dim,
            "embed_dim": embed_dim,
            "hidden_dims": hidden_dims,
            "num_categories": num_categories,
            "c2_codebook_size": c2_codebook_size,
            "c3_codebook_size": c3_codebook_size,
            "codebook_normalize": codebook_normalize,
            "codebook_sim_vq": codebook_sim_vq,
            "codebook_mode": codebook_mode,
            "lambda_rec": lambda_rec,
            "lambda_cb": lambda_cb,
            "lambda_com": lambda_com,
        }

        # ----------------------------------------------------
        # Encoder
        #
        # x(a) ∈ R^input_dim
        #     ↓
        # h(a) ∈ R^embed_dim
        # ----------------------------------------------------

        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            out_dim=embed_dim,
            normalize=codebook_normalize,
        )

        # ----------------------------------------------------
        # Decoder
        #
        # q1 + q2 + q3
        #     ↓
        # x_hat ∈ R^input_dim
        # ----------------------------------------------------

        self.decoder = MLP(
            input_dim=embed_dim,
            hidden_dims=hidden_dims[-1::-1],
            out_dim=input_dim,
            normalize=False,
        )

        # ----------------------------------------------------
        # Q1
        #
        # C1 index:
        # category ID로 deterministic하게 지정
        #
        # Q1 vector:
        # trainable embedding
        # ----------------------------------------------------

        self.quantizer_1 = Quantize(
            embed_dim=embed_dim,
            n_embed=num_categories,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Q2
        #
        # 학습:
        # r1 기준 quantization
        #
        # 최종 SID 생성:
        # event 단위로 미리 계산한 fixed_c2_ids를
        # 전달할 수도 있음
        # ----------------------------------------------------

        self.quantizer_2 = Quantize(
            embed_dim=embed_dim,
            n_embed=c2_codebook_size,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Q3
        #
        # r2에 대해 article-level quantization
        # ----------------------------------------------------

        self.quantizer_3 = Quantize(
            embed_dim=embed_dim,
            n_embed=c3_codebook_size,
            codebook_normalize=codebook_normalize,
            sim_vq=codebook_sim_vq,
            forward_mode=codebook_mode,
        )

        # ----------------------------------------------------
        # Loss
        # ----------------------------------------------------

        self.reconstruction_loss_fn = ReconstructionLoss()

        self.loss_fn = RqVaeLoss(
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
        )


    # ========================================================
    # Properties
    # ========================================================

    @cached_property
    def config(self) -> dict:
        return self._config


    @property
    def device(self) -> torch.device:
        return next(
            self.encoder.parameters()
        ).device


    # ========================================================
    # Encoder / Decoder
    # ========================================================

    def encode(
        self,
        x: Tensor,
    ) -> Tensor:

        return self.encoder(x)


    def decode(
        self,
        x: Tensor,
    ) -> Tensor:

        return self.decoder(x)


    # ========================================================
    # Q2 Codebook Initialization
    # ========================================================

    @torch.no_grad()
    def set_c2_codebook(
        self,
        centroids: Tensor,
    ) -> None:

        expected_shape = (
            self.c2_codebook_size,
            self.embed_dim,
        )

        actual_shape = tuple(
            centroids.shape
        )

        if actual_shape != expected_shape:

            raise ValueError(
                "C2 centroid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )

        self.quantizer_2.set_codebook(
            centroids
        )


    # ========================================================
    # C2 Residual
    #
    # Event representation 계산 시 사용 가능
    #
    # x
    # ↓
    # Encoder
    # ↓
    # h
    # ↓
    # Q1[category]
    # ↓
    # r1 = h - q1
    #
    # Q2가 실제로 받는 입력은 r1이므로
    # event-level C2를 생성할 때도 r1을 평균낼 수 있음
    # ========================================================

    def get_c2_residual(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
    ) -> Tensor:

        x = x.to(
            device=self.device,
            dtype=next(
                self.encoder.parameters()
            ).dtype,
        )

        category_ids = category_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        # Encoder
        h = self.encode(x)

        # Q1:
        # category ID가 C1 index를 결정
        q1_out = self.quantizer_1(
            x=h,
            temperature=gumbel_t,
            fixed_ids=category_ids,
        )

        q1 = q1_out.embeddings

        # Q2가 받는 residual
        r1 = h - q1

        return r1


    # ========================================================
    # Semantic ID Generation
    # ========================================================

    def get_semantic_ids(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
        fixed_c2_ids: Optional[Tensor] = None,
    ) -> RqVaeOutput:

        # ----------------------------------------------------
        # Input device / dtype
        # ----------------------------------------------------

        x = x.to(
            device=self.device,
            dtype=next(
                self.encoder.parameters()
            ).dtype,
        )

        category_ids = category_ids.to(
            device=self.device,
            dtype=torch.long,
        )

        if fixed_c2_ids is not None:

            fixed_c2_ids = fixed_c2_ids.to(
                device=self.device,
                dtype=torch.long,
            )

            fixed_c2_ids = fixed_c2_ids.view(-1)

            if (
                fixed_c2_ids.shape[0]
                != x.shape[0]
            ):

                raise ValueError(
                    "fixed_c2_ids batch size "
                    "must match x. "
                    f"x batch={x.shape[0]}, "
                    f"fixed_c2_ids batch="
                    f"{fixed_c2_ids.shape[0]}."
                )

        # ----------------------------------------------------
        # Encoder
        #
        # x → h
        # ----------------------------------------------------

        h = self.encode(x)

        # ----------------------------------------------------
        # Q1
        #
        # c1 = category ID
        # q1 = Q1[c1]
        # ----------------------------------------------------

        q1_out = self.quantizer_1(
            x=h,
            temperature=gumbel_t,
            fixed_ids=category_ids,
        )

        q1 = q1_out.embeddings
        c1 = q1_out.ids

        # ----------------------------------------------------
        # Residual 1
        #
        # r1 = h - q1
        # ----------------------------------------------------

        r1 = h - q1

        # ----------------------------------------------------
        # Q2
        #
        # fixed_c2_ids == None
        # → 기존 방식
        # → r1에서 Q2 code 선택
        #
        # fixed_c2_ids != None
        # → event-level에서 미리 정해둔
        #   C2 index를 그대로 사용
        # ----------------------------------------------------

        q2_out = self.quantizer_2(
            x=r1,
            temperature=gumbel_t,
            fixed_ids=fixed_c2_ids,
        )

        q2 = q2_out.embeddings
        c2 = q2_out.ids

        # ----------------------------------------------------
        # Residual 2
        #
        # r2 = r1 - q2
        #    = h - q1 - q2
        # ----------------------------------------------------

        r2 = r1 - q2

        # ----------------------------------------------------
        # Q3
        #
        # r2에서 article-level code 선택
        # ----------------------------------------------------

        q3_out = self.quantizer_3(
            x=r2,
            temperature=gumbel_t,
        )

        q3 = q3_out.embeddings
        c3 = q3_out.ids

        # ----------------------------------------------------
        # Quantization Loss
        # ----------------------------------------------------

        codebook_loss = (
            q1_out.codebook_loss
            + q2_out.codebook_loss
            + q3_out.codebook_loss
        )

        commitment_loss = (
            q1_out.commitment_loss
            + q2_out.commitment_loss
            + q3_out.commitment_loss
        )

        # ----------------------------------------------------
        # Quantized embeddings
        #
        # shape:
        # [batch, embed_dim, 3]
        # ----------------------------------------------------

        embeddings = torch.stack(
            [
                q1,
                q2,
                q3,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Residuals
        #
        # Q1 input = h
        # Q2 input = r1
        # Q3 input = r2
        # ----------------------------------------------------

        residuals = torch.stack(
            [
                h,
                r1,
                r2,
            ],
            dim=-1,
        )

        # ----------------------------------------------------
        # Semantic IDs
        #
        # (c1, c2, c3)
        # ----------------------------------------------------

        sem_ids = torch.stack(
            [
                c1,
                c2,
                c3,
            ],
            dim=-1,
        )

        return RqVaeOutput(
            embeddings=embeddings,
            residuals=residuals,
            sem_ids=sem_ids,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )


    # ========================================================
    # Training Forward
    # ========================================================

    def forward(
        self,
        x: Tensor,
        category_ids: Tensor,
        gumbel_t: float = 0.001,
    ) -> RqVaeComputedLosses:

        # ----------------------------------------------------
        # 학습 단계에서는 fixed_c2_ids를 주지 않음
        #
        # 즉 Q2는 기존 학습 방식대로
        # r1에서 code를 선택
        # ----------------------------------------------------

        quantized = self.get_semantic_ids(
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
            fixed_c2_ids=None,
        )

        # ----------------------------------------------------
        # q1 + q2 + q3
        # ----------------------------------------------------

        quantized_embedding = (
            quantized
            .embeddings
            .sum(dim=-1)
        )

        # ----------------------------------------------------
        # Decoder
        #
        # q1 + q2 + q3
        # ↓
        # x_hat
        # ----------------------------------------------------

        x_hat = self.decode(
            quantized_embedding
        )

        # ----------------------------------------------------
        # Reconstruction Loss
        # ----------------------------------------------------

        reconstruction_loss = (
            self.reconstruction_loss_fn(
                x_hat=x_hat,
                x=x,
            )
        )

        # ----------------------------------------------------
        # Quantization Loss
        # ----------------------------------------------------

        codebook_loss = (
            quantized.codebook_loss
        )

        commitment_loss = (
            quantized.commitment_loss
        )

        # ----------------------------------------------------
        # Total Loss
        #
        # λ_rec * reconstruction
        # + λ_cb * codebook
        # + λ_com * commitment
        # ----------------------------------------------------

        total_loss_per_sample = (
            self.loss_fn(
                reconstruction_loss=(
                    reconstruction_loss
                ),
                codebook_loss=(
                    codebook_loss
                ),
                commitment_loss=(
                    commitment_loss
                ),
            )
        )

        loss = (
            total_loss_per_sample
            .mean()
        )

        # ----------------------------------------------------
        # RQ-VAE Quantization Loss
        # ----------------------------------------------------

        rqvae_loss_per_sample = (
            self.loss_fn.lambda_cb
            * codebook_loss
            +
            self.loss_fn.lambda_com
            * commitment_loss
        )

        # ----------------------------------------------------
        # Logging Metrics
        # ----------------------------------------------------

        with torch.no_grad():

            embs_norm = (
                quantized
                .embeddings
                .norm(
                    p=2,
                    dim=1,
                )
            )

            num_articles = (
                quantized
                .sem_ids
                .shape[0]
            )

            if num_articles == 0:

                p_unique_ids = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=torch.float32,
                )

            else:

                num_unique = (
                    torch.unique(
                        quantized.sem_ids,
                        dim=0,
                    )
                    .shape[0]
                )

                p_unique_ids = torch.tensor(
                    num_unique
                    / num_articles,
                    device=self.device,
                    dtype=torch.float32,
                )

        return RqVaeComputedLosses(
            loss=loss,
            reconstruction_loss=(
                reconstruction_loss.mean()
            ),
            codebook_loss=(
                codebook_loss.mean()
            ),
            commitment_loss=(
                commitment_loss.mean()
            ),
            rqvae_loss=(
                rqvae_loss_per_sample.mean()
            ),
            embs_norm=embs_norm,
            p_unique_ids=p_unique_ids,
        )


    # ========================================================
    # Load Pretrained
    # ========================================================

    def load_pretrained(
        self,
        path: str,
    ) -> None:

        state = torch.load(
            path,
            map_location=self.device,
            weights_only=False,
        )

        self.load_state_dict(
            state["model"]
        )

        if "iter" in state:

            print(
                "Loaded RQ-VAE checkpoint "
                f"(iteration={state['iter']})"
            )

        else:

            print(
                "Loaded RQ-VAE checkpoint."
            )