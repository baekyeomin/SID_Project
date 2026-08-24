import torch  
from functools import cached_property  
from huggingface_hub import PyTorchModelHubMixin  
from typing import List, NamedTuple  
from torch import nn  
from torch import Tensor  

from modules.encoder import MLP  
from modules.loss import ReconstructionLoss, RqVaeLoss  
from modules.quantize import Quantize, QuantizeForwardMode  

torch.set_float32_matmul_precision("high")  


class RqVaeOutput(NamedTuple):  
    embeddings: Tensor  
    residuals: Tensor  
    sem_ids: Tensor  
    codebook_loss: Tensor  
    commitment_loss: Tensor  


class RqVaeComputedLosses(NamedTuple):  
    loss: Tensor  
    reconstruction_loss: Tensor  
    codebook_loss: Tensor  
    commitment_loss: Tensor  
    rqvae_loss: Tensor  
    embs_norm: Tensor  
    p_unique_ids: Tensor  


class RqVae(nn.Module, PyTorchModelHubMixin):  
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

        self.encoder = MLP(  
            input_dim=input_dim,  
            hidden_dims=hidden_dims,  
            out_dim=embed_dim,  
            normalize=codebook_normalize,  
        )

        self.decoder = MLP(  
            input_dim=embed_dim,  
            hidden_dims=hidden_dims[-1::-1],  
            out_dim=input_dim,  
            normalize=False,  
        )

        self.quantizer_1 = Quantize(  
            embed_dim=embed_dim,  
            n_embed=num_categories,  
            codebook_normalize=codebook_normalize,  
            sim_vq=codebook_sim_vq,  
            forward_mode=codebook_mode,  
        )

        self.quantizer_2 = Quantize(  
            embed_dim=embed_dim,  
            n_embed=c2_codebook_size,  
            codebook_normalize=codebook_normalize,  
            sim_vq=codebook_sim_vq,  
            forward_mode=codebook_mode,  
        )

        self.quantizer_3 = Quantize(  
            embed_dim=embed_dim,  
            n_embed=c3_codebook_size,  
            codebook_normalize=codebook_normalize,  
            sim_vq=codebook_sim_vq,  
            forward_mode=codebook_mode,  
        )

        self.reconstruction_loss_fn = ReconstructionLoss()  

        self.loss_fn = RqVaeLoss(  
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
        )

    @cached_property
    def config(self) -> dict:  
        return self._config

    @property
    def device(self) -> torch.device:  
        return next(self.encoder.parameters()).device

    def encode(self, x: Tensor) -> Tensor:  
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:  
        return self.decoder(x)

    @torch.no_grad()  
    def set_c2_codebook(self, centroids: Tensor) -> None:  
        expected_shape = (self.c2_codebook_size, self.embed_dim)  

        if tuple(centroids.shape) != expected_shape:  
            raise ValueError(
                "C2 centroid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {tuple(centroids.shape)}."
            )

        self.quantizer_2.set_codebook(centroids)  

    def get_semantic_ids(
        self,
        x: Tensor,  
        category_ids: Tensor,  
        gumbel_t: float = 0.001,  
    ) -> RqVaeOutput:

        x = x.to(  
            device=self.device,
            dtype=next(self.encoder.parameters()).dtype,
        )

        category_ids = category_ids.to(  
            device=self.device,
            dtype=torch.long,  
        )

        h = self.encode(x)  

        q1_out = self.quantizer_1(  
            x=h,  
            temperature=gumbel_t,  
            fixed_ids=category_ids,  
        )

        q1 = q1_out.embeddings  
        c1 = q1_out.ids  

        r1 = h - q1  

        q2_out = self.quantizer_2(  
            x=r1,
            temperature=gumbel_t,
        )

        q2 = q2_out.embeddings  
        c2 = q2_out.ids  

        r2 = r1 - q2  

        q3_out = self.quantizer_3(  
            x=r2,
            temperature=gumbel_t,
        )

        q3 = q3_out.embeddings  
        c3 = q3_out.ids  

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

        embeddings = torch.stack(  
            [q1, q2, q3],
            dim=-1,
        )  

        residuals = torch.stack(  
            [h, r1, r2],
            dim=-1,
        )  

        sem_ids = torch.stack(  
            [c1, c2, c3],
            dim=-1,
        )  

        return RqVaeOutput(  
            embeddings=embeddings,
            residuals=residuals,
            sem_ids=sem_ids,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

    def forward(
        self,
        x: Tensor,  
        category_ids: Tensor,  
        gumbel_t: float = 0.001,  
    ) -> RqVaeComputedLosses:

        quantized = self.get_semantic_ids(  
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
        )

        quantized_embedding = quantized.embeddings.sum(dim=-1)  

        x_hat = self.decode(quantized_embedding)  

        reconstruction_loss = self.reconstruction_loss_fn(  
            x_hat=x_hat,
            x=x,
        )

        codebook_loss = quantized.codebook_loss  
        commitment_loss = quantized.commitment_loss  

        total_loss_per_sample = self.loss_fn(  
            reconstruction_loss=reconstruction_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

        loss = total_loss_per_sample.mean()  

        rqvae_loss_per_sample = (  
            self.loss_fn.lambda_cb * codebook_loss
            + self.loss_fn.lambda_com * commitment_loss
        )

        with torch.no_grad():  
            embs_norm = quantized.embeddings.norm(
                p=2,
                dim=1,
            )  

            num_articles = quantized.sem_ids.shape[0]  

            if num_articles == 0:  
                p_unique_ids = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                num_unique = torch.unique(
                    quantized.sem_ids,
                    dim=0,
                ).shape[0]  

                p_unique_ids = torch.tensor(
                    num_unique / num_articles,  
                    device=self.device,
                    dtype=torch.float32,
                )

        return RqVaeComputedLosses(  
            loss=loss,  
            reconstruction_loss=reconstruction_loss.mean(),  
            codebook_loss=codebook_loss.mean(),  
            commitment_loss=commitment_loss.mean(),  
            rqvae_loss=rqvae_loss_per_sample.mean(),  
            embs_norm=embs_norm,  
            p_unique_ids=p_unique_ids,  
        )

    def load_pretrained(self, path: str) -> None:  
        state = torch.load(
            path,
            map_location=self.device,  
            weights_only=False,  
        )

        self.load_state_dict(state["model"])  

        if "iter" in state:  
            print(f"Loaded RQ-VAE checkpoint (iteration={state['iter']})")
        else:
            print("Loaded RQ-VAE checkpoint.")  