import torch  # PyTorch Tensor 연산 및 모델 구현
from functools import cached_property  # 한 번 계산한 property 값을 캐싱
from huggingface_hub import PyTorchModelHubMixin  # Hugging Face Hub 모델 저장/로드 기능
from typing import List, NamedTuple  # 타입 힌트와 여러 출력값 묶음 정의
from torch import nn  # PyTorch 신경망 모듈
from torch import Tensor  # Tensor 타입 힌트

from modules.encoder import MLP  # Encoder/Decoder에 사용할 MLP
from modules.loss import ReconstructionLoss, RqVaeLoss  # reconstruction loss와 최종 RQ-VAE loss
from modules.quantize import Quantize, QuantizeForwardMode  # Q1/Q2/Q3 quantizer와 quantization 방식

torch.set_float32_matmul_precision("high")  # float32 행렬곱 연산 성능을 높이는 설정


class RqVaeOutput(NamedTuple):  # Semantic ID 생성 과정의 quantization 결과
    embeddings: Tensor  # 선택된 q1/q2/q3, shape=[B,D,3]
    residuals: Tensor  # 각 quantizer 입력 h/r1/r2, shape=[B,D,3]
    sem_ids: Tensor  # Semantic ID [c1,c2,c3], shape=[B,3]
    codebook_loss: Tensor  # Q1+Q2+Q3의 raw codebook loss
    commitment_loss: Tensor  # Q1+Q2+Q3의 raw commitment loss


class RqVaeComputedLosses(NamedTuple):  # RQ-VAE training forward 결과
    loss: Tensor  # reconstruction+codebook+commitment를 합친 최종 weighted loss
    reconstruction_loss: Tensor  # raw reconstruction loss
    codebook_loss: Tensor  # raw codebook loss
    commitment_loss: Tensor  # raw commitment loss
    rqvae_loss: Tensor  # weighted codebook+commitment loss
    embs_norm: Tensor  # q1/q2/q3 vector norm 확인용
    p_unique_ids: Tensor  # 현재 batch의 unique Semantic ID 비율


class RqVae(nn.Module, PyTorchModelHubMixin):  # 뉴스용 전체 RQ-VAE 모델
    def __init__(
        self,
        input_dim: int,  # 원본 기사 embedding 차원, 예: 768
        embed_dim: int,  # RQ-VAE latent/code vector 차원, 예: 128
        hidden_dims: List[int],  # Encoder hidden dimensions, 예: [512,256]
        num_categories: int,  # Q1 category 수, 예: 25
        c2_codebook_size: int = 256,  # Q2 code 개수
        c3_codebook_size: int = 256,  # Q3 code 개수
        codebook_normalize: bool = False,  # latent/codebook vector L2 normalization 여부
        codebook_sim_vq: bool = False,  # codebook에 SIM-VQ projection을 적용할지 여부
        codebook_mode: QuantizeForwardMode = QuantizeForwardMode.STE,  # Q2/Q3 gradient 전달 방식
        lambda_rec: float = 1.0,  # reconstruction loss 가중치
        lambda_cb: float = 1.0,  # codebook loss 가중치
        lambda_com: float = 0.25,  # commitment loss 가중치
    ) -> None:
        super().__init__()  # nn.Module 초기화

        self.input_dim = input_dim  # 원본 embedding 차원 저장
        self.embed_dim = embed_dim  # latent embedding 차원 저장
        self.hidden_dims = hidden_dims  # hidden layer 차원 저장
        self.num_categories = num_categories  # Q1 category 개수 저장
        self.c2_codebook_size = c2_codebook_size  # Q2 codebook 크기 저장
        self.c3_codebook_size = c3_codebook_size  # Q3 codebook 크기 저장

        self._config = {  # checkpoint에서 같은 모델 구조를 다시 만들기 위한 설정 저장
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

        self.encoder = MLP(  # 원본 기사 embedding x(a)를 latent h(a)로 변환
            input_dim=input_dim,  # 768차원 입력
            hidden_dims=hidden_dims,  # 예: 512→256
            out_dim=embed_dim,  # 최종 latent 차원, 예: 128
            normalize=codebook_normalize,  # 설정에 따라 Encoder 출력 L2 normalization
        )

        self.decoder = MLP(  # q1+q2+q3를 원래 기사 embedding 차원으로 복원
            input_dim=embed_dim,  # latent 입력, 예: 128
            hidden_dims=hidden_dims[-1::-1],  # Encoder hidden dims를 역순으로 사용, 예: [256,512]
            out_dim=input_dim,  # 최종 원본 embedding 차원, 예: 768
            normalize=False,  # 복원 embedding에는 L2 normalization 적용하지 않음
        )

        self.quantizer_1 = Quantize(  # Q1 category-level quantizer
            embed_dim=embed_dim,  # Q1 vector 차원
            n_embed=num_categories,  # category 수만큼 Q1 vector 생성
            codebook_normalize=codebook_normalize,  # Q1 vector normalization 여부
            sim_vq=codebook_sim_vq,  # SIM-VQ projection 여부
            forward_mode=codebook_mode,  # Q1은 fixed_ids 방식이라 실제 nearest STE에는 사용되지 않음
        )

        self.quantizer_2 = Quantize(  # Q2 event-level quantizer
            embed_dim=embed_dim,  # Q2 vector 차원
            n_embed=c2_codebook_size,  # Q2 code 개수, 예: 256
            codebook_normalize=codebook_normalize,  # Q2 vector normalization 여부
            sim_vq=codebook_sim_vq,  # SIM-VQ projection 여부
            forward_mode=codebook_mode,  # STE/Gumbel/Rotation 방식 지정
        )

        self.quantizer_3 = Quantize(  # Q3 residual-level quantizer
            embed_dim=embed_dim,  # Q3 vector 차원
            n_embed=c3_codebook_size,  # Q3 code 개수, 예: 256
            codebook_normalize=codebook_normalize,  # Q3 vector normalization 여부
            sim_vq=codebook_sim_vq,  # SIM-VQ projection 여부
            forward_mode=codebook_mode,  # STE/Gumbel/Rotation 방식 지정
        )

        self.reconstruction_loss_fn = ReconstructionLoss()  # x와 x_hat 사이 reconstruction loss 계산

        self.loss_fn = RqVaeLoss(  # 세 loss를 가중합하는 최종 loss 함수
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
        )

    @cached_property
    def config(self) -> dict:  # 모델 구조를 checkpoint에 함께 저장할 때 사용하는 config
        return self._config

    @property
    def device(self) -> torch.device:  # 현재 모델이 올라가 있는 CPU/GPU device 반환
        return next(self.encoder.parameters()).device

    def encode(self, x: Tensor) -> Tensor:  # x(a)를 Encoder에 넣어 h(a) 생성
        return self.encoder(x)

    def decode(self, x: Tensor) -> Tensor:  # quantized latent를 Decoder에 넣어 x_hat 생성
        return self.decoder(x)

    @torch.no_grad()  # Q2 초기화 과정에서는 gradient를 기록하지 않음
    def set_c2_codebook(self, centroids: Tensor) -> None:  # K-means centroid로 Q2 codebook 초기화
        expected_shape = (self.c2_codebook_size, self.embed_dim)  # 기대 shape, 예: [256,128]

        if tuple(centroids.shape) != expected_shape:  # centroid shape가 Q2와 일치하는지 확인
            raise ValueError(
                "C2 centroid shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {tuple(centroids.shape)}."
            )

        self.quantizer_2.set_codebook(centroids)  # Q2 random vector를 K-means centroid로 덮어씀

    def get_semantic_ids(
        self,
        x: Tensor,  # 원본 기사 embedding, shape=[B,input_dim]
        category_ids: Tensor,  # 기사 category ID, shape=[B]
        gumbel_t: float = 0.001,  # Gumbel-Softmax 사용 시 temperature
    ) -> RqVaeOutput:

        x = x.to(  # 입력 embedding을 모델과 같은 device/dtype으로 변환
            device=self.device,
            dtype=next(self.encoder.parameters()).dtype,
        )

        category_ids = category_ids.to(  # category ID를 모델 device로 이동
            device=self.device,
            dtype=torch.long,  # nn.Embedding index이므로 long 타입 사용
        )

        h = self.encode(x)  # 기사 embedding x(a)를 latent h(a)로 변환

        q1_out = self.quantizer_1(  # Q1 category code 선택
            x=h,  # Q1 query는 h(a)
            temperature=gumbel_t,  # Q1 fixed-id 방식에서는 사실상 사용되지 않음
            fixed_ids=category_ids,  # category ID를 그대로 c1으로 사용
        )

        q1 = q1_out.embeddings  # category에 대응하는 Q1 vector
        c1 = q1_out.ids  # category ID 자체가 c1

        r1 = h - q1  # Q1이 설명한 부분을 제거해 첫 번째 residual 생성

        q2_out = self.quantizer_2(  # r1에서 가장 가까운 Q2 code 선택
            x=r1,
            temperature=gumbel_t,
        )

        q2 = q2_out.embeddings  # 선택된 Q2 vector
        c2 = q2_out.ids  # 선택된 Q2 code index

        r2 = r1 - q2  # Q2가 설명한 부분까지 제거해 두 번째 residual 생성

        q3_out = self.quantizer_3(  # r2에서 가장 가까운 Q3 code 선택
            x=r2,
            temperature=gumbel_t,
        )

        q3 = q3_out.embeddings  # 선택된 Q3 vector
        c3 = q3_out.ids  # 선택된 Q3 code index

        codebook_loss = (  # 세 quantization level의 codebook loss를 합침
            q1_out.codebook_loss
            + q2_out.codebook_loss
            + q3_out.codebook_loss
        )

        commitment_loss = (  # 세 quantization level의 commitment loss를 합침
            q1_out.commitment_loss
            + q2_out.commitment_loss
            + q3_out.commitment_loss
        )

        embeddings = torch.stack(  # q1/q2/q3를 마지막 차원에 쌓음
            [q1, q2, q3],
            dim=-1,
        )  # shape=[B,D,3]

        residuals = torch.stack(  # 각 quantizer에 실제 입력된 h/r1/r2를 저장
            [h, r1, r2],
            dim=-1,
        )  # shape=[B,D,3]

        sem_ids = torch.stack(  # c1/c2/c3를 하나의 Semantic ID로 묶음
            [c1, c2, c3],
            dim=-1,
        )  # shape=[B,3]

        return RqVaeOutput(  # quantization 결과 반환
            embeddings=embeddings,
            residuals=residuals,
            sem_ids=sem_ids,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

    def forward(
        self,
        x: Tensor,  # 원본 기사 embedding
        category_ids: Tensor,  # category ID
        gumbel_t: float = 0.001,  # Gumbel temperature
    ) -> RqVaeComputedLosses:

        quantized = self.get_semantic_ids(  # Q1/Q2/Q3 quantization 및 Semantic ID 생성
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
        )

        quantized_embedding = quantized.embeddings.sum(dim=-1)  # q1+q2+q3를 더해 최종 quantized latent 생성

        x_hat = self.decode(quantized_embedding)  # quantized latent를 원본 embedding 차원으로 복원

        reconstruction_loss = self.reconstruction_loss_fn(  # 기사별 reconstruction loss 계산
            x_hat=x_hat,
            x=x,
        )

        codebook_loss = quantized.codebook_loss  # Q1+Q2+Q3 raw codebook loss
        commitment_loss = quantized.commitment_loss  # Q1+Q2+Q3 raw commitment loss

        total_loss_per_sample = self.loss_fn(  # 기사별 최종 weighted loss 계산
            reconstruction_loss=reconstruction_loss,
            codebook_loss=codebook_loss,
            commitment_loss=commitment_loss,
        )

        loss = total_loss_per_sample.mean()  # batch 전체 loss 평균을 backward용 scalar로 변환

        rqvae_loss_per_sample = (  # reconstruction을 제외한 quantization loss만 logging용으로 계산
            self.loss_fn.lambda_cb * codebook_loss
            + self.loss_fn.lambda_com * commitment_loss
        )

        with torch.no_grad():  # 아래 값들은 디버깅/로그용이라 gradient가 필요 없음
            embs_norm = quantized.embeddings.norm(
                p=2,
                dim=1,
            )  # 각 기사의 q1/q2/q3 vector norm 계산, shape=[B,3]

            num_articles = quantized.sem_ids.shape[0]  # 현재 batch 기사 개수

            if num_articles == 0:  # 빈 batch라면 unique 비율을 0으로 설정
                p_unique_ids = torch.tensor(
                    0.0,
                    device=self.device,
                    dtype=torch.float32,
                )
            else:
                num_unique = torch.unique(
                    quantized.sem_ids,
                    dim=0,
                ).shape[0]  # batch에서 서로 다른 full SID 개수 계산

                p_unique_ids = torch.tensor(
                    num_unique / num_articles,  # unique SID 개수 / 전체 기사 수
                    device=self.device,
                    dtype=torch.float32,
                )

        return RqVaeComputedLosses(  # 학습 및 logging에 필요한 값 반환
            loss=loss,  # 최종 weighted batch loss
            reconstruction_loss=reconstruction_loss.mean(),  # 평균 raw reconstruction loss
            codebook_loss=codebook_loss.mean(),  # 평균 raw codebook loss
            commitment_loss=commitment_loss.mean(),  # 평균 raw commitment loss
            rqvae_loss=rqvae_loss_per_sample.mean(),  # 평균 weighted quantization loss
            embs_norm=embs_norm,  # q1/q2/q3 norm
            p_unique_ids=p_unique_ids,  # batch unique SID 비율
        )

    def load_pretrained(self, path: str) -> None:  # 저장된 RQ-VAE checkpoint를 현재 모델에 불러오는 함수
        state = torch.load(
            path,
            map_location=self.device,  # 현재 모델 device에 checkpoint 로드
            weights_only=False,  # weight 외의 iteration/config 등도 함께 읽음
        )

        self.load_state_dict(state["model"])  # checkpoint의 모델 parameter를 현재 모델에 적용

        if "iter" in state:  # checkpoint에 저장된 iteration 정보가 있으면 함께 출력
            print(f"Loaded RQ-VAE checkpoint (iteration={state['iter']})")
        else:
            print("Loaded RQ-VAE checkpoint.")  # iteration 정보가 없으면 단순 완료 메시지 출력