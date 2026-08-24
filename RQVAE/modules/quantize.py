import gin #config/gin에서 값 직접 설정할 수 있도록
import torch

from distributions.gumbel_softmax import gumbel_softmax
from distributions.rotationtrick import rotation_trick_transform # 일단 지금은 STE는 한줄이라 distribution에 빼두지 않음
from enum import Enum  # STE, Gumbel 등의 동작 방식을 Enum으로 정의하기 위해 사
from typing import NamedTuple, Optional  # 반환 객체와 optional argument의 타입 힌트를 위해 사용

from torch import nn
from torch import Tensor
from torch.nn import functional as F

from modules.loss import QuantizeLoss
from modules.normalize import L2NormalizationLayer


@gin.constants_from_enum  # Gin config에서 QuantizeForwardMode.STE 같은 Enum 값을 직접 사용할 수 있게 등록
class QuantizeForwardMode(Enum):  # Q2/Q3 quantization 시 gradient를 어떻게 전달할지 정의하는 Enum
    """
    Q2 / Q3에서 gradient를 전달하는 방식을 정의

    GUMBEL_SOFTMAX: soft한 code 선택

    STE: foward에서는 그대로 armin,
         backward에서는 gradient를 query 쪽으로 전달

    ROTATION_TRICK: Rotation Trick 기반 gradient 전달
    """
    GUMBEL_SOFTMAX = 1
    STE = 2
    ROTATION_TRICK = 3


class QuantizeDistance(Enum):# query와 codebook 사이의 거리를 어떤 방식으로 계산할지 정의
    L2 = 1
    COSINE = 2


class QuantizeOutput(NamedTuple): # quantizer가 반환할 결과들을 하나의 객체로 묶기 위한 자료구조
    """
    [임베딩]
    Q1: q1 = Q1[category_id]
    Q2: q2 = Q2[c2]
    Q3: q3 = Q3[c3]

    [인덱스]
    Q1: category_id 자체
    Q2 / Q3: nearest code index
    """

    embeddings: Tensor
    ids: Tensor

    codebook_loss: Tensor
    commitment_loss: Tensor


# Q1/Q2/Q3 각각 하나의 codebook을 양자화하는 클래스

class Quantize(nn.Module):
    def __init__(
        self,
        embed_dim: int, #각 코드벡터 차원
        n_embed: int, #코드북크기 (코드 수) / Q1=25, Q2/Q3=256
        codebook_normalize: bool = False,  # codebook vector를 L2 normalization할지 여부
        sim_vq: bool = False, # codebook에 추가 Linear projection을 적용할지 여부
        forward_mode: QuantizeForwardMode = QuantizeForwardMode.STE,  # Q2/Q3 gradient 전달 방식= STE
        distance_mode: QuantizeDistance = QuantizeDistance.L2, #유클리드 거리 사용
    ) -> None:

        super().__init__() # nn.Module 초기화

        self.embed_dim = embed_dim    # embedding/code vector 차원 저장
        self.n_embed = n_embed   # codebook 크기 저장

        self.forward_mode = forward_mode   # STE/Gumbel/Rotation Trick 중 사용할 방식 저장
        self.distance_mode = distance_mode   # L2/Cosine 중 사용할 거리 방식 저장

        # ========================================================
        # [Codebook]
        # shape: [n_embed, embed_dim]
        # Q1: [num_categories, embed_dim]
        # Q2: [256, embed_dim]
        # Q3: [256, embed_dim]
        # ========================================================

        self.embedding = nn.Embedding(
            num_embeddings=n_embed,  # code vector 개수
            embedding_dim=embed_dim,   # 각 code vector 차원
        )

        # ========================================================
        # Optional projection / normalization
        # ========================================================
        # 현재 설정에서는 기본적으로
        #     sim_vq = False
        #     codebook_normalize = False
        # 이므로 Identity로 동작
        # ========================================================

        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim, bias=False,)
            if sim_vq  # sim_vq=True일 경우 code vector에 Linear projection 적용
            else nn.Identity(), # False이면 아무 변환도 하지 않음

            L2NormalizationLayer(dim=-1)
            if codebook_normalize # codebook_normalize=True이면 code vector의 norm을 1로 맞춤
            else nn.Identity(),  # normalize=False이면 원래 vector 그대로 사용
        )

        self.quantize_loss = QuantizeLoss()

        # ========================================================
        # Random initialization
        #
        # Q1: 그대로 사용
        # Q2: 임시로 랜덤 초기화 후 train_rqvae.py에서 K-means centroid로 덮어씀
        # Q3: 그대로 사용
        # ========================================================

        self._init_weights()

    # ============================================================
    # Properties
    # ============================================================

    @property # 함수 호출이 아닌 객체 속성처럼 quantizer.weight로 접근할 수 있게 함
    def weight(self) -> Tensor:  #shape: [n_embed, embed_dim]
        return self.embedding.weight # raw codebook parameter를 그대로 반환

    @property # quantizer.device 형태로 현재 device에 접근 가능하게 함
    def device(self) -> torch.device:
        return self.embedding.weight.device

    # ============================================================
    # Random Initialization
    # ============================================================

    def _init_weights(self) -> None:
        nn.init.uniform_(
            self.embedding.weight,
            a=-1.0 / self.n_embed,
            b=1.0 / self.n_embed,
        )

    # ============================================================
    # External Codebook Initialization
    # ============================================================

    @torch.no_grad() # Q2 K-means 초기화는 학습이 아니므로 gradient를 기록하지 않음
    def set_codebook(self,codebook_vectors: Tensor, ) -> None: # 외부에서 계산한 centroid들, 예: [256,128]

        # 현재 Quantize가 기대하는 codebook shape 정의 [ code수 , 코드벡터차원]
        expected_shape = (self.n_embed, self.embed_dim,)

        # 전달받은 centroid의 실제 shape를 tuple로 변환
        actual_shape = tuple(codebook_vectors.shape)

        if actual_shape != expected_shape:  # K-means centroid shape가 Q2 shape와 다른지 검사

            raise ValueError(
                "Codebook shape mismatch. "
                f"Expected {expected_shape}, "
                f"got {actual_shape}."
            )

        self.embedding.weight.copy_( # 기존 random codebook weight를 K-means centroid로 덮어씀
            codebook_vectors.to( # 디바이스나 자료형 다 맞춤
                device=self.device,
                dtype=self.embedding.weight.dtype,
            )
        )

    # ============================================================
    # Code Lookup
    # ============================================================

    def get_item_embeddings( self, item_ids: Tensor,) -> Tensor:
        """
        code index에 해당하는 vector를 가져오는 부분

        예: ids = [3, 5, 8]

        결과: [Q[3], Q[5], Q[8]]

        Q1: item_ids = category_ids

        Q2/Q3: item_ids = nearest code IDs
        """

        embeddings = self.embedding(item_ids)

        embeddings = self.out_proj(embeddings)

        return embeddings

    # 한 article당 모든 code와의 거리계산

    def _compute_distance(self,x: Tensor,codebook: Tensor,) -> Tensor:
        # L2
        if (self.distance_mode== QuantizeDistance.L2):
            x_squared = ((x ** 2).sum(dim=1,keepdim=True,))
            codebook_squared = ((codebook ** 2).sum(dim=1,keepdim=True,).T)
            cross_term = (2* x@ codebook.T)

            dist = (x_squared + codebook_squared - cross_term)

        # Cosine Distance

        elif (self.distance_mode == QuantizeDistance.COSINE):

            x_normalized = F.normalize(x, p=2, dim=1, eps=1e-8,)

            codebook_normalized = (F.normalize(codebook, p=2, dim=1, eps=1e-8,))

            # cosine similarity가 클수록 가까우므로
            # 음수를 붙여 distance처럼 사용
            dist = -( x_normalized @ codebook_normalized.T)

        else:
            raise ValueError(
                "Unsupported distance mode: "
                f"{self.distance_mode}"
            )

        return dist # 각 query와 모든 code의 거리 matrix [B,K] 반환

    # Q1: Deterministic Category Quantization
    def _forward_fixed_ids(self, x: Tensor, fixed_ids: Tensor,) -> QuantizeOutput:
        """
        Q1에서는
        c1 = category_id
             q1 = Q1[c1]

        따라서:
            distance 계산 X
            nearest search X
            argmin X
            Gumbel X
            STE X
        """
        # category ID를 Q1 codebook과 같은 device로 이동 / nn.Embedding index는 long 타입이어야 하므로 변환
        fixed_ids = fixed_ids.to(device=self.device, dtype=torch.long,)
        # [B,1] 등으로 들어와도 강제로 [B] 형태로 평탄화
        fixed_ids = fixed_ids.view(-1)
        # category 개수와 현재 batch 기사 개수가 같은지 확인
        if (fixed_ids.shape[0] != x.shape[0]):

            raise ValueError(
                "fixed_ids batch size must match x. "
                f"x batch={x.shape[0]}, "
                f"fixed_ids batch="
                f"{fixed_ids.shape[0]}."
            )

        # 음수 category ID가 있는지 검사
        if torch.any(fixed_ids < 0):
            raise ValueError( "Category IDs must be >= 0.")

        # Q1 codebook 범위를 넘어가는 category ID가 있는지 확인
        if torch.any(fixed_ids >= self.n_embed):
            max_id = (fixed_ids.max().item())

            raise ValueError(
                "Category ID exceeds Q1 codebook range. "
                f"max category ID={max_id}, "
                f"Q1 codebook size={self.n_embed}."
            )

        ids = fixed_ids  # Q1에서는 별도 argmin 없이 category ID 자체가 c1 Semantic ID

        emb = self.get_item_embeddings(ids) # category ID에 해당하는 Q1 vector lookup

        emb_out = emb # Q1은 STE 없이 실제 Q1 vector를 그대로 다음 residual 계산에 사용

        loss_output = (self.quantize_loss(query=x,value=emb,)) # Encoder 출력 h(a)와 카테고리에 대응하는 q1

        return QuantizeOutput(  # Q1 quantization 결과 반환
            embeddings=emb_out,  # q1 vector
            ids=ids,  # category ID = c1
            codebook_loss=(
                loss_output.codebook_loss  # L_cb1
            ),
            commitment_loss=(
                loss_output.commitment_loss  # L_com1
            ),
        )

    # Q2 / Q3: Nearest Code Quantization

    def _forward_nearest(
        self,
        x: Tensor,  # Q2에서는 r1, Q3에서는 r2
        temperature: float,  # Gumbel-Softmax를 사용할 때의 temperature
    ) -> QuantizeOutput:

        # 전체 codebook에 선택적 projection/normalization 적용
        codebook = self.out_proj(self.embedding.weight)

        # 현재 residual과 모든 code 사이 거리 계산
        dist = self._compute_distance(x=x,codebook=codebook,)

        ids = (
            dist
            .detach() # argmin 선택 과정에는 gradient를 흘리지 않음
            .argmin(dim=1) # 각 article/query마다 가장 작은 distance를 가진 code 선택
        )

        # ========================================================
        # Training Mode
        # ========================================================
        # model.train() 상태라면 training용 gradient 처리 수행
        if self.training:
            # Gumbel Softmax
            if (self.forward_mode== QuantizeForwardMode.GUMBEL_SOFTMAX):
                weights = (gumbel_softmax(-dist,temperature=temperature,device=self.device,))

                # weighted combination
                emb = (weights @ codebook)
                emb_out = emb

            # STE
            elif (self.forward_mode == QuantizeForwardMode.STE):
                emb = (self.get_item_embeddings(ids))

                # ------------------------------------------------
                # Straight-Through Estimator
                #
                # Forward: emb_out == emb
                # 왜냐하면: x + (emb - x) = emb
                #
                # Backward:
                # detach된 부분에는 gradient가 없으므로
                # d emb_out / d x = 1
                # 따라서 quantization의 argmin을 우회하여
                # Encoder 쪽으로 gradient 전달.
                # ------------------------------------------------
                # Straight-Through Estimator 적용
                emb_out = (x + (emb - x).detach())

            # Rotation Trick
            elif (self.forward_mode== QuantizeForwardMode.ROTATION_TRICK):
                # nearest code vector lookup
                emb = (self.get_item_embeddings(ids))
                # query의 L2 norm 계산 / zero division 방지
                normalized_x = (x / (x.norm( dim=-1, keepdim=True,) + 1e-8))
                # code vector의 L2 norm 계산
                normalized_emb = (emb / (emb.norm(dim=-1,keepdim=True,) + 1e-8))
                # 정규화된 query와 code를 이용해 Rotation Trick 수행
                emb_out = (rotation_trick_transform(normalized_x,normalized_emb, x,))
                #norm 적용?
                emb_out = (emb_out * (torch.norm(emb,dim=1,keepdim=True,) / (torch.norm(x,dim=1,keepdim=True,) + 1e-6)).detach())

            else: # 정의되지 않은 forward mode 사용 시 오류
                raise ValueError(
                    "Unsupported forward mode: "
                    f"{self.forward_mode}"
                )

            loss_output = (
                self.quantize_loss( # 선택된 code와 query 사이 quantization loss 계산
                    query=x,  # Q2=r1, Q3=r2
                    value=emb, # 실제 선택된 code vector
                )
            )

        else: # model.eval() 상태, 즉 validation/inference 시
            emb = (self.get_item_embeddings(ids))
            emb_out = emb  # inference에서는 STE 없이 실제 code vector 그대로 사용
             # 필요시 validation/debug용 loss 계산
            loss_output = (self.quantize_loss(query=x,value=emb,))

        return QuantizeOutput(  # Q2 또는 Q3 결과 반환
            embeddings=emb_out,  # 선택된 code vector 또는 STE output
            ids=ids,  # c2 또는 c3 Semantic ID
            codebook_loss=(
                loss_output.codebook_loss
            ),
            commitment_loss=(
                loss_output.commitment_loss
            ),
        )

    # ============================================================
    # Forward
    # ============================================================

    def forward(
        self,
        x: Tensor,  # quantization할 입력, Q1=h, Q2=r1, Q3=r2
        temperature: float = 1.0,  # Gumbel 사용 시 temperature, STE에서는 사실상 사용 안 함
        fixed_ids: Optional[Tensor] = None,  # Q1에서 category ID를 전달하며 Q2/Q3에서는 None
    ) -> QuantizeOutput:

        if x.ndim != 2:  # 입력이 반드시 [batch_size, embed_dim]의 2차원 Tensor인지 확인

            raise ValueError(
                "Quantize input x must have shape "
                "[batch_size, embed_dim]. "
                f"Got {tuple(x.shape)}."
            )

        if (
            x.shape[-1]
            != self.embed_dim  # 실제 embedding dimension이 Quantize 생성 시 지정한 embed_dim과 같은지 확인
        ):

            raise ValueError(
                "Input embedding dimension mismatch. "
                f"Expected {self.embed_dim}, "
                f"got {x.shape[-1]}."
            )

        if fixed_ids is not None:  # category ID가 전달되었다면 Q1이라고 판단

            return (
                self._forward_fixed_ids(  # deterministic category lookup 수행
                    x=x,  # h(a)
                    fixed_ids=fixed_ids,  # category ID
                )
            )

        return (
            self._forward_nearest(  # fixed_ids가 없으면 Q2/Q3 nearest-code quantization 수행
                x=x,  # r1 또는 r2
                temperature=temperature,
            )
        )