from torch import Tensor  # Tensor 타입 힌트
from torch.nn import functional as F  # L2 normalization 사용
from einops import rearrange  # 행렬곱을 위한 Tensor 차원 변환


def rotation_trick_transform(
    u: Tensor,  # 정규화된 query vector
    q: Tensor,  # 정규화된 quantized code vector
    e: Tensor,  # gradient를 변환할 원본 query vector
) -> Tensor:
    """Rotation Trick gradient transformation. Reference: arXiv:2410.06424, Sec. 4.2."""

    e = rearrange(e, "b d -> b 1 d")  # [B,D] -> [B,1,D]로 변환

    w = F.normalize(
        u + q,  # query와 code의 합 방향
        p=2,  # L2 norm 사용
        dim=1,  # embedding 차원 기준 normalization
        eps=1e-6,  # 0으로 나누는 문제 방지
    ).detach()  # w 자체로는 gradient를 전달하지 않음

    transformed = (
        e
        - 2
        * (
            e
            @ rearrange(w, "b d -> b d 1")  # w를 [B,D,1]로 변환
            @ rearrange(w, "b d -> b 1 d")  # w를 [B,1,D]로 변환
        )
        + 2
        * (
            e
            @ rearrange(u, "b d -> b d 1").detach()  # u를 column 형태로 바꾸고 gradient 차단
            @ rearrange(q, "b d -> b 1 d").detach()  # q를 row 형태로 바꾸고 gradient 차단
        )
    )  # Rotation Trick 수식에 따라 e의 gradient 전달 방향 변환

    return transformed.squeeze(1)  # [B,1,D] -> [B,D]로 복원