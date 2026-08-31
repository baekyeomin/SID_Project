import torch
import torch.nn.functional as F

from torch import Tensor


def rotation_trick_transform(
    u: Tensor,
    q: Tensor,
    e: Tensor,
) -> Tensor:
    """
    Rotation Trick 기반 gradient transform.

    Parameters
    ----------
    u : Tensor
        정규화된 quantizer input.
        Q2에서는 normalized r1,
        Q3에서는 normalized r2.

        shape: [batch_size, embed_dim]

    q : Tensor
        정규화된 selected code vector.

        shape: [batch_size, embed_dim]

    e : Tensor
        원래 quantizer input.

        Q2에서는 r1,
        Q3에서는 r2.

        shape: [batch_size, embed_dim]

    Returns
    -------
    Tensor
        Rotation Trick이 적용된 quantized output.

        shape: [batch_size, embed_dim]

    Notes
    -----
    forward에서는 선택된 code 방향으로 mapping되지만,
    backward에서는 STE처럼 단순 identity gradient를 쓰지 않고
    input과 code 사이의 angle 정보를 반영한 gradient를 전달한다.
    """

    # --------------------------------------------------------
    # Input shape sanity check
    # --------------------------------------------------------

    if u.ndim != 2:
        raise ValueError(
            "u must have shape "
            "[batch_size, embed_dim]. "
            f"Got {tuple(u.shape)}."
        )

    if q.ndim != 2:
        raise ValueError(
            "q must have shape "
            "[batch_size, embed_dim]. "
            f"Got {tuple(q.shape)}."
        )

    if e.ndim != 2:
        raise ValueError(
            "e must have shape "
            "[batch_size, embed_dim]. "
            f"Got {tuple(e.shape)}."
        )

    if (
        u.shape != q.shape
        or u.shape != e.shape
    ):
        raise ValueError(
            "u, q, and e must have "
            "the same shape. "
            f"u={tuple(u.shape)}, "
            f"q={tuple(q.shape)}, "
            f"e={tuple(e.shape)}."
        )

    # --------------------------------------------------------
    # e:
    #
    # [B, D]
    #   ↓
    # [B, 1, D]
    #
    # batch별 matrix multiplication을 하기 위해
    # row vector 형태로 차원을 하나 추가.
    # --------------------------------------------------------

    e_row = e.unsqueeze(1)

    # --------------------------------------------------------
    # w = normalize(u + q)
    #
    # u:
    # quantizer input 방향
    #
    # q:
    # selected code 방향
    #
    # u와 q 사이의 rotation을 정의하는
    # Householder reflection vector.
    #
    # w 자체는 gradient 계산 대상이 아니므로 detach.
    # --------------------------------------------------------

    w = F.normalize(
        u + q,
        p=2,
        dim=1,
        eps=1e-6,
    ).detach()

    # --------------------------------------------------------
    # Matrix multiplication을 위한 shape 변환
    #
    # w_column : [B, D, 1]
    # w_row    : [B, 1, D]
    # --------------------------------------------------------

    w_column = (
        w
        .unsqueeze(2)
    )

    w_row = (
        w
        .unsqueeze(1)
    )

    # --------------------------------------------------------
    # u / q도 outer-product 계산을 위해
    # column / row 형태로 변환.
    #
    # Rotation matrix를 직접 gradient로 학습하는 것이
    # 아니므로 detach.
    # --------------------------------------------------------

    u_column = (
        u
        .unsqueeze(2)
        .detach()
    )

    q_row = (
        q
        .unsqueeze(1)
        .detach()
    )

    # --------------------------------------------------------
    # Rotation Trick
    #
    # e
    # - 2 e w w^T
    # + 2 e u q^T
    #
    # shape은 모두 [B, 1, D].
    #
    # forward에서는 e의 방향을
    # selected code q 방향으로 회전시키고,
    #
    # backward에서는 input-code angle에 따른
    # gradient transformation을 제공한다.
    # --------------------------------------------------------

    rotated = (
        e_row
        - 2.0
        * (
            e_row
            @ w_column
            @ w_row
        )
        + 2.0
        * (
            e_row
            @ u_column
            @ q_row
        )
    )

    # --------------------------------------------------------
    # [B, 1, D]
    #     ↓
    # [B, D]
    #
    # squeeze() 전체가 아니라 squeeze(1)을 써야
    # batch_size=1이어도 batch 차원이 사라지지 않는다.
    # --------------------------------------------------------

    rotated = (
        rotated
        .squeeze(1)
    )

    return rotated 