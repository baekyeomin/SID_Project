from __future__ import annotations

from typing import NamedTuple

import gin
import torch
import torch.nn.functional as F

from torch import Tensor, nn


class TransformerLossOutput(NamedTuple):
    total_loss: Tensor

    c1_loss: Tensor
    c2_loss: Tensor
    c3_loss: Tensor
    c4_loss: Tensor

@gin.configurable
class TransformerLoss(nn.Module):
    def __init__(
        self,
        lambda_c1: float = 1.0,
        lambda_c2: float = 1.0,
        lambda_c3: float = 1.0,
        lambda_c4: float = 1.0,
    ) -> None:

        super().__init__()

        # 각 hierarchy loss의 가중치
        self.lambda_c1 = lambda_c1
        self.lambda_c2 = lambda_c2
        self.lambda_c3 = lambda_c3
        self.lambda_c4 = lambda_c4

    def forward(
        self,
        c1_logits: Tensor,
        c2_logits: Tensor,
        c3_logits: Tensor,
        c4_logits: Tensor,
        target_sids: Tensor,
    ) -> TransformerLossOutput:
        """
        각 hierarchy별 Cross Entropy Loss를 계산하고
        weighted sum으로 total loss를 반환한다.
        """

        if target_sids.ndim != 2:
            raise ValueError(
                "target_sids must have shape [B, 4]. "
                f"Received: {tuple(target_sids.shape)}"
            )

        if target_sids.shape[1] != 4:
            raise ValueError(
                "target_sids must contain exactly "
                "(c1, c2, c3, c4)."
            )

        target_c1 = target_sids[:, 0]
        target_c2 = target_sids[:, 1]
        target_c3 = target_sids[:, 2]
        target_c4 = target_sids[:, 3]

        c1_loss = F.cross_entropy(
            c1_logits,
            target_c1,
        )

        c2_loss = F.cross_entropy(
            c2_logits,
            target_c2,
        )

        c3_loss = F.cross_entropy(
            c3_logits,
            target_c3,
        )

        c4_loss = F.cross_entropy(
            c4_logits,
            target_c4,
        )

        # Total Loss
        total_loss = (
            self.lambda_c1 * c1_loss
            + self.lambda_c2 * c2_loss
            + self.lambda_c3 * c3_loss
            + self.lambda_c4 * c4_loss
        )

        return TransformerLossOutput(
            total_loss=total_loss,
            c1_loss=c1_loss,
            c2_loss=c2_loss,
            c3_loss=c3_loss,
            c4_loss=c4_loss,
        )

if __name__ == "__main__":

    batch_size = 2

    # 테스트용 가상의 model logits
    c1_logits = torch.randn(
        batch_size,
        18,
    )

    c2_logits = torch.randn(
        batch_size,
        128,
    )

    c3_logits = torch.randn(
        batch_size,
        512,
    )

    c4_logits = torch.randn(
        batch_size,
        32,
    )

    # 실제 target SID
    target_sids = torch.tensor(
        [
            [2, 6, 7, 0],
            [3, 4, 9, 1],
        ],
        dtype=torch.long,
    )

    # Loss 계산
    loss_fn = TransformerLoss(
        lambda_c1=1.0,
        lambda_c2=1.0,
        lambda_c3=1.0,
        lambda_c4=1.0,
    )

    loss_output = loss_fn(
        c1_logits=c1_logits,
        c2_logits=c2_logits,
        c3_logits=c3_logits,
        c4_logits=c4_logits,
        target_sids=target_sids,
    )

    print("Total loss:", loss_output.total_loss.item())
    print("C1 loss   :", loss_output.c1_loss.item())
    print("C2 loss   :", loss_output.c2_loss.item())
    print("C3 loss   :", loss_output.c3_loss.item())
    print("C4 loss   :", loss_output.c4_loss.item())