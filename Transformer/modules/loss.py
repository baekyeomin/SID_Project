from __future__ import annotations

from typing import NamedTuple

import gin
import torch
import torch.nn.functional as F

from torch import Tensor, nn


# ============================================================
# Constants
# ============================================================

NUM_CANDIDATES = 5
NUM_POSITIVES = 1


# ============================================================
# Loss output
# ============================================================

class TransformerLossOutput(NamedTuple):
    """
    Transformer 학습에서 사용하는 loss 값.
    """

    total_loss: Tensor
    preference_loss: Tensor


# ============================================================
# Transformer loss
# ============================================================

@gin.configurable
class TransformerLoss(nn.Module):

    def __init__(
        self,
        lambda_preference: float = 1.0,
    ) -> None:

        super().__init__()

        self.lambda_preference = lambda_preference

    # ========================================================
    # Forward
    # ========================================================

    def forward(
        self,
        candidate_scores: Tensor,
        candidate_labels: Tensor,
    ) -> TransformerLossOutput:

        # ====================================================
        # 1. Shape 검증
        # ====================================================

        # candidate_scores:
        # [B, 5]

        if candidate_scores.ndim != 2:

            raise ValueError(
                "candidate_scores must have shape [B,5]. "
                f"Received: {tuple(candidate_scores.shape)}"
            )

        if (
            candidate_scores.shape[1]
            != NUM_CANDIDATES
        ):

            raise ValueError(
                f"Expected {NUM_CANDIDATES} candidates, "
                f"but received "
                f"{candidate_scores.shape[1]}."
            )

        # candidate_labels:
        # [B, 5]

        if (
            candidate_labels.shape
            != candidate_scores.shape
        ):

            raise ValueError(
                "candidate_labels shape must match "
                "candidate_scores shape."
            )

        # ====================================================
        # 2. Label type 변환
        # ====================================================

        candidate_labels = (
            candidate_labels.float()
        )

        # ====================================================
        # 3. Label 값 검증
        #
        # label은 반드시 0 또는 1
        # ====================================================

        valid_labels = (
            (candidate_labels == 0)
            | (candidate_labels == 1)
        )

        if not torch.all(valid_labels):

            raise ValueError(
                "candidate_labels must contain "
                "only 0 or 1."
            )

        # ====================================================
        # 4. 각 row에 positive가 정확히 1개인지 확인
        # ====================================================

        positive_counts = (
            candidate_labels.sum(
                dim=1
            )
        )

        if not torch.all(
            positive_counts == NUM_POSITIVES
        ):

            raise ValueError(
                "Each sample must contain exactly "
                f"{NUM_POSITIVES} positive candidate."
            )

        # ====================================================
        # 5. Positive candidate 위치 찾기
        #
        # 예:
        #
        # labels
        # [0, 0, 1, 0, 0]
        #
        #        ↓
        #
        # target_index = 2
        # ====================================================

        target_indices = (
            candidate_labels.argmax(
                dim=1
            )
            .long()
        )

        # ====================================================
        # 6. Preference loss
        #
        # candidate_scores:
        #
        # [-5.2, -3.1, -1.8, -4.7, -6.0]
        #
        # softmax:
        #
        # 후보 5개 사이의 상대 확률 계산
        #
        # Cross Entropy:
        #
        # positive 후보의 확률이 높아지도록 학습
        # ====================================================

        preference_loss = F.cross_entropy(
            candidate_scores,
            target_indices,
        )

        # ====================================================
        # 7. Total loss
        #
        # tie loss는 완전히 제거
        # ====================================================

        total_loss = (
            self.lambda_preference
            * preference_loss
        )

        # ====================================================
        # 8. 반환
        # ====================================================

        return TransformerLossOutput(
            total_loss=total_loss,
            preference_loss=preference_loss,
        )


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # Example candidate scores
    #
    # Batch size = 2
    # 후보 = 5개
    #
    # score는 log probability 기반이므로
    # 음수여도 정상
    # ========================================================

    candidate_scores = torch.tensor(
        [
            [
                -5.0,
                -3.0,
                -1.5,
                -4.0,
                -6.0,
            ],
            [
                -2.0,
                -4.0,
                -5.0,
                -1.0,
                -3.0,
            ],
        ],
        dtype=torch.float32,
    )

    # ========================================================
    # Positive label
    #
    # 첫 번째 sample:
    # candidate index 2가 positive
    #
    # 두 번째 sample:
    # candidate index 3이 positive
    # ========================================================

    candidate_labels = torch.tensor(
        [
            [
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
            ],
            [
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
            ],
        ],
        dtype=torch.float32,
    )

    # ========================================================
    # Loss function
    # ========================================================

    loss_fn = TransformerLoss(
        lambda_preference=1.0,
    )

    # ========================================================
    # Loss 계산
    # ========================================================

    loss_output = loss_fn(
        candidate_scores=candidate_scores,
        candidate_labels=candidate_labels,
    )

    # ========================================================
    # 결과 출력
    # ========================================================

    print(
        "Total loss:",
        loss_output.total_loss.item(),
    )

    print(
        "Preference loss:",
        loss_output.preference_loss.item(),
    )

    # ========================================================
    # 참고:
    # 사람이 확인하기 위한 candidate probability
    # ========================================================

    candidate_probs = torch.softmax(
        candidate_scores,
        dim=1,
    )

    print()
    print(
        "Candidate probabilities:"
    )

    print(
        candidate_probs
    )

    print()
    print(
        "Probability sums:",
        candidate_probs.sum(dim=1),
    )