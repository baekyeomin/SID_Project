from __future__ import annotations

from typing import NamedTuple

import gin
import torch
import torch.nn.functional as F

from torch import Tensor, nn


# Constants
NUM_CANDIDATES = 5
NUM_POSITIVES = 1

class TransformerLossOutput(NamedTuple):
    total_loss: Tensor
    preference_loss: Tensor


@gin.configurable
class TransformerLoss(nn.Module):

    def __init__(self, lambda_preference: float = 1.0,) -> None:
        super().__init__()
        self.lambda_preference = lambda_preference

    # Forward
    def forward(self, candidate_scores: Tensor, candidate_labels: Tensor) -> TransformerLossOutput:

        # 1. Shape 검증
        # candidate_scores: [B, 5]
        if candidate_scores.ndim != 2:
            raise ValueError(
                "candidate_scores must have shape [B,5]. "
                f"Received: {tuple(candidate_scores.shape)}"
            )

        if (candidate_scores.shape[1] != NUM_CANDIDATES):
            raise ValueError(
                f"Expected {NUM_CANDIDATES} candidates, "
                f"but received "
                f"{candidate_scores.shape[1]}."
            )

        # candidate_labels: [B, 5]
        if (candidate_labels.shape != candidate_scores.shape):
            raise ValueError("candidate_labels shape must match candidate_scores shape.")

        # 2. Label type 변환
        candidate_labels = (candidate_labels.float())

        # 3. Label 값 검증: label은 반드시 0 또는 1
        valid_labels = ((candidate_labels == 0) | (candidate_labels == 1))
        if not torch.all(valid_labels):
            raise ValueError("candidate_labels must contain only 0 or 1.")

        # 4. 각 row에 positive가 1개인지 확인
        positive_counts = candidate_labels.sum(dim=1)

        if not torch.all(positive_counts == NUM_POSITIVES):
            raise ValueError(
                "Each sample must contain exactly "
                f"{NUM_POSITIVES} positive candidate."
            )

        # 5. Positive candidate 인덱스 찾기
        target_indices = candidate_labels.argmax(dim=1).long()

        # 6. Preference loss
        # 후보 5개의 score에 Cross Entropy를 적용하여
        # positive 후보의 score가 가장 높아지도록 학습
        preference_loss = F.cross_entropy(candidate_scores, target_indices)

        # 7. Total loss
        total_loss = (self.lambda_preference * preference_loss)

        return TransformerLossOutput(
            total_loss=total_loss,
            preference_loss=preference_loss,
        )


# Simple test
if __name__ == "__main__":

    # ========================================================
    # Example candidate scores
    #
    # Batch size = 2
    # 후보 = 5개
    #
    # score는 log probability 기반이므로 음수여도 정상
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

    # Loss function
    loss_fn = TransformerLoss(lambda_preference=1.0,)

    # Loss 계산
    loss_output = loss_fn(candidate_scores=candidate_scores, candidate_labels=candidate_labels,)

    print("Total loss:", loss_output.total_loss.item())
    print("Preference loss:", loss_output.preference_loss.item())

    candidate_probs = torch.softmax(candidate_scores, dim=1)

    print()
    print("Candidate probabilities:")
    print(candidate_probs)

    print()
    print("Probability sums:", candidate_probs.sum(dim=1))      
        