from __future__ import annotations

from typing import NamedTuple, Optional

import gin
import torch
import torch.nn.functional as F

from torch import Tensor, nn


class TransformerLossOutput(NamedTuple):
    total_loss: Tensor
    preference_loss: Tensor
    tie_loss: Tensor


@gin.configurable
class TransformerLoss(nn.Module):
    def __init__(
        self,
        lambda_preference: float = 1.0,
        lambda_tie: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_preference = lambda_preference
        self.lambda_tie = lambda_tie

    def forward(
        self,
        candidate_scores: Tensor,
        tie_scores: Optional[Tensor],
        candidate_sids: Tensor,
        candidate_labels: Tensor,
        candidate_mask: Tensor,
    ) -> TransformerLossOutput:

        if candidate_scores.ndim != 2:
            raise ValueError(
                "candidate_scores must have shape [B,C]. "
                f"Received: {tuple(candidate_scores.shape)}"
            )

        if candidate_labels.shape != candidate_scores.shape:
            raise ValueError(
                "candidate_labels shape must match candidate_scores."
            )

        if candidate_mask.shape != candidate_scores.shape:
            raise ValueError(
                "candidate_mask shape must match candidate_scores."
            )

        if candidate_sids.ndim != 3 or candidate_sids.shape[-1] != 3:
            raise ValueError(
                "candidate_sids must have shape [B,C,3]."
            )

        if candidate_sids.shape[:2] != candidate_scores.shape:
            raise ValueError(
                "candidate_sids [B,C] must match candidate_scores [B,C]."
            )

        if tie_scores is not None and tie_scores.shape != candidate_scores.shape:
            raise ValueError(
                "tie_scores shape must match candidate_scores."
            )

        if self.lambda_tie > 0 and tie_scores is None:
            raise ValueError(
                "tie_scores is required when lambda_tie > 0."
            )

        candidate_labels = candidate_labels.float()
        candidate_mask = candidate_mask.bool()

        valid_scores = candidate_scores[candidate_mask]
        valid_labels = candidate_labels[candidate_mask]

        if valid_scores.numel() == 0:
            raise ValueError(
                "No valid candidates were found in the batch."
            )

        if not torch.all(
            (valid_labels == 0) | (valid_labels == 1)
        ):
            raise ValueError(
                "candidate_labels must contain only 0 or 1."
            )

        candidate_probs = torch.exp(valid_scores)
        candidate_probs = candidate_probs.clamp(
            min=1e-7,
            max=1.0 - 1e-7,
        )

        preference_loss = F.binary_cross_entropy(
            candidate_probs,
            valid_labels,
        )

        tie_pair_losses = []

        if tie_scores is not None:
            batch_size = candidate_scores.shape[0]

            for batch_idx in range(batch_size):
                valid = candidate_mask[batch_idx]

                sids = candidate_sids[batch_idx][valid]
                labels = candidate_labels[batch_idx][valid]
                scores = tie_scores[batch_idx][valid]

                positive_indices = torch.where(
                    labels == 1
                )[0]

                negative_indices = torch.where(
                    labels == 0
                )[0]

                if (
                    positive_indices.numel() == 0
                    or negative_indices.numel() == 0
                ):
                    continue

                negative_sids = sids[negative_indices]
                negative_scores = scores[negative_indices]

                for positive_idx in positive_indices:
                    positive_sid = sids[positive_idx]
                    positive_score = scores[positive_idx]

                    collision_mask = (
                        negative_sids
                        == positive_sid.unsqueeze(0)
                    ).all(dim=-1)

                    if not collision_mask.any():
                        continue

                    collision_negative_scores = (
                        negative_scores[collision_mask]
                    )

                    pair_loss = F.softplus(
                        collision_negative_scores
                        - positive_score
                    )

                    tie_pair_losses.append(
                        pair_loss
                    )

        if tie_pair_losses:
            tie_loss = torch.cat(
                tie_pair_losses
            ).mean()
        else:
            tie_loss = candidate_scores.new_zeros(())

        total_loss = (
            self.lambda_preference
            * preference_loss
            + self.lambda_tie
            * tie_loss
        )

        return TransformerLossOutput(
            total_loss=total_loss,
            preference_loss=preference_loss,
            tie_loss=tie_loss,
        )


if __name__ == "__main__":
    candidate_scores = torch.tensor(
        [
            [-0.5, -0.5, -4.0, -6.0],
            [-1.0, -1.0, -5.0, 0.0],
        ],
        dtype=torch.float32,
    )

    candidate_sids = torch.tensor(
        [
            [
                [4, 72, 301],
                [4, 72, 301],
                [3, 15, 100],
                [8, 22, 50],
            ],
            [
                [2, 31, 120],
                [2, 31, 120],
                [5, 10, 80],
                [0, 0, 0],
            ],
        ],
        dtype=torch.long,
    )

    candidate_labels = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    candidate_mask = torch.tensor(
        [
            [True, True, True, True],
            [True, True, True, False],
        ],
        dtype=torch.bool,
    )

    tie_scores = torch.tensor(
        [
            [1.8, 0.4, 0.3, 0.1],
            [1.2, 0.7, 0.2, 0.0],
        ],
        dtype=torch.float32,
    )

    loss_fn = TransformerLoss(
        lambda_preference=1.0,
        lambda_tie=0.1,
    )

    loss_output = loss_fn(
        candidate_scores=candidate_scores,
        tie_scores=tie_scores,
        candidate_sids=candidate_sids,
        candidate_labels=candidate_labels,
        candidate_mask=candidate_mask,
    )

    print(
        "Total loss:",
        loss_output.total_loss.item(),
    )

    print(
        "Preference loss:",
        loss_output.preference_loss.item(),
    )

    print(
        "Tie loss:",
        loss_output.tie_loss.item(),
    )