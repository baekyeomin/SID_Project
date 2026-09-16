from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List

import gin
import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import DataLoader

from data.sequence import (
    NewsSequenceDataset,
    collate_news_sequences,
)

from modules.model import NewsEncoderDecoderTransformer


BASE_DIR = Path(__file__).resolve().parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return BASE_DIR / path_obj


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def normalize_article_id(article_id):
    if article_id is None:
        return None

    try:
        if pd.isna(article_id):
            return None
    except (TypeError, ValueError):
        pass

    return str(article_id)


def load_checkpoint(
    checkpoint_path: Path,
    model: NewsEncoderDecoderTransformer,
    device: torch.device,
) -> dict:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:
        model.load_state_dict(
            checkpoint
        )

    return checkpoint


def has_positive_negative_collision(
    candidate_sids: Tensor,
    candidate_labels: Tensor,
) -> bool:

    positive_indices = torch.where(
        candidate_labels == 1
    )[0]

    negative_indices = torch.where(
        candidate_labels == 0
    )[0]

    if (
        positive_indices.numel() == 0
        or negative_indices.numel() == 0
    ):
        return False

    positive_sids = candidate_sids[
        positive_indices
    ]

    negative_sids = candidate_sids[
        negative_indices
    ]

    for positive_sid in positive_sids:

        collision = (
            negative_sids
            == positive_sid.unsqueeze(0)
        ).all(dim=-1)

        if collision.any():
            return True

    return False


def get_collision_group_sizes(
    candidate_sids: Tensor,
) -> Dict[tuple, int]:

    counts: Dict[tuple, int] = {}

    for sid in candidate_sids:

        key = tuple(
            int(v)
            for v in sid.tolist()
        )

        counts[key] = (
            counts.get(key, 0)
            + 1
        )

    return counts


def rank_candidates(
    candidate_scores: Tensor,
    tie_scores: Tensor,
    candidate_sids: Tensor,
) -> List[int]:

    num_candidates = (
        candidate_scores.shape[0]
    )

    order = sorted(
        range(num_candidates),
        key=lambda idx: float(
            candidate_scores[
                idx
            ].item()
        ),
        reverse=True,
    )

    sid_to_positions: Dict[
        tuple,
        List[int],
    ] = {}

    for position, candidate_idx in enumerate(
        order
    ):

        sid = tuple(
            int(v)
            for v in candidate_sids[
                candidate_idx
            ].tolist()
        )

        sid_to_positions.setdefault(
            sid,
            [],
        ).append(
            position
        )

    for positions in sid_to_positions.values():

        if len(positions) <= 1:
            continue

        candidate_indices = [
            order[position]
            for position in positions
        ]

        candidate_indices.sort(
            key=lambda idx: float(
                tie_scores[
                    idx
                ].item()
            ),
            reverse=True,
        )

        for position, candidate_idx in zip(
            positions,
            candidate_indices,
        ):

            order[position] = (
                candidate_idx
            )

    return order


@torch.no_grad()
def predict(
    model: NewsEncoderDecoderTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    Dict[str, float],
]:

    model.eval()

    prediction_rows = []

    num_impressions = 0
    num_correct = 0

    num_collision_impressions = 0
    num_collision_correct = 0

    sample_index = 0

    for batch_idx, batch in enumerate(
        dataloader
    ):

        history_sids = batch[
            "history_sids"
        ].to(
            device,
            non_blocking=True,
        )

        history_mask = batch[
            "history_mask"
        ].to(
            device,
            non_blocking=True,
        )

        candidate_sids = batch[
            "candidate_sids"
        ].to(
            device,
            non_blocking=True,
        )

        candidate_c4 = batch[
            "candidate_c4"
        ].to(
            device,
            non_blocking=True,
        )

        candidate_labels = batch[
            "candidate_labels"
        ].to(
            device,
            non_blocking=True,
        )

        candidate_mask = batch[
            "candidate_mask"
        ].to(
            device,
            non_blocking=True,
        )

        model_output = model(
            history_sids=
                history_sids,

            history_mask=
                history_mask,

            candidate_sids=
                candidate_sids,

            candidate_c4=
                candidate_c4,
        )

        batch_size = (
            candidate_sids.shape[0]
        )

        for i in range(
            batch_size
        ):

            valid = candidate_mask[
                i
            ]

            sids = candidate_sids[
                i
            ][valid]

            c4_values = candidate_c4[
                i
            ][valid]

            labels = candidate_labels[
                i
            ][valid]

            scores = (
                model_output
                .candidate_scores[
                    i
                ][valid]
            )

            c1_log_probs = (
                model_output
                .c1_log_probs[
                    i
                ][valid]
            )

            c2_log_probs = (
                model_output
                .c2_log_probs[
                    i
                ][valid]
            )

            c3_log_probs = (
                model_output
                .c3_log_probs[
                    i
                ][valid]
            )

            tie_scores = (
                model_output
                .tie_scores[
                    i
                ][valid]
            )

            num_candidates = (
                sids.shape[0]
            )

            if num_candidates == 0:
                continue

            article_ids = (
                batch[
                    "candidate_article_ids"
                ][i]
            )

            if article_ids is None:
                raise ValueError(
                    "candidate_article_ids is required."
                )

            if (
                len(article_ids)
                != num_candidates
            ):
                raise ValueError(
                    f"candidate_article_ids length mismatch "
                    f"at sample {sample_index}: "
                    f"{len(article_ids)} vs "
                    f"{num_candidates}"
                )

            order = rank_candidates(
                candidate_scores=
                    scores,

                tie_scores=
                    tie_scores,

                candidate_sids=
                    sids,
            )

            top_index = order[
                0
            ]

            is_correct = bool(
                labels[
                    top_index
                ].item()
                == 1
            )

            collision = (
                has_positive_negative_collision(
                    candidate_sids=
                        sids,

                    candidate_labels=
                        labels,
                )
            )

            num_impressions += 1

            if is_correct:
                num_correct += 1

            if collision:

                num_collision_impressions += 1

                if is_correct:
                    num_collision_correct += 1

            collision_group_sizes = (
                get_collision_group_sizes(
                    sids
                )
            )

            rank_by_candidate = {
                candidate_idx: rank
                for rank, candidate_idx
                in enumerate(
                    order,
                    start=1,
                )
            }

            impression_id = (
                batch[
                    "impression_ids"
                ][i]
            )

            user_id = (
                batch[
                    "user_ids"
                ][i]
            )

            impression_time = (
                batch[
                    "impression_times"
                ][i]
            )

            for candidate_idx in range(
                num_candidates
            ):

                sid = tuple(
                    int(v)
                    for v in sids[
                        candidate_idx
                    ].tolist()
                )

                article_id = (
                    normalize_article_id(
                        article_ids[
                            candidate_idx
                        ]
                    )
                )

                prediction_rows.append(
                    {
                        "sample_index":
                            sample_index,

                        "batch_index":
                            batch_idx,

                        "impression_id":
                            impression_id,

                        "user_id":
                            user_id,

                        "impression_time":
                            impression_time,

                        "article_id":
                            article_id,

                        "c1":
                            sid[0],

                        "c2":
                            sid[1],

                        "c3":
                            sid[2],

                        "c4":
                            int(
                                c4_values[
                                    candidate_idx
                                ].item()
                            ),

                        "label":
                            float(
                                labels[
                                    candidate_idx
                                ].item()
                            ),

                        "candidate_score":
                            float(
                                scores[
                                    candidate_idx
                                ].item()
                            ),

                        "c1_log_prob":
                            float(
                                c1_log_probs[
                                    candidate_idx
                                ].item()
                            ),

                        "c2_log_prob":
                            float(
                                c2_log_probs[
                                    candidate_idx
                                ].item()
                            ),

                        "c3_log_prob":
                            float(
                                c3_log_probs[
                                    candidate_idx
                                ].item()
                            ),

                        "tie_score":
                            float(
                                tie_scores[
                                    candidate_idx
                                ].item()
                            ),

                        "collision_group_size":
                            collision_group_sizes[
                                sid
                            ],

                        "rank":
                            rank_by_candidate[
                                candidate_idx
                            ],

                        "is_top1":
                            rank_by_candidate[
                                candidate_idx
                            ]
                            == 1,

                        "top1_correct":
                            is_correct,

                        "has_positive_negative_collision":
                            collision,
                    }
                )

            sample_index += 1

        print(
            f"Processed batch "
            f"{batch_idx + 1}/"
            f"{len(dataloader)}"
        )

    top1_accuracy = (
        num_correct
        / num_impressions
        if num_impressions > 0
        else 0.0
    )

    collision_rate = (
        num_collision_impressions
        / num_impressions
        if num_impressions > 0
        else 0.0
    )

    collision_top1_accuracy = (
        num_collision_correct
        / num_collision_impressions
        if num_collision_impressions > 0
        else float("nan")
    )

    metrics = {
        "num_impressions":
            num_impressions,

        "num_correct":
            num_correct,

        "top1_accuracy":
            top1_accuracy,

        "num_collision_impressions":
            num_collision_impressions,

        "collision_rate":
            collision_rate,

        "num_collision_correct":
            num_collision_correct,

        "collision_top1_accuracy":
            collision_top1_accuracy,
    }

    return (
        pd.DataFrame(
            prediction_rows
        ),
        metrics,
    )


def print_metrics(
    metrics: Dict[str, float],
) -> None:

    print()

    print(
        "Impressions:",
        f"{int(metrics['num_impressions']):,}",
    )

    print(
        "Correct Top-1:",
        f"{int(metrics['num_correct']):,}",
    )

    print(
        "Top-1 Accuracy:",
        f"{metrics['top1_accuracy']:.4%}",
    )

    print(
        "Collision impressions:",
        f"{int(metrics['num_collision_impressions']):,}",
    )

    print(
        "Collision rate:",
        f"{metrics['collision_rate']:.4%}",
    )

    collision_accuracy = (
        metrics[
            "collision_top1_accuracy"
        ]
    )

    if math.isnan(
        collision_accuracy
    ):

        print(
            "Collision Top-1 Accuracy: N/A"
        )

    else:

        print(
            "Collision Top-1 Accuracy:",
            f"{collision_accuracy:.4%}",
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate news ranking "
            "using Semantic-ID preference scores."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default=(
            "datasets/ebnerd/"
            "test_sequences.parquet"
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "test_candidate_scores.parquet"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    test_path = resolve_path(
        args.test_path
    )

    output_path = resolve_path(
        args.output_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gin config not found:\n"
            f"{config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{test_path}"
        )

    gin.parse_config_file(
        str(config_path),
        skip_unknown=True,
    )

    device = get_device()

    print()
    print(
        "Candidate Ranking Evaluation"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Test path:",
        test_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    test_dataset = (
        NewsSequenceDataset(
            parquet_path=
                str(test_path),
        )
    )

    test_loader = DataLoader(
        dataset=
            test_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        num_workers=
            args.num_workers,

        collate_fn=
            collate_news_sequences,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    print(
        "Test samples:",
        f"{len(test_dataset):,}",
    )

    model = (
        NewsEncoderDecoderTransformer()
        .to(device)
    )

    checkpoint = load_checkpoint(
        checkpoint_path=
            checkpoint_path,

        model=
            model,

        device=
            device,
    )

    print(
        "Checkpoint loaded."
    )

    if (
        isinstance(checkpoint, dict)
        and "epoch" in checkpoint
    ):

        print(
            "Epoch:",
            checkpoint[
                "epoch"
            ],
        )

    predictions_df, metrics = predict(
        model=
            model,

        dataloader=
            test_loader,

        device=
            device,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_parquet(
        output_path,
        index=False,
    )

    print_metrics(
        metrics
    )

    print()

    print(
        "Candidate scores saved:",
        output_path,
    )


if __name__ == "__main__":
    main()