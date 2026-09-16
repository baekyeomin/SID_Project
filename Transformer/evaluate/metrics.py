'''
Top-1 Accuracy: 최종 1위가 클릭 기사인지. 교수님이 말씀하신 가장 직접적인 지표.
AUC: positive가 negative보다 위에 오는 비율.
MRR: 첫 positive가 얼마나 위에 있는지.
nDCG@5 / nDCG@10: top-K 전체 순위 품질.
Collision Rate: positive와 negative가 같은 (c1,c2,c3)인 impression 비율.
Collision Top-1 / AUC / MRR / nDCG: collision이 실제 발생한 어려운 케이스에서 성능.
Tie Pair Accuracy: 같은 (c1,c2,c3)인 positive-negative 쌍에서 tie_score(pos) > tie_score(neg)가 얼마나 자주 성립하는지.
Semantic AUC: tie 적용 전 candidate_score만 사용한 AUC. 최종 AUC와 비교하면 tie-breaking 효과를 볼 수 있음.
'''


from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent.parent


def resolve_path(path: str) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return BASE_DIR / path_obj


def auc_from_scores(
    labels: np.ndarray,
    scores: np.ndarray,
) -> Optional[float]:
    positive_scores = scores[
        labels == 1
    ]

    negative_scores = scores[
        labels == 0
    ]

    if (
        len(positive_scores) == 0
        or len(negative_scores) == 0
    ):
        return None

    comparisons = (
        positive_scores[:, None]
        - negative_scores[None, :]
    )

    wins = (
        comparisons > 0
    ).sum()

    ties = (
        comparisons == 0
    ).sum()

    total_pairs = (
        len(positive_scores)
        * len(negative_scores)
    )

    return float(
        (
            wins
            + 0.5 * ties
        )
        / total_pairs
    )


def auc_from_ranks(
    labels: np.ndarray,
    ranks: np.ndarray,
) -> Optional[float]:
    positive_ranks = ranks[
        labels == 1
    ]

    negative_ranks = ranks[
        labels == 0
    ]

    if (
        len(positive_ranks) == 0
        or len(negative_ranks) == 0
    ):
        return None

    comparisons = (
        positive_ranks[:, None]
        - negative_ranks[None, :]
    )

    wins = (
        comparisons < 0
    ).sum()

    ties = (
        comparisons == 0
    ).sum()

    total_pairs = (
        len(positive_ranks)
        * len(negative_ranks)
    )

    return float(
        (
            wins
            + 0.5 * ties
        )
        / total_pairs
    )


def reciprocal_rank(
    labels: np.ndarray,
    ranks: np.ndarray,
) -> Optional[float]:
    positive_ranks = ranks[
        labels == 1
    ]

    if len(positive_ranks) == 0:
        return None

    first_positive_rank = int(
        positive_ranks.min()
    )

    return (
        1.0
        / first_positive_rank
    )


def ndcg_at_k(
    labels: np.ndarray,
    ranks: np.ndarray,
    k: int,
) -> Optional[float]:
    num_positive = int(
        (labels == 1).sum()
    )

    if num_positive == 0:
        return None

    dcg = 0.0

    for label, rank in zip(
        labels,
        ranks,
    ):
        if (
            label == 1
            and rank <= k
        ):
            dcg += (
                1.0
                / np.log2(
                    rank + 1
                )
            )

    ideal_count = min(
        num_positive,
        k,
    )

    idcg = sum(
        1.0
        / np.log2(
            rank + 1
        )
        for rank in range(
            1,
            ideal_count + 1,
        )
    )

    if idcg == 0:
        return None

    return float(
        dcg / idcg
    )


def top1_accuracy(
    labels: np.ndarray,
    ranks: np.ndarray,
) -> Optional[float]:
    if (
        labels == 1
    ).sum() == 0:
        return None

    top1_labels = labels[
        ranks == 1
    ]

    if len(top1_labels) == 0:
        return None

    return float(
        np.any(
            top1_labels == 1
        )
    )


def has_collision(
    group: pd.DataFrame,
) -> bool:
    positive = group[
        group["label"] == 1
    ]

    negative = group[
        group["label"] == 0
    ]

    if (
        len(positive) == 0
        or len(negative) == 0
    ):
        return False

    positive_sids = set(
        zip(
            positive["c1"],
            positive["c2"],
            positive["c3"],
        )
    )

    negative_sids = set(
        zip(
            negative["c1"],
            negative["c2"],
            negative["c3"],
        )
    )

    return bool(
        positive_sids
        & negative_sids
    )


def evaluate_ranking(
    df: pd.DataFrame,
    group_column: str,
) -> Dict[str, float]:
    auc_values = []
    semantic_auc_values = []

    mrr_values = []
    ndcg5_values = []
    ndcg10_values = []
    top1_values = []

    num_impressions = 0

    for _, group in df.groupby(
        group_column,
        sort=False,
    ):
        labels = (
            group["label"]
            .to_numpy(
                dtype=np.int64
            )
        )

        ranks = (
            group["rank"]
            .to_numpy(
                dtype=np.int64
            )
        )

        candidate_scores = (
            group["candidate_score"]
            .to_numpy(
                dtype=np.float64
            )
        )

        num_impressions += 1

        auc = auc_from_ranks(
            labels,
            ranks,
        )

        semantic_auc = auc_from_scores(
            labels,
            candidate_scores,
        )

        mrr = reciprocal_rank(
            labels,
            ranks,
        )

        ndcg5 = ndcg_at_k(
            labels,
            ranks,
            5,
        )

        ndcg10 = ndcg_at_k(
            labels,
            ranks,
            10,
        )

        top1 = top1_accuracy(
            labels,
            ranks,
        )

        if auc is not None:
            auc_values.append(
                auc
            )

        if semantic_auc is not None:
            semantic_auc_values.append(
                semantic_auc
            )

        if mrr is not None:
            mrr_values.append(
                mrr
            )

        if ndcg5 is not None:
            ndcg5_values.append(
                ndcg5
            )

        if ndcg10 is not None:
            ndcg10_values.append(
                ndcg10
            )

        if top1 is not None:
            top1_values.append(
                top1
            )

    return {
        "num_impressions":
            num_impressions,

        "top1_accuracy":
            float(
                np.mean(
                    top1_values
                )
            )
            if top1_values
            else float("nan"),

        "auc":
            float(
                np.mean(
                    auc_values
                )
            )
            if auc_values
            else float("nan"),

        "semantic_auc_without_tie":
            float(
                np.mean(
                    semantic_auc_values
                )
            )
            if semantic_auc_values
            else float("nan"),

        "mrr":
            float(
                np.mean(
                    mrr_values
                )
            )
            if mrr_values
            else float("nan"),

        "ndcg@5":
            float(
                np.mean(
                    ndcg5_values
                )
            )
            if ndcg5_values
            else float("nan"),

        "ndcg@10":
            float(
                np.mean(
                    ndcg10_values
                )
            )
            if ndcg10_values
            else float("nan"),
    }


def evaluate_tie_pairs(
    df: pd.DataFrame,
    group_column: str,
) -> Dict[str, float]:
    total_pairs = 0

    correct_sum = 0.0

    positive_score_sum = 0.0
    negative_score_sum = 0.0

    margin_sum = 0.0

    for _, impression in df.groupby(
        group_column,
        sort=False,
    ):
        for _, sid_group in impression.groupby(
            [
                "c1",
                "c2",
                "c3",
            ],
            sort=False,
        ):
            positives = sid_group[
                sid_group["label"] == 1
            ]

            negatives = sid_group[
                sid_group["label"] == 0
            ]

            if (
                len(positives) == 0
                or len(negatives) == 0
            ):
                continue

            positive_scores = (
                positives[
                    "tie_score"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            negative_scores = (
                negatives[
                    "tie_score"
                ]
                .to_numpy(
                    dtype=np.float64
                )
            )

            for positive_score in positive_scores:

                for negative_score in negative_scores:

                    total_pairs += 1

                    positive_score_sum += (
                        positive_score
                    )

                    negative_score_sum += (
                        negative_score
                    )

                    margin_sum += (
                        positive_score
                        - negative_score
                    )

                    if (
                        positive_score
                        > negative_score
                    ):
                        correct_sum += 1.0

                    elif (
                        positive_score
                        == negative_score
                    ):
                        correct_sum += 0.5

    if total_pairs == 0:
        return {
            "num_tie_pairs":
                0,

            "tie_pair_accuracy":
                float("nan"),

            "mean_positive_tie_score":
                float("nan"),

            "mean_negative_tie_score":
                float("nan"),

            "mean_tie_margin":
                float("nan"),
        }

    return {
        "num_tie_pairs":
            total_pairs,

        "tie_pair_accuracy":
            correct_sum
            / total_pairs,

        "mean_positive_tie_score":
            positive_score_sum
            / total_pairs,

        "mean_negative_tie_score":
            negative_score_sum
            / total_pairs,

        "mean_tie_margin":
            margin_sum
            / total_pairs,
    }


def evaluate(
    prediction_path: Path,
) -> Dict:

    if not prediction_path.exists():
        raise FileNotFoundError(
            f"Prediction file not found:\n"
            f"{prediction_path}"
        )

    df = pd.read_parquet(
        prediction_path
    )

    required_columns = [
        "label",
        "candidate_score",
        "tie_score",
        "rank",
        "c1",
        "c2",
        "c3",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if "sample_index" in df.columns:
        group_column = (
            "sample_index"
        )

    elif "impression_id" in df.columns:
        group_column = (
            "impression_id"
        )

    else:
        raise ValueError(
            "Prediction file must contain "
            "sample_index or impression_id."
        )

    df["label"] = (
        df["label"]
        .astype(int)
    )

    collision_map = {}

    for key, group in df.groupby(
        group_column,
        sort=False,
    ):
        collision_map[
            key
        ] = has_collision(
            group
        )

    collision_keys = [
        key
        for key, value
        in collision_map.items()
        if value
    ]

    collision_mask = (
        df[
            group_column
        ]
        .isin(
            collision_keys
        )
    )

    collision_df = df[
        collision_mask
    ].copy()

    overall_metrics = (
        evaluate_ranking(
            df=
                df,

            group_column=
                group_column,
        )
    )

    if len(collision_df) > 0:

        collision_metrics = (
            evaluate_ranking(
                df=
                    collision_df,

                group_column=
                    group_column,
            )
        )

    else:

        collision_metrics = {
            "num_impressions":
                0,

            "top1_accuracy":
                float("nan"),

            "auc":
                float("nan"),

            "semantic_auc_without_tie":
                float("nan"),

            "mrr":
                float("nan"),

            "ndcg@5":
                float("nan"),

            "ndcg@10":
                float("nan"),
        }

    tie_metrics = (
        evaluate_tie_pairs(
            df=
                df,

            group_column=
                group_column,
        )
    )

    num_impressions = (
        overall_metrics[
            "num_impressions"
        ]
    )

    num_collision_impressions = (
        collision_metrics[
            "num_impressions"
        ]
    )

    collision_rate = (
        num_collision_impressions
        / num_impressions
        if num_impressions > 0
        else 0.0
    )

    return {
        "overall":
            overall_metrics,

        "collision_subset":
            collision_metrics,

        "tie_breaking":
            tie_metrics,

        "collision_rate":
            collision_rate,

        "num_collision_impressions":
            num_collision_impressions,
    }


def print_metrics(
    metrics: Dict,
) -> None:

    overall = metrics[
        "overall"
    ]

    collision = metrics[
        "collision_subset"
    ]

    tie = metrics[
        "tie_breaking"
    ]

    print()
    print(
        "========================================"
    )

    print(
        "Overall Ranking Performance"
    )

    print(
        "========================================"
    )

    print(
        f"Impressions : "
        f"{overall['num_impressions']:,}"
    )

    print(
        f"Top-1       : "
        f"{overall['top1_accuracy']:.6f}"
    )

    print(
        f"AUC         : "
        f"{overall['auc']:.6f}"
    )

    print(
        f"MRR         : "
        f"{overall['mrr']:.6f}"
    )

    print(
        f"nDCG@5      : "
        f"{overall['ndcg@5']:.6f}"
    )

    print(
        f"nDCG@10     : "
        f"{overall['ndcg@10']:.6f}"
    )

    print(
        f"Semantic AUC without tie : "
        f"{overall['semantic_auc_without_tie']:.6f}"
    )

    print()

    print(
        "========================================"
    )

    print(
        "Collision Analysis"
    )

    print(
        "========================================"
    )

    print(
        f"Collision impressions : "
        f"{metrics['num_collision_impressions']:,}"
    )

    print(
        f"Collision rate        : "
        f"{metrics['collision_rate']:.6f}"
    )

    if (
        collision[
            "num_impressions"
        ]
        > 0
    ):

        print(
            f"Collision Top-1       : "
            f"{collision['top1_accuracy']:.6f}"
        )

        print(
            f"Collision AUC         : "
            f"{collision['auc']:.6f}"
        )

        print(
            f"Collision MRR         : "
            f"{collision['mrr']:.6f}"
        )

        print(
            f"Collision nDCG@5      : "
            f"{collision['ndcg@5']:.6f}"
        )

        print(
            f"Collision nDCG@10     : "
            f"{collision['ndcg@10']:.6f}"
        )

        print(
            f"Collision Semantic AUC without tie : "
            f"{collision['semantic_auc_without_tie']:.6f}"
        )

    else:

        print(
            "No collision impressions."
        )

    print()

    print(
        "========================================"
    )

    print(
        "Tie-Breaking Performance"
    )

    print(
        "========================================"
    )

    print(
        f"Tie pairs             : "
        f"{tie['num_tie_pairs']:,}"
    )

    print(
        f"Tie pair accuracy     : "
        f"{tie['tie_pair_accuracy']:.6f}"
    )

    print(
        f"Positive tie score    : "
        f"{tie['mean_positive_tie_score']:.6f}"
    )

    print(
        f"Negative tie score    : "
        f"{tie['mean_negative_tie_score']:.6f}"
    )

    print(
        f"Mean tie margin       : "
        f"{tie['mean_tie_margin']:.6f}"
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate candidate ranking "
            "for news recommendation."
        )
    )

    parser.add_argument(
        "--prediction_path",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "test_candidate_scores.parquet"
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "test_metrics.json"
        ),
    )

    args = parser.parse_args()

    prediction_path = (
        resolve_path(
            args.prediction_path
        )
    )

    output_path = (
        resolve_path(
            args.output_path
        )
    )

    metrics = evaluate(
        prediction_path=
            prediction_path
    )

    print_metrics(
        metrics
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        output_path,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            metrics,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print()

    print(
        "Metrics saved:",
        output_path,
    )


if __name__ == "__main__":
    main()