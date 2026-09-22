from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import gin
import pandas as pd
import torch

from torch.utils.data import DataLoader

from data.sequence import (
    NewsSequenceDataset,
    collate_news_sequences,
)

from modules.model import NewsEncoderDecoderTransformer


# ============================================================
# Base path
# ============================================================

BASE_DIR = Path(__file__).resolve().parent


# ============================================================
# Path
# ============================================================

def resolve_path(path: str) -> Path:
    """
    상대경로를 Transformer 프로젝트 기준 절대경로로 변환한다.
    """

    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return BASE_DIR / path_obj


# ============================================================
# Device
# ============================================================

def get_device() -> torch.device:
    """
    CUDA 사용 가능 시 GPU, 아니면 CPU를 사용한다.
    """

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# Article ID 정리
# ============================================================

def normalize_article_id(article_id):
    """
    article_id를 저장하기 쉬운 문자열 형태로 변환한다.
    """

    if article_id is None:
        return None

    try:
        if pd.isna(article_id):
            return None
    except (TypeError, ValueError):
        pass

    return str(article_id)


# ============================================================
# Checkpoint load
# ============================================================

def load_checkpoint(
    checkpoint_path: Path,
    model: NewsEncoderDecoderTransformer,
    device: torch.device,
) -> dict:
    """
    저장된 Transformer checkpoint를 불러온다.
    """

    if not checkpoint_path.exists():

        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    # train_transformer.py에서 저장한
    # checkpoint 형식
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    # model state_dict 자체만 저장된 경우
    else:

        model.load_state_dict(
            checkpoint
        )

    return checkpoint


# ============================================================
# Prediction / Test evaluation
# ============================================================

@torch.no_grad()
def predict(
    model: NewsEncoderDecoderTransformer,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[
    pd.DataFrame,
    Dict[str, float],
]:
    """
    후보 5개의 score를 계산하고
    Top-1 Accuracy와 후보별 결과를 반환한다.
    """

    model.eval()

    prediction_rows = []

    # --------------------------------------------------------
    # Metric 누적값
    # --------------------------------------------------------

    num_impressions = 0
    num_correct = 0

    positive_score_sum = 0.0
    negative_score_sum = 0.0

    positive_prob_sum = 0.0
    negative_prob_sum = 0.0

    num_positive_candidates = 0
    num_negative_candidates = 0

    sample_index = 0

    num_batches = len(dataloader)

    # ========================================================
    # Batch loop
    # ========================================================

    for batch_idx, batch in enumerate(
        dataloader
    ):

        # ----------------------------------------------------
        # Input → device
        # ----------------------------------------------------

        history_sids = (
            batch["history_sids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        history_mask = (
            batch["history_mask"]
            .to(
                device,
                non_blocking=True,
            )
        )

        candidate_sids = (
            batch["candidate_sids"]
            .to(
                device,
                non_blocking=True,
            )
        )

        candidate_labels = (
            batch["candidate_labels"]
            .to(
                device,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Model forward
        #
        # history:
        # [B,H,4]
        #
        # candidates:
        # [B,5,3]
        #
        # output:
        # candidate_scores [B,5]
        # ----------------------------------------------------

        model_output = model(
            history_sids=
                history_sids,

            history_mask=
                history_mask,

            candidate_sids=
                candidate_sids,
        )

        candidate_scores = (
            model_output.candidate_scores
        )

        # ====================================================
        # Candidate relative probability
        #
        # raw score는 log probability 합이므로
        # 음수여도 정상.
        #
        # 사람이 해석하기 쉽게 5개 후보 사이에서
        # softmax probability도 계산한다.
        #
        # [B,5]
        # ====================================================

        candidate_probs = torch.softmax(
            candidate_scores,
            dim=1,
        )

        # ====================================================
        # Top-1 prediction
        # ====================================================

        predicted_indices = (
            candidate_scores.argmax(
                dim=1
            )
        )

        target_indices = (
            candidate_labels.argmax(
                dim=1
            )
        )

        correct_mask = (
            predicted_indices
            == target_indices
        )

        batch_size = (
            candidate_sids.shape[0]
        )

        num_candidates = (
            candidate_sids.shape[1]
        )

        # ====================================================
        # Ranking
        #
        # score가 클수록 높은 rank
        #
        # 예:
        # [-5.0, -2.0, -7.0, -3.0, -6.0]
        #
        # rank:
        # [3, 1, 5, 2, 4]
        # ====================================================

        sorted_indices = torch.argsort(
            candidate_scores,
            dim=1,
            descending=True,
        )

        ranks = torch.empty_like(
            sorted_indices
        )

        rank_values = (
            torch.arange(
                1,
                num_candidates + 1,
                device=device,
            )
            .unsqueeze(0)
            .expand(
                batch_size,
                -1,
            )
        )

        ranks.scatter_(
            dim=1,
            index=sorted_indices,
            src=rank_values,
        )

        # ====================================================
        # Metric 계산
        # ====================================================

        positive_mask = (
            candidate_labels == 1
        )

        negative_mask = (
            candidate_labels == 0
        )

        batch_num_positive = int(
            positive_mask.sum().item()
        )

        batch_num_negative = int(
            negative_mask.sum().item()
        )

        positive_score_sum += float(
            candidate_scores[
                positive_mask
            ].sum().item()
        )

        negative_score_sum += float(
            candidate_scores[
                negative_mask
            ].sum().item()
        )

        positive_prob_sum += float(
            candidate_probs[
                positive_mask
            ].sum().item()
        )

        negative_prob_sum += float(
            candidate_probs[
                negative_mask
            ].sum().item()
        )

        num_positive_candidates += (
            batch_num_positive
        )

        num_negative_candidates += (
            batch_num_negative
        )

        num_correct += int(
            correct_mask.sum().item()
        )

        num_impressions += (
            batch_size
        )

        # ====================================================
        # 후보별 prediction 결과 저장
        # ====================================================

        for i in range(
            batch_size
        ):

            # ------------------------------------------------
            # Metadata
            # ------------------------------------------------

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

            article_ids = (
                batch[
                    "candidate_article_ids"
                ][i]
            )

            # article ID가 없는 경우도 허용
            if article_ids is None:

                article_ids = [
                    None
                ] * num_candidates

            if (
                len(article_ids)
                != num_candidates
            ):

                raise ValueError(
                    "candidate_article_ids length mismatch "
                    f"at sample {sample_index}: "
                    f"{len(article_ids)} vs "
                    f"{num_candidates}"
                )

            is_correct = bool(
                correct_mask[
                    i
                ].item()
            )

            # =================================================
            # Candidate 5개 각각 저장
            # =================================================

            for candidate_idx in range(
                num_candidates
            ):

                sid = (
                    candidate_sids[
                        i,
                        candidate_idx,
                    ]
                    .detach()
                    .cpu()
                    .tolist()
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
                        # -------------------------------------
                        # Sample metadata
                        # -------------------------------------

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

                        # -------------------------------------
                        # Semantic ID
                        # -------------------------------------

                        "c1":
                            int(
                                sid[0]
                            ),

                        "c2":
                            int(
                                sid[1]
                            ),

                        "c3":
                            int(
                                sid[2]
                            ),

                        # -------------------------------------
                        # Ground truth
                        # -------------------------------------

                        "label":
                            float(
                                candidate_labels[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        # -------------------------------------
                        # Main candidate score
                        #
                        # log p(c1)
                        # + log p(c2)
                        # + log p(c3)
                        # -------------------------------------

                        "candidate_score":
                            float(
                                candidate_scores[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        # -------------------------------------
                        # 5개 후보 내 상대 probability
                        # -------------------------------------

                        "candidate_probability":
                            float(
                                candidate_probs[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        # -------------------------------------
                        # SID level별 log probability
                        # -------------------------------------

                        "c1_log_prob":
                            float(
                                model_output
                                .c1_log_probs[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        "c2_log_prob":
                            float(
                                model_output
                                .c2_log_probs[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        "c3_log_prob":
                            float(
                                model_output
                                .c3_log_probs[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        # -------------------------------------
                        # Ranking
                        # -------------------------------------

                        "rank":
                            int(
                                ranks[
                                    i,
                                    candidate_idx,
                                ].item()
                            ),

                        "is_top1":
                            bool(
                                predicted_indices[
                                    i
                                ].item()
                                == candidate_idx
                            ),

                        # 같은 impression의 후보 5개에
                        # 동일하게 기록
                        "top1_correct":
                            is_correct,
                    }
                )

            sample_index += 1

        # ====================================================
        # Progress
        # ====================================================

        if (
            (batch_idx + 1) % 50 == 0
            or (batch_idx + 1) == num_batches
        ):

            print(
                f"Processed batch "
                f"{batch_idx + 1:,}/"
                f"{num_batches:,}"
            )

    # ========================================================
    # Final metrics
    # ========================================================

    top1_accuracy = (
        num_correct
        / num_impressions
        if num_impressions > 0
        else 0.0
    )

    positive_score = (
        positive_score_sum
        / num_positive_candidates
        if num_positive_candidates > 0
        else float("nan")
    )

    negative_score = (
        negative_score_sum
        / num_negative_candidates
        if num_negative_candidates > 0
        else float("nan")
    )

    positive_probability = (
        positive_prob_sum
        / num_positive_candidates
        if num_positive_candidates > 0
        else float("nan")
    )

    negative_probability = (
        negative_prob_sum
        / num_negative_candidates
        if num_negative_candidates > 0
        else float("nan")
    )

    metrics = {
        "num_impressions":
            num_impressions,

        "num_correct":
            num_correct,

        "top1_accuracy":
            top1_accuracy,

        "positive_score":
            positive_score,

        "negative_score":
            negative_score,

        "positive_probability":
            positive_probability,

        "negative_probability":
            negative_probability,

        "num_positive_candidates":
            num_positive_candidates,

        "num_negative_candidates":
            num_negative_candidates,
    }

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    return (
        predictions_df,
        metrics,
    )


# ============================================================
# Print metrics
# ============================================================

def print_metrics(
    metrics: Dict[str, float],
) -> None:
    """
    Test dataset의 최종 평가 결과를 출력한다.
    """

    print()

    print(
        "=============================="
    )

    print(
        "Test Evaluation"
    )

    print(
        "=============================="
    )

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

    print()

    print(
        "Score | "
        f"Positive={metrics['positive_score']:.4f} | "
        f"Negative={metrics['negative_score']:.4f}"
    )

    print(
        "Probability | "
        f"Positive={metrics['positive_probability']:.6f} | "
        f"Negative={metrics['negative_probability']:.6f}"
    )

    print()

    print(
        "Candidates | "
        f"Positive="
        f"{int(metrics['num_positive_candidates']):,} | "
        f"Negative="
        f"{int(metrics['num_negative_candidates']):,}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate 1-positive 4-negative "
            "candidate news ranking using "
            "Semantic-ID Transformer scores."
        )
    )

    # --------------------------------------------------------
    # Gin config
    # --------------------------------------------------------

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    # --------------------------------------------------------
    # Best checkpoint
    # --------------------------------------------------------

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "checkpoint_best.pt"
        ),
    )

    # --------------------------------------------------------
    # Test dataset
    # --------------------------------------------------------

    parser.add_argument(
        "--test_path",
        type=str,
        default=(
            "datasets/ebnerd/"
            "test_sequences_1pos4neg.parquet"
        ),
    )

    # --------------------------------------------------------
    # Candidate별 prediction 결과
    # --------------------------------------------------------

    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "test_candidate_scores.parquet"
        ),
    )

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    # ========================================================
    # Path
    # ========================================================

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

    # ========================================================
    # File check
    # ========================================================

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

    # ========================================================
    # Gin
    #
    # config에는 train/loss 설정도 있으므로
    # predict 단계에서 필요 없는 설정은 skip
    # ========================================================

    gin.parse_config_file(
        str(config_path),
        skip_unknown=True,
    )

    # ========================================================
    # Device
    # ========================================================

    device = get_device()

    print()

    print(
        "Candidate Ranking Evaluation"
    )

    print(
        "Device:",
        device,
    )

    if device.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    print(
        "Test path:",
        test_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    # ========================================================
    # Test Dataset
    # ========================================================

    test_dataset = NewsSequenceDataset(
        parquet_path=
            str(test_path),
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

    # ========================================================
    # Model
    #
    # vocab size와 architecture는
    # gin config에서 주입
    # ========================================================

    model = (
        NewsEncoderDecoderTransformer()
        .to(device)
    )

    # ========================================================
    # Checkpoint
    # ========================================================

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

    if (
        isinstance(checkpoint, dict)
        and "validation_loss" in checkpoint
    ):

        print(
            "Checkpoint validation loss:",
            checkpoint[
                "validation_loss"
            ],
        )

    if (
        isinstance(checkpoint, dict)
        and "validation_top1_accuracy"
        in checkpoint
    ):

        print(
            "Checkpoint validation Top-1:",
            f"{checkpoint['validation_top1_accuracy']:.4%}",
        )

    # ========================================================
    # Test prediction
    # ========================================================

    predictions_df, metrics = predict(
        model=
            model,

        dataloader=
            test_loader,

        device=
            device,
    )

    # ========================================================
    # Save candidate predictions
    # ========================================================

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_parquet(
        output_path,
        index=False,
    )

    # ========================================================
    # Final result
    # ========================================================

    print_metrics(
        metrics
    )

    print()

    print(
        "Candidate scores saved:",
        output_path,
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()