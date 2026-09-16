from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import gin
import numpy as np
import torch

from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.sequence import (
    NewsSequenceDataset,
    collate_news_sequences,
)

from modules.model import NewsEncoderDecoderTransformer
from modules.loss import TransformerLoss


BASE_DIR = Path(__file__).resolve().parent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def resolve_path(path: str) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return BASE_DIR / path_obj


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_metric_dict() -> Dict[str, float]:
    return {
        "preference_loss_sum": 0.0,
        "tie_loss_sum": 0.0,

        "positive_score_sum": 0.0,
        "negative_score_sum": 0.0,

        "positive_prob_sum": 0.0,
        "negative_prob_sum": 0.0,

        "tie_positive_score_sum": 0.0,
        "tie_negative_score_sum": 0.0,

        "num_valid_candidates": 0,
        "num_positive_candidates": 0,
        "num_negative_candidates": 0,

        "num_tie_pairs": 0,
        "num_collision_impressions": 0,
        "num_impressions": 0,
    }


@torch.no_grad()
def get_tie_statistics(
    candidate_sids: Tensor,
    candidate_labels: Tensor,
    candidate_mask: Tensor,
    tie_scores: Tensor,
) -> Dict[str, float]:

    num_tie_pairs = 0
    num_collision_impressions = 0

    positive_score_sum = 0.0
    negative_score_sum = 0.0

    batch_size = candidate_sids.shape[0]

    for batch_idx in range(batch_size):

        valid = candidate_mask[batch_idx].bool()

        sids = candidate_sids[
            batch_idx
        ][valid]

        labels = candidate_labels[
            batch_idx
        ][valid]

        scores = tie_scores[
            batch_idx
        ][valid]

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

        negative_sids = sids[
            negative_indices
        ]

        negative_scores = scores[
            negative_indices
        ]

        collision_found = False

        for positive_idx in positive_indices:

            positive_sid = sids[
                positive_idx
            ]

            positive_score = scores[
                positive_idx
            ]

            collision_mask = (
                negative_sids
                == positive_sid.unsqueeze(0)
            ).all(dim=-1)

            pair_count = int(
                collision_mask.sum().item()
            )

            if pair_count == 0:
                continue

            collision_found = True
            num_tie_pairs += pair_count

            positive_score_sum += float(
                positive_score.item()
                * pair_count
            )

            negative_score_sum += float(
                negative_scores[
                    collision_mask
                ].sum().item()
            )

        if collision_found:
            num_collision_impressions += 1

    return {
        "num_tie_pairs":
            num_tie_pairs,

        "num_collision_impressions":
            num_collision_impressions,

        "tie_positive_score_sum":
            positive_score_sum,

        "tie_negative_score_sum":
            negative_score_sum,
    }


@torch.no_grad()
def update_metrics(
    metrics: Dict[str, float],
    loss_output,
    model_output,
    candidate_sids: Tensor,
    candidate_labels: Tensor,
    candidate_mask: Tensor,
) -> None:

    candidate_labels = candidate_labels.float()
    candidate_mask = candidate_mask.bool()

    positive_mask = (
        candidate_mask
        & (candidate_labels == 1)
    )

    negative_mask = (
        candidate_mask
        & (candidate_labels == 0)
    )

    num_valid = int(
        candidate_mask.sum().item()
    )

    num_positive = int(
        positive_mask.sum().item()
    )

    num_negative = int(
        negative_mask.sum().item()
    )

    metrics[
        "preference_loss_sum"
    ] += (
        loss_output.preference_loss.item()
        * num_valid
    )

    candidate_scores = (
        model_output.candidate_scores
    )

    candidate_probs = torch.exp(
        candidate_scores
    )

    if num_positive > 0:

        metrics[
            "positive_score_sum"
        ] += float(
            candidate_scores[
                positive_mask
            ].sum().item()
        )

        metrics[
            "positive_prob_sum"
        ] += float(
            candidate_probs[
                positive_mask
            ].sum().item()
        )

    if num_negative > 0:

        metrics[
            "negative_score_sum"
        ] += float(
            candidate_scores[
                negative_mask
            ].sum().item()
        )

        metrics[
            "negative_prob_sum"
        ] += float(
            candidate_probs[
                negative_mask
            ].sum().item()
        )

    tie_stats = get_tie_statistics(
        candidate_sids=
            candidate_sids,

        candidate_labels=
            candidate_labels,

        candidate_mask=
            candidate_mask,

        tie_scores=
            model_output.tie_scores,
    )

    num_tie_pairs = int(
        tie_stats[
            "num_tie_pairs"
        ]
    )

    if num_tie_pairs > 0:

        metrics[
            "tie_loss_sum"
        ] += (
            loss_output.tie_loss.item()
            * num_tie_pairs
        )

    metrics[
        "tie_positive_score_sum"
    ] += tie_stats[
        "tie_positive_score_sum"
    ]

    metrics[
        "tie_negative_score_sum"
    ] += tie_stats[
        "tie_negative_score_sum"
    ]

    metrics[
        "num_tie_pairs"
    ] += num_tie_pairs

    metrics[
        "num_collision_impressions"
    ] += int(
        tie_stats[
            "num_collision_impressions"
        ]
    )

    metrics[
        "num_valid_candidates"
    ] += num_valid

    metrics[
        "num_positive_candidates"
    ] += num_positive

    metrics[
        "num_negative_candidates"
    ] += num_negative

    metrics[
        "num_impressions"
    ] += candidate_sids.shape[0]


def finalize_metrics(
    metrics: Dict[str, float],
    loss_fn: TransformerLoss,
) -> Dict[str, float]:

    num_valid = int(
        metrics[
            "num_valid_candidates"
        ]
    )

    num_positive = int(
        metrics[
            "num_positive_candidates"
        ]
    )

    num_negative = int(
        metrics[
            "num_negative_candidates"
        ]
    )

    num_tie_pairs = int(
        metrics[
            "num_tie_pairs"
        ]
    )

    num_impressions = int(
        metrics[
            "num_impressions"
        ]
    )

    if num_valid == 0:
        raise ValueError(
            "No valid candidates were processed."
        )

    preference_loss = (
        metrics[
            "preference_loss_sum"
        ]
        / num_valid
    )

    tie_loss = (
        metrics[
            "tie_loss_sum"
        ]
        / num_tie_pairs
        if num_tie_pairs > 0
        else 0.0
    )

    positive_score = (
        metrics[
            "positive_score_sum"
        ]
        / num_positive
        if num_positive > 0
        else float("nan")
    )

    negative_score = (
        metrics[
            "negative_score_sum"
        ]
        / num_negative
        if num_negative > 0
        else float("nan")
    )

    positive_prob = (
        metrics[
            "positive_prob_sum"
        ]
        / num_positive
        if num_positive > 0
        else float("nan")
    )

    negative_prob = (
        metrics[
            "negative_prob_sum"
        ]
        / num_negative
        if num_negative > 0
        else float("nan")
    )

    tie_positive_score = (
        metrics[
            "tie_positive_score_sum"
        ]
        / num_tie_pairs
        if num_tie_pairs > 0
        else float("nan")
    )

    tie_negative_score = (
        metrics[
            "tie_negative_score_sum"
        ]
        / num_tie_pairs
        if num_tie_pairs > 0
        else float("nan")
    )

    collision_rate = (
        metrics[
            "num_collision_impressions"
        ]
        / num_impressions
        if num_impressions > 0
        else 0.0
    )

    total_loss = (
        loss_fn.lambda_preference
        * preference_loss

        + loss_fn.lambda_tie
        * tie_loss
    )

    return {
        "total_loss":
            total_loss,

        "preference_loss":
            preference_loss,

        "tie_loss":
            tie_loss,

        "positive_score":
            positive_score,

        "negative_score":
            negative_score,

        "positive_prob":
            positive_prob,

        "negative_prob":
            negative_prob,

        "tie_positive_score":
            tie_positive_score,

        "tie_negative_score":
            tie_negative_score,

        "num_valid_candidates":
            num_valid,

        "num_positive_candidates":
            num_positive,

        "num_negative_candidates":
            num_negative,

        "num_tie_pairs":
            num_tie_pairs,

        "num_collision_impressions":
            int(
                metrics[
                    "num_collision_impressions"
                ]
            ),

        "collision_rate":
            collision_rate,
    }


def train_one_epoch(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:

    model.train()

    metrics = create_metric_dict()

    for batch in dataloader:

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

        optimizer.zero_grad(
            set_to_none=True
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

        loss_output = loss_fn(
            candidate_scores=
                model_output.candidate_scores,

            tie_scores=
                model_output.tie_scores,

            candidate_sids=
                candidate_sids,

            candidate_labels=
                candidate_labels,

            candidate_mask=
                candidate_mask,
        )

        loss_output.total_loss.backward()

        if gradient_clip_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=
                    gradient_clip_norm,
            )

        optimizer.step()

        update_metrics(
            metrics=
                metrics,

            loss_output=
                loss_output,

            model_output=
                model_output,

            candidate_sids=
                candidate_sids,

            candidate_labels=
                candidate_labels,

            candidate_mask=
                candidate_mask,
        )

    return finalize_metrics(
        metrics=
            metrics,

        loss_fn=
            loss_fn,
    )


@torch.no_grad()
def evaluate(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:

    model.eval()

    metrics = create_metric_dict()

    for batch in dataloader:

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

        loss_output = loss_fn(
            candidate_scores=
                model_output.candidate_scores,

            tie_scores=
                model_output.tie_scores,

            candidate_sids=
                candidate_sids,

            candidate_labels=
                candidate_labels,

            candidate_mask=
                candidate_mask,
        )

        update_metrics(
            metrics=
                metrics,

            loss_output=
                loss_output,

            model_output=
                model_output,

            candidate_sids=
                candidate_sids,

            candidate_labels=
                candidate_labels,

            candidate_mask=
                candidate_mask,
        )

    return finalize_metrics(
        metrics=
            metrics,

        loss_fn=
            loss_fn,
    )


def print_metrics(
    split_name: str,
    metrics: Dict[str, float],
) -> None:

    print(
        f"{split_name} "
        f"Loss={metrics['total_loss']:.6f} | "
        f"Preference={metrics['preference_loss']:.6f} | "
        f"Tie={metrics['tie_loss']:.6f}"
    )

    print(
        f"{split_name} "
        f"Score | "
        f"Positive={metrics['positive_score']:.4f} | "
        f"Negative={metrics['negative_score']:.4f}"
    )

    print(
        f"{split_name} "
        f"Probability | "
        f"Positive={metrics['positive_prob']:.6f} | "
        f"Negative={metrics['negative_prob']:.6f}"
    )

    print(
        f"{split_name} "
        f"Tie score | "
        f"Positive={metrics['tie_positive_score']:.4f} | "
        f"Negative={metrics['tie_negative_score']:.4f}"
    )

    print(
        f"{split_name} "
        f"Collision | "
        f"Pairs={int(metrics['num_tie_pairs']):,} | "
        f"Impressions="
        f"{int(metrics['num_collision_impressions']):,} | "
        f"Rate={metrics['collision_rate']:.4%}"
    )

    print(
        f"{split_name} "
        f"Candidates | "
        f"Positive="
        f"{int(metrics['num_positive_candidates']):,} | "
        f"Negative="
        f"{int(metrics['num_negative_candidates']):,} | "
        f"Total="
        f"{int(metrics['num_valid_candidates']):,}"
    )


def save_checkpoint(
    path: Path,
    model: NewsEncoderDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch":
            epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "validation_loss":
            validation_loss,

        "gin_config":
            gin.config_str(),
    }

    torch.save(
        checkpoint,
        path,
    )


@gin.configurable
def train(
    train_path: str,
    validation_path: str,
    save_dir: str,

    batch_size: int = 128,
    num_epochs: int = 30,

    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,

    gradient_clip_norm: Optional[
        float
    ] = 1.0,

    num_workers: int = 0,
    seed: int = 42,

    early_stopping_patience: Optional[
        int
    ] = None,
) -> None:

    set_seed(seed)

    device = get_device()

    print()
    print("Transformer Training")
    print("Device:", device)

    if device.type == "cuda":
        print(
            "GPU:",
            torch.cuda.get_device_name(0),
        )

    train_path = resolve_path(
        train_path
    )

    validation_path = resolve_path(
        validation_path
    )

    save_dir = resolve_path(
        save_dir
    )

    print(
        "Train path      :",
        train_path,
    )

    print(
        "Validation path :",
        validation_path,
    )

    print(
        "Save directory  :",
        save_dir,
    )

    if not train_path.exists():
        raise FileNotFoundError(
            f"Train file not found:\n"
            f"{train_path}"
        )

    if not validation_path.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n"
            f"{validation_path}"
        )

    train_dataset = NewsSequenceDataset(
        parquet_path=
            str(train_path),
    )

    validation_dataset = NewsSequenceDataset(
        parquet_path=
            str(validation_path),
    )

    print(
        "Train samples      :",
        f"{len(train_dataset):,}",
    )

    print(
        "Validation samples :",
        f"{len(validation_dataset):,}",
    )

    train_loader = DataLoader(
        dataset=
            train_dataset,

        batch_size=
            batch_size,

        shuffle=True,

        num_workers=
            num_workers,

        collate_fn=
            collate_news_sequences,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        dataset=
            validation_dataset,

        batch_size=
            batch_size,

        shuffle=False,

        num_workers=
            num_workers,

        collate_fn=
            collate_news_sequences,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    model = (
        NewsEncoderDecoderTransformer()
        .to(device)
    )

    loss_fn = (
        TransformerLoss()
        .to(device)
    )

    optimizer = AdamW(
        model.parameters(),

        lr=
            learning_rate,

        weight_decay=
            weight_decay,
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "Total parameters     :",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters :",
        f"{trainable_parameters:,}",
    )

    print()
    print("Before Training")

    initial_validation_metrics = evaluate(
        model=
            model,

        loss_fn=
            loss_fn,

        dataloader=
            validation_loader,

        device=
            device,
    )

    print_metrics(
        "Validation",
        initial_validation_metrics,
    )

    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    epoch = 0
    validation_loss = float(
        "inf"
    )

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        print()
        print(
            f"Epoch {epoch}/{num_epochs}"
        )

        train_metrics = train_one_epoch(
            model=
                model,

            loss_fn=
                loss_fn,

            dataloader=
                train_loader,

            optimizer=
                optimizer,

            device=
                device,

            gradient_clip_norm=
                gradient_clip_norm,
        )

        print_metrics(
            "Train",
            train_metrics,
        )

        validation_metrics = evaluate(
            model=
                model,

            loss_fn=
                loss_fn,

            dataloader=
                validation_loader,

            device=
                device,
        )

        print_metrics(
            "Validation",
            validation_metrics,
        )

        validation_loss = (
            validation_metrics[
                "total_loss"
            ]
        )

        save_checkpoint(
            path=
                save_dir
                / f"checkpoint_epoch_{epoch}.pt",

            model=
                model,

            optimizer=
                optimizer,

            epoch=
                epoch,

            validation_loss=
                validation_loss,
        )

        if validation_loss < best_validation_loss:

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            save_checkpoint(
                path=
                    save_dir
                    / "checkpoint_best.pt",

                model=
                    model,

                optimizer=
                    optimizer,

                epoch=
                    epoch,

                validation_loss=
                    validation_loss,
            )

            print(
                "✓ Best checkpoint updated "
                f"(validation loss "
                f"{best_validation_loss:.6f})"
            )

        else:

            epochs_without_improvement += 1

            print(
                "Validation loss did not improve."
            )

        if (
            early_stopping_patience
            is not None
            and epochs_without_improvement
            >= early_stopping_patience
        ):

            print(
                f"Early stopping. "
                f"No validation improvement for "
                f"{early_stopping_patience} epochs."
            )

            break

    final_checkpoint_path = (
        save_dir
        / "checkpoint_final.pt"
    )

    save_checkpoint(
        path=
            final_checkpoint_path,

        model=
            model,

        optimizer=
            optimizer,

        epoch=
            epoch,

        validation_loss=
            validation_loss,
    )

    print()
    print("Training Finished")

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Best checkpoint:",
        save_dir
        / "checkpoint_best.pt",
    )

    print(
        "Final checkpoint:",
        final_checkpoint_path,
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Train News Semantic-ID "
            "Encoder-Decoder Transformer"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gin config not found:\n"
            f"{config_path}"
        )

    print(
        "Gin config:",
        config_path,
    )

    gin.parse_config_file(
        str(config_path)
    )

    train()


if __name__ == "__main__":
    main()