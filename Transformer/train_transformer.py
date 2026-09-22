from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Dict, Optional

import gin
import numpy as np
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader

from data.sequence import NewsSequenceDataset, collate_news_sequences
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
    # 상대경로를 Transformer 프로젝트 기준 절대경로로 변환
    path_obj = Path(path)
    return path_obj if path_obj.is_absolute() else BASE_DIR / path_obj


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_metric_dict() -> Dict[str, float]:
    # 한 epoch 동안 누적할 학습/평가 통계
    return {
        "preference_loss_sum": 0.0,
        "positive_score_sum": 0.0,
        "negative_score_sum": 0.0,
        "positive_prob_sum": 0.0,
        "negative_prob_sum": 0.0,
        "top1_correct": 0,
        "num_impressions": 0,
        "num_positive_candidates": 0,
        "num_negative_candidates": 0,
    }


@torch.no_grad()
def update_metrics(metrics: Dict[str, float], loss_output, model_output, candidate_labels: torch.Tensor) -> None:
    # 한 batch의 loss, score, probability, Top-1 결과 누적
    candidate_labels = candidate_labels.float()
    candidate_scores = model_output.candidate_scores
    batch_size = candidate_scores.shape[0]

    positive_mask = candidate_labels == 1
    negative_mask = candidate_labels == 0

    num_positive = int(positive_mask.sum().item())
    num_negative = int(negative_mask.sum().item())

    # Loss
    metrics["preference_loss_sum"] += loss_output.preference_loss.item() * batch_size

    # Raw candidate score
    metrics["positive_score_sum"] += float(candidate_scores[positive_mask].sum().item())
    metrics["negative_score_sum"] += float(candidate_scores[negative_mask].sum().item())

    # 후보 5개 사이의 상대확률
    candidate_probs = torch.softmax(candidate_scores, dim=1)
    metrics["positive_prob_sum"] += float(candidate_probs[positive_mask].sum().item())
    metrics["negative_prob_sum"] += float(candidate_probs[negative_mask].sum().item())

    # Top-1 Accuracy
    predicted_indices = candidate_scores.argmax(dim=1)
    target_indices = candidate_labels.argmax(dim=1)
    metrics["top1_correct"] += int((predicted_indices == target_indices).sum().item())

    # Count
    metrics["num_impressions"] += batch_size
    metrics["num_positive_candidates"] += num_positive
    metrics["num_negative_candidates"] += num_negative


def finalize_metrics(metrics: Dict[str, float], loss_fn: TransformerLoss) -> Dict[str, float]:
    # epoch 누적값을 평균 metric으로 변환
    num_impressions = int(metrics["num_impressions"])
    num_positive = int(metrics["num_positive_candidates"])
    num_negative = int(metrics["num_negative_candidates"])

    if num_impressions == 0:
        raise ValueError("No impressions were processed.")

    preference_loss = metrics["preference_loss_sum"] / num_impressions
    total_loss = loss_fn.lambda_preference * preference_loss

    positive_score = metrics["positive_score_sum"] / num_positive if num_positive > 0 else float("nan")
    negative_score = metrics["negative_score_sum"] / num_negative if num_negative > 0 else float("nan")
    positive_prob = metrics["positive_prob_sum"] / num_positive if num_positive > 0 else float("nan")
    negative_prob = metrics["negative_prob_sum"] / num_negative if num_negative > 0 else float("nan")
    top1_accuracy = metrics["top1_correct"] / num_impressions

    return {
        "total_loss": total_loss,
        "preference_loss": preference_loss,
        "positive_score": positive_score,
        "negative_score": negative_score,
        "positive_prob": positive_prob,
        "negative_prob": negative_prob,
        "top1_accuracy": top1_accuracy,
        "top1_correct": int(metrics["top1_correct"]),
        "num_impressions": num_impressions,
        "num_positive_candidates": num_positive,
        "num_negative_candidates": num_negative,
    }


def train_one_epoch(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:
    # training dataset 전체를 한 번 학습
    model.train()
    metrics = create_metric_dict()

    for batch in dataloader:
        # Batch → GPU
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        # Gradient 초기화
        optimizer.zero_grad(set_to_none=True)

        # Forward
        model_output = model(history_sids=history_sids, history_mask=history_mask, candidate_sids=candidate_sids)

        # Loss
        loss_output = loss_fn(candidate_scores=model_output.candidate_scores, candidate_labels=candidate_labels)

        # Backpropagation
        loss_output.total_loss.backward()

        # Gradient clipping
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=gradient_clip_norm)

        # Parameter update
        optimizer.step()

        # Metric update
        update_metrics(metrics=metrics, loss_output=loss_output, model_output=model_output, candidate_labels=candidate_labels)

    return finalize_metrics(metrics=metrics, loss_fn=loss_fn)


@torch.no_grad()
def evaluate(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    # validation dataset에서 loss와 Top-1 성능 계산
    model.eval()
    metrics = create_metric_dict()

    for batch in dataloader:
        # Batch → GPU
        history_sids = batch["history_sids"].to(device, non_blocking=True)
        history_mask = batch["history_mask"].to(device, non_blocking=True)
        candidate_sids = batch["candidate_sids"].to(device, non_blocking=True)
        candidate_labels = batch["candidate_labels"].to(device, non_blocking=True)

        # Forward
        model_output = model(history_sids=history_sids, history_mask=history_mask, candidate_sids=candidate_sids)

        # Loss
        loss_output = loss_fn(candidate_scores=model_output.candidate_scores, candidate_labels=candidate_labels)

        # Metric
        update_metrics(metrics=metrics, loss_output=loss_output, model_output=model_output, candidate_labels=candidate_labels)

    return finalize_metrics(metrics=metrics, loss_fn=loss_fn)


def print_metrics(split_name: str, metrics: Dict[str, float]) -> None:
    # train/validation 주요 metric 출력
    print(f"{split_name} Loss={metrics['total_loss']:.6f} | Preference={metrics['preference_loss']:.6f}")
    print(f"{split_name} Top-1 Accuracy={metrics['top1_accuracy']:.4%} | Correct={int(metrics['top1_correct']):,}/{int(metrics['num_impressions']):,}")
    print(f"{split_name} Score | Positive={metrics['positive_score']:.4f} | Negative={metrics['negative_score']:.4f}")
    print(f"{split_name} Probability | Positive={metrics['positive_prob']:.6f} | Negative={metrics['negative_prob']:.6f}")
    print(f"{split_name} Candidates | Positive={int(metrics['num_positive_candidates']):,} | Negative={int(metrics['num_negative_candidates']):,}")


def save_checkpoint(
    path: Path,
    model: NewsEncoderDecoderTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_loss: float,
    validation_top1_accuracy: float,
) -> None:
    # model/optimizer 상태와 gin 설정 저장
    path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "validation_loss": validation_loss,
        "validation_top1_accuracy": validation_top1_accuracy,
        "gin_config": gin.config_str(),
    }

    torch.save(checkpoint, path)


@gin.configurable
def train(
    train_path: str,
    validation_path: str,
    save_dir: str,
    batch_size: int = 128,
    num_epochs: int = 30,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    gradient_clip_norm: Optional[float] = 1.0,
    num_workers: int = 0,
    seed: int = 42,
    early_stopping_patience: Optional[int] = None,
) -> None:
    # train/validation dataset을 이용해 Transformer 전체 학습
    set_seed(seed)
    device = get_device()

    print()
    print("Transformer Training")
    print("Device:", device)

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    # Path
    train_path = resolve_path(train_path)
    validation_path = resolve_path(validation_path)
    save_dir = resolve_path(save_dir)

    print("Train path      :", train_path)
    print("Validation path :", validation_path)
    print("Save directory  :", save_dir)

    # File check
    if not train_path.exists():
        raise FileNotFoundError(f"Train file not found:\n{train_path}")

    if not validation_path.exists():
        raise FileNotFoundError(f"Validation file not found:\n{validation_path}")

    # Dataset
    train_dataset = NewsSequenceDataset(parquet_path=str(train_path))
    validation_dataset = NewsSequenceDataset(parquet_path=str(validation_path))

    print("Train samples      :", f"{len(train_dataset):,}")
    print("Validation samples :", f"{len(validation_dataset):,}")

    # DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_news_sequences,
        pin_memory=(device.type == "cuda"),
    )

    # Model / Loss / Optimizer
    model = NewsEncoderDecoderTransformer().to(device)
    loss_fn = TransformerLoss().to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Parameter count
    total_parameters = sum(p.numel() for p in model.parameters())
    trainable_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("Total parameters     :", f"{total_parameters:,}")
    print("Trainable parameters :", f"{trainable_parameters:,}")

    # Before training validation
    print()
    print("Before Training")

    initial_validation_metrics = evaluate(model=model, loss_fn=loss_fn, dataloader=validation_loader, device=device)
    print_metrics("Validation", initial_validation_metrics)

    # Best model tracking
    best_validation_loss = float("inf")
    best_validation_top1 = 0.0
    epochs_without_improvement = 0

    save_dir.mkdir(parents=True, exist_ok=True)

    epoch = 0
    validation_loss = float("inf")
    validation_top1 = 0.0

    # Epoch loop
    for epoch in range(1, num_epochs + 1):
        print()
        print(f"Epoch {epoch}/{num_epochs}")

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )
        print_metrics("Train", train_metrics)

        # Validation
        validation_metrics = evaluate(model=model, loss_fn=loss_fn, dataloader=validation_loader, device=device)
        print_metrics("Validation", validation_metrics)

        validation_loss = validation_metrics["total_loss"]
        validation_top1 = validation_metrics["top1_accuracy"]

        # Every epoch checkpoint
        save_checkpoint(
            path=save_dir / f"checkpoint_epoch_{epoch}.pt",
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
            validation_top1_accuracy=validation_top1,
        )

        # Best checkpoint
        if validation_loss < best_validation_loss:
            best_validation_loss = validation_loss
            best_validation_top1 = validation_top1
            epochs_without_improvement = 0

            save_checkpoint(
                path=save_dir / "checkpoint_best.pt",
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
                validation_top1_accuracy=validation_top1,
            )

            print(f"✓ Best checkpoint updated (Validation Loss={best_validation_loss:.6f}, Top-1={best_validation_top1:.4%})")

        else:
            epochs_without_improvement += 1
            print("Validation loss did not improve.")

        # Early stopping
        if early_stopping_patience is not None and epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping. No validation improvement for {early_stopping_patience} epochs.")
            break

    # Final checkpoint
    final_checkpoint_path = save_dir / "checkpoint_final.pt"

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        validation_loss=validation_loss,
        validation_top1_accuracy=validation_top1,
    )

    print()
    print("Training Finished")
    print("Best validation loss:", f"{best_validation_loss:.6f}")
    print("Top-1 at best checkpoint:", f"{best_validation_top1:.4%}")
    print("Best checkpoint:", save_dir / "checkpoint_best.pt")
    print("Final checkpoint:", final_checkpoint_path)


def main() -> None:
    # --config으로 gin 파일을 받아 학습 시작
    parser = argparse.ArgumentParser(description="Train News Semantic-ID Encoder-Decoder Transformer")
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config_path = resolve_path(args.config)

    if not config_path.exists():
        raise FileNotFoundError(f"Gin config not found:\n{config_path}")

    print("Gin config:", config_path)
    gin.parse_config_file(str(config_path))
    train()


if __name__ == "__main__":
    main()