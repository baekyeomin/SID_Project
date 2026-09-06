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

# 현재 train_transformer.py가 있는 Transformer/ 폴더
BASE_DIR = Path(__file__).resolve().parent

# 실험 재현을 위해 랜덤 시드 고정
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


# Device
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# Metric accumulator
def create_metric_dict() -> Dict[str, float]:

    return {
        "total_loss": 0.0,
        "c1_loss": 0.0,
        "c2_loss": 0.0,
        "c3_loss": 0.0,
        "c4_loss": 0.0,

        "c1_correct": 0,
        "c2_correct": 0,
        "c3_correct": 0,
        "c4_correct": 0,
        "full_sid_correct": 0,

        "num_samples": 0,
    }

# 한 배치의 loss와 정확도 누적
def update_metrics(
    metrics: Dict[str, float],
    loss_output,
    model_output,
    target_sids: Tensor,
) -> None:
    
    batch_size = target_sids.shape[0]

    # loss_output은 batch 평균 loss이므로
    # batch_size를 곱해서 전체 sample 기준으로 누적

    metrics["total_loss"] += (
        loss_output.total_loss.item()
        * batch_size
    )

    metrics["c1_loss"] += (
        loss_output.c1_loss.item()
        * batch_size
    )

    metrics["c2_loss"] += (
        loss_output.c2_loss.item()
        * batch_size
    )

    metrics["c3_loss"] += (
        loss_output.c3_loss.item()
        * batch_size
    )

    metrics["c4_loss"] += (
        loss_output.c4_loss.item()
        * batch_size
    )

    # Prediction
    pred_c1 = torch.argmax(
        model_output.c1_logits,
        dim=-1,
    )

    pred_c2 = torch.argmax(
        model_output.c2_logits,
        dim=-1,
    )

    pred_c3 = torch.argmax(
        model_output.c3_logits,
        dim=-1,
    )

    pred_c4 = torch.argmax(
        model_output.c4_logits,
        dim=-1,
    )

    # Ground Truth
    target_c1 = target_sids[:, 0]
    target_c2 = target_sids[:, 1]
    target_c3 = target_sids[:, 2]
    target_c4 = target_sids[:, 3]
    
    # Hierarchy accuracy
    correct_c1 = pred_c1.eq(target_c1)
    correct_c2 = pred_c2.eq(target_c2)
    correct_c3 = pred_c3.eq(target_c3)
    correct_c4 = pred_c4.eq(target_c4)

    metrics["c1_correct"] += (
        correct_c1.sum().item()
    )

    metrics["c2_correct"] += (
        correct_c2.sum().item()
    )

    metrics["c3_correct"] += (
        correct_c3.sum().item()
    )

    metrics["c4_correct"] += (
        correct_c4.sum().item()
    )

    # Full SID accuracy
    # c1,c2,c3,c4를 전부 맞춘 경우에만 correct

    full_correct = (
        correct_c1
        & correct_c2
        & correct_c3
        & correct_c4
    )

    metrics["full_sid_correct"] += (
        full_correct.sum().item()
    )

    metrics["num_samples"] += batch_size

# 누적된 값을 샘플 평균 loss / 정확도 로 변환
def finalize_metrics(
    metrics: Dict[str, float],
) -> Dict[str, float]:

    n = metrics["num_samples"]

    if n == 0:
        raise ValueError(
            "No samples were processed."
        )

    return {
        "total_loss":
            metrics["total_loss"] / n,

        "c1_loss":
            metrics["c1_loss"] / n,

        "c2_loss":
            metrics["c2_loss"] / n,

        "c3_loss":
            metrics["c3_loss"] / n,

        "c4_loss":
            metrics["c4_loss"] / n,

        "c1_acc":
            metrics["c1_correct"] / n,

        "c2_acc":
            metrics["c2_correct"] / n,

        "c3_acc":
            metrics["c3_correct"] / n,

        "c4_acc":
            metrics["c4_correct"] / n,

        "full_sid_acc":
            metrics["full_sid_correct"] / n,
    }


def train_one_epoch(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    gradient_clip_norm: Optional[float] = 1.0,
) -> Dict[str, float]:

    # 학습 모드
    model.train()

    metrics = create_metric_dict()

    for batch in dataloader:

        history_sids = batch[
            "history_sids"
        ].to(device)

        history_mask = batch[
            "history_mask"
        ].to(device)

        target_sids = batch[
            "target_sids"
        ].to(device)

        optimizer.zero_grad(
            set_to_none=True
        )

        # Forward
        model_output = model(
            history_sids=history_sids,
            history_mask=history_mask,
            target_sids=target_sids,
        )

        # Loss
        loss_output = loss_fn(
            c1_logits=model_output.c1_logits,
            c2_logits=model_output.c2_logits,
            c3_logits=model_output.c3_logits,
            c4_logits=model_output.c4_logits,
            target_sids=target_sids,
        )

        # Backpropagation
        loss_output.total_loss.backward()

        # Gradient clipping
        #
        # gradient가 지나치게 커지는 것을 방지.
        # None이면 사용하지 않음.
        if gradient_clip_norm is not None:

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=gradient_clip_norm,
            )

        # Parameter update
        optimizer.step()

        # Metric
        update_metrics(
            metrics=metrics,
            loss_output=loss_output,
            model_output=model_output,
            target_sids=target_sids,
        )

    return finalize_metrics(
        metrics
    )

# Validation
@torch.no_grad()
def evaluate(
    model: NewsEncoderDecoderTransformer,
    loss_fn: TransformerLoss,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Validation dataset에서 loss / accuracy 계산.

    model parameter는 업데이트X
    """

    model.eval()

    metrics = create_metric_dict()

    for batch in dataloader:

        history_sids = batch[
            "history_sids"
        ].to(device)

        history_mask = batch[
            "history_mask"
        ].to(device)

        target_sids = batch[
            "target_sids"
        ].to(device)

        # Forward
        model_output = model(
            history_sids=history_sids,
            history_mask=history_mask,
            target_sids=target_sids,
        )

        # Loss
        loss_output = loss_fn(
            c1_logits=model_output.c1_logits,
            c2_logits=model_output.c2_logits,
            c3_logits=model_output.c3_logits,
            c4_logits=model_output.c4_logits,
            target_sids=target_sids,
        )

        update_metrics(
            metrics=metrics,
            loss_output=loss_output,
            model_output=model_output,
            target_sids=target_sids,
        )

    return finalize_metrics(
        metrics
    )

def print_metrics(
    split_name: str,
    metrics: Dict[str, float],
) -> None:

    print(
        f"{split_name} "
        f"Loss={metrics['total_loss']:.6f} | "
        f"C1={metrics['c1_loss']:.6f} | "
        f"C2={metrics['c2_loss']:.6f} | "
        f"C3={metrics['c3_loss']:.6f} | "
        f"C4={metrics['c4_loss']:.6f}"
    )

    print(
        f"{split_name} "
        f"Acc | "
        f"C1={metrics['c1_acc']:.4f} | "
        f"C2={metrics['c2_acc']:.4f} | "
        f"C3={metrics['c3_acc']:.4f} | "
        f"C4={metrics['c4_acc']:.4f} | "
        f"Full SID={metrics['full_sid_acc']:.4f}"
    )


# Save checkpoint
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
        "epoch": epoch,

        "model_state_dict":
            model.state_dict(),

        "optimizer_state_dict":
            optimizer.state_dict(),

        "validation_loss":
            validation_loss,

        # 나중에 predict_sid.py에서
        # 어떤 gin 설정으로 학습했는지 확인 가능
        "gin_config":
            gin.config_str(),
    }

    torch.save(
        checkpoint,
        path,
    )

# Main Train
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
    """
    [트랜스포머 파라미터]
    
    train_path: train_sequences.parquet 경로

    validation_path: validation_sequences.parquet 경로

    save_dir: checkpoint 저장 폴더

    batch_size: 한 번에 학습할 sample 수

    num_epochs: 전체 train dataset을 몇 번 반복할지

    learning_rate: AdamW learning rate

    weight_decay: AdamW weight decay

    gradient_clip_norm: gradient clipping.(None이면 사용하지 않음.)

    num_workers: DataLoader worker 수.(Windows에서는 처음에는 0 권장한다고함)

    early_stopping_patience: validation loss가 개선되지 않는 epoch를 몇 번까지 기다릴지.
    """

    set_seed(
        seed
    )

    device = get_device()

    print()
    print("========================================")
    print("Transformer Training")
    print("========================================")
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
        parquet_path=str(train_path),
    )

    validation_dataset = NewsSequenceDataset(
        parquet_path=str(validation_path),
    )

    print()
    print(
        "Train samples      :",
        f"{len(train_dataset):,}",
    )

    print(
        "Validation samples :",
        f"{len(validation_dataset):,}",
    )


    train_loader = DataLoader(
        dataset=train_dataset,

        batch_size=batch_size,

        shuffle=True,

        num_workers=num_workers,

        collate_fn=collate_news_sequences,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    validation_loader = DataLoader(
        dataset=validation_dataset,

        batch_size=batch_size,

        shuffle=False,

        num_workers=num_workers,

        collate_fn=collate_news_sequences,

        pin_memory=(
            device.type == "cuda"
        ),
    )

    # Model
    model = (
        NewsEncoderDecoderTransformer()
        .to(device)
    )

    # Loss
    loss_fn = (
        TransformerLoss()
        .to(device)
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),

        lr=learning_rate,

        weight_decay=weight_decay,
    )

    # Model parameter count : 모델크기 확인용
    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print()
    print(
        "Total parameters     :",
        f"{total_parameters:,}",
    )

    print(
        "Trainable parameters :",
        f"{trainable_parameters:,}",
    )

    print()

    # 학습 전 validation loss도 출력
    print(
        "========================================"
    )

    print(
        "Before Training"
    )

    print(
        "========================================"
    )

    initial_validation_metrics = evaluate(
        model=model,
        loss_fn=loss_fn,
        dataloader=validation_loader,
        device=device,
    )

    print_metrics(
        "Validation",
        initial_validation_metrics,
    )

    # Training state
    best_validation_loss = float(
        "inf"
    )

    epochs_without_improvement = 0

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for epoch in range(
        1,
        num_epochs + 1,
    ):

        print()
        print(
            "========================================"
        )

        print(
            f"Epoch {epoch}/{num_epochs}"
        )

        print(
            "========================================"
        )

        # Train
        train_metrics = train_one_epoch(
            model=model,
            loss_fn=loss_fn,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            gradient_clip_norm=gradient_clip_norm,
        )

        print_metrics(
            "Train",
            train_metrics,
        )

        # Validation
        validation_metrics = evaluate(
            model=model,
            loss_fn=loss_fn,
            dataloader=validation_loader,
            device=device,
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

        # 매 epoch checkpoint
        epoch_checkpoint_path = (
            save_dir
            / f"checkpoint_epoch_{epoch}.pt"
        )

        save_checkpoint(
            path=epoch_checkpoint_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_loss=validation_loss,
        )

        # validation total loss 기준 Best checkpoint

        if (
            validation_loss
            < best_validation_loss
        ):

            best_validation_loss = (
                validation_loss
            )

            epochs_without_improvement = 0

            best_checkpoint_path = (
                save_dir
                / "checkpoint_best.pt"
            )

            save_checkpoint(
                path=best_checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_loss=validation_loss,
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

        # Early stopping
        if (
            early_stopping_patience
            is not None
        ):

            if (
                epochs_without_improvement
                >= early_stopping_patience
            ):

                print()
                print(
                    "Early stopping."
                )

                print(
                    f"No validation improvement for "
                    f"{early_stopping_patience} epochs."
                )

                break

    # Final checkpoint
    final_checkpoint_path = (
        save_dir
        / "checkpoint_final.pt"
    )

    save_checkpoint(
        path=final_checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=epoch,
        validation_loss=validation_loss,
    )

    print()
    print(
        "========================================"
    )

    print(
        "Training Finished"
    )

    print(
        "========================================"
    )

    print(
        "Best validation loss:",
        f"{best_validation_loss:.6f}",
    )

    print(
        "Best checkpoint:",
        save_dir / "checkpoint_best.pt",
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
        help=(
            "Gin config path. "
            "Example: "
            "configs/transformer_ebnerd.gin"
        ),
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