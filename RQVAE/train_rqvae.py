import gin
import importlib
import math
import os
import random

import numpy as np
import torch
import wandb

from accelerate import Accelerator
from sklearn.cluster import KMeans
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.news import NewsArticleDataset
from modules.rqvae import RqVae
from modules.quantize import QuantizeForwardMode
from modules.utils import parse_config


# ============================================================
# PyTorch serialization compatibility
# ============================================================

def ensure_torch_serialization_compatibility() -> None:
    if not hasattr(torch, "Tensor"):
        tensor_class = None

        try:
            tensor_module = importlib.import_module("torch._tensor")
            tensor_class = getattr(tensor_module, "Tensor", None)
        except Exception:
            tensor_class = None

        if tensor_class is None:
            try:
                tensor_class = type(torch.empty(0))
            except Exception as exc:
                raise RuntimeError(
                    "torch.Tensor attribute를 복구하지 못했습니다."
                ) from exc

        setattr(torch, "Tensor", tensor_class)

    if not hasattr(torch, "_utils"):
        try:
            torch_utils_module = importlib.import_module("torch._utils")
            setattr(torch, "_utils", torch_utils_module)
        except Exception as exc:
            raise RuntimeError(
                "torch._utils attribute를 복구하지 못했습니다."
            ) from exc

    if not hasattr(torch, "Tensor"):
        raise RuntimeError("torch.Tensor is still unavailable.")

    if not hasattr(torch, "_utils"):
        raise RuntimeError("torch._utils is still unavailable.")


ensure_torch_serialization_compatibility()


def safe_torch_save(state, path: str) -> None:
    ensure_torch_serialization_compatibility()

    temp_path = path + ".tmp"

    if os.path.exists(temp_path):
        os.remove(temp_path)

    try:
        torch.save(state, temp_path)
        os.replace(temp_path, path)
    except Exception:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise


def safe_torch_load(path: str, map_location=None):
    ensure_torch_serialization_compatibility()

    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


def test_torch_serialization(save_dir_root: str) -> None:
    os.makedirs(save_dir_root, exist_ok=True)

    test_path = os.path.join(
        save_dir_root,
        "_serialization_test.pt",
    )

    test_state = {
        "tensor": torch.zeros(2, dtype=torch.float32)
    }

    safe_torch_save(test_state, test_path)
    loaded = safe_torch_load(test_path, map_location="cpu")

    if "tensor" not in loaded:
        raise RuntimeError(
            "torch serialization smoke test failed."
        )

    if os.path.exists(test_path):
        os.remove(test_path)

    print("PyTorch checkpoint save/load test: OK")


# ============================================================
# Utility
# ============================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def unpack_batch(batch):
    x = batch["x"]

    if "category_id" in batch:
        category_ids = batch["category_id"]
    elif "model_category_id" in batch:
        category_ids = batch["model_category_id"]
    else:
        raise KeyError(
            "Batch must contain 'category_id' or "
            "'model_category_id'."
        )

    if "event_id" not in batch:
        raise KeyError(
            "Batch must contain 'event_id'."
        )

    event_ids = batch["event_id"]

    return x, category_ids, event_ids


def get_gumbel_temperature(
    global_step: int,
    gumbel_t0: float,
    gumbel_min_t: float,
    gumbel_anneal_rate: float,
) -> float:
    """
    기존 iteration 기반 annealing을 그대로 유지하기 위해
    epoch 번호가 아니라 optimizer update 횟수(global_step)를 사용한다.

    STE / Rotation Trick에서는 temperature가 실질적으로 사용되지 않는다.
    """

    temperature = (
        gumbel_t0
        * math.exp(
            -gumbel_anneal_rate
            * global_step
        )
    )

    return max(
        gumbel_min_t,
        temperature,
    )


def print_train_config(
    epochs,
    batch_size,
    learning_rate,
    weight_decay,
    gradient_accumulate_every,
    dataset_folder,
    vae_input_dim,
    vae_hidden_dims,
    vae_embed_dim,
    vae_num_categories,
    vae_c2_codebook_size,
    vae_c3_codebook_size,
    vae_codebook_normalize,
    vae_sim_vq,
    vae_codebook_mode,
    lambda_rec,
    lambda_cb,
    lambda_com,
    gumbel_t0,
    gumbel_min_t,
    gumbel_anneal_rate,
    amp,
    mixed_precision_type,
    do_eval,
    eval_every_epochs,
    save_model_every_epochs,
    save_dir_root,
    seed,
):
    print("\n" + "=" * 70)
    print("RQ-VAE TRAIN CONFIG")
    print("=" * 70)
    print(f"dataset_folder             : {dataset_folder}")

    print("\n[Training]")
    print(f"epochs                     : {epochs}")
    print(f"batch_size                 : {batch_size}")
    print(f"learning_rate              : {learning_rate}")
    print(f"weight_decay               : {weight_decay}")
    print(
        "gradient_accumulate_every  : "
        f"{gradient_accumulate_every}"
    )

    print("\n[Network]")
    print(f"vae_input_dim              : {vae_input_dim}")
    print(f"vae_hidden_dims            : {vae_hidden_dims}")
    print(f"vae_embed_dim              : {vae_embed_dim}")

    print("\n[Codebooks]")
    print(
        "Q1 num_categories          : "
        f"{vae_num_categories}"
    )
    print(
        "Q2 codebook size           : "
        f"{vae_c2_codebook_size}"
    )
    print(
        "Q3 codebook size           : "
        f"{vae_c3_codebook_size}"
    )

    print("\n[Quantization]")
    print(
        "codebook_normalize         : "
        f"{vae_codebook_normalize}"
    )
    print(f"sim_vq                     : {vae_sim_vq}")
    print(f"forward_mode               : {vae_codebook_mode}")
    print(f"gumbel_t0                  : {gumbel_t0}")
    print(f"gumbel_min_t               : {gumbel_min_t}")
    print(
        "gumbel_anneal_rate         : "
        f"{gumbel_anneal_rate}"
    )

    print("\n[Loss]")
    print(f"lambda_rec                 : {lambda_rec}")
    print(f"lambda_cb                  : {lambda_cb}")
    print(f"lambda_com                 : {lambda_com}")

    print("\n[Runtime]")
    print(f"amp                        : {amp}")
    print(
        "mixed_precision_type       : "
        f"{mixed_precision_type}"
    )
    print(f"do_eval                    : {do_eval}")
    print(
        "eval_every_epochs          : "
        f"{eval_every_epochs}"
    )
    print(
        "save_model_every_epochs    : "
        f"{save_model_every_epochs}"
    )
    print(f"seed                       : {seed}")
    print(f"save_dir_root              : {save_dir_root}")
    print("=" * 70 + "\n")


# ============================================================
# Q2 K-means initialization
# ============================================================

@torch.no_grad()
def encode_all_train_articles(
    model: RqVae,
    train_dataset: NewsArticleDataset,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
):
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )

    was_training = model.training
    model.eval()

    h_list = []
    event_id_list = []

    for batch in tqdm(
        loader,
        desc=(
            "Encoding train articles "
            "for Q2 initialization"
        ),
    ):
        x, _, event_ids = unpack_batch(batch)

        x = x.to(
            device=device,
            dtype=next(
                model.encoder.parameters()
            ).dtype,
        )

        h = model.encode(x)

        h_list.append(
            h.detach().cpu()
        )
        event_id_list.append(
            event_ids.detach().cpu()
        )

    h_all = torch.cat(
        h_list,
        dim=0,
    )

    event_ids_all = torch.cat(
        event_id_list,
        dim=0,
    )

    if was_training:
        model.train()

    return h_all, event_ids_all


@torch.no_grad()
def compute_event_representations(
    h_all: torch.Tensor,
    event_ids_all: torch.Tensor,
):
    unique_event_ids, inverse_indices = torch.unique(
        event_ids_all,
        sorted=True,
        return_inverse=True,
    )

    num_events = unique_event_ids.shape[0]
    embed_dim = h_all.shape[1]

    event_sums = torch.zeros(
        num_events,
        embed_dim,
        dtype=h_all.dtype,
    )

    event_sums.index_add_(
        0,
        inverse_indices,
        h_all,
    )

    event_counts = torch.zeros(
        num_events,
        dtype=h_all.dtype,
    )

    ones = torch.ones(
        event_ids_all.shape[0],
        dtype=h_all.dtype,
    )

    event_counts.index_add_(
        0,
        inverse_indices,
        ones,
    )

    z_events = (
        event_sums
        / event_counts
        .unsqueeze(1)
        .clamp_min(1.0)
    )

    return unique_event_ids, z_events


@torch.no_grad()
def initialize_c2_codebook(
    model: RqVae,
    train_dataset: NewsArticleDataset,
    device: torch.device,
    c2_codebook_size: int,
    encode_batch_size: int = 512,
    kmeans_n_init: int = 10,
    seed: int = 42,
):
    print("\n" + "=" * 70)
    print("Q2 CODEBOOK INITIALIZATION")
    print("=" * 70)

    h_all, event_ids_all = encode_all_train_articles(
        model=model,
        train_dataset=train_dataset,
        device=device,
        batch_size=encode_batch_size,
    )

    print(f"Train articles : {h_all.shape[0]}")

    unique_event_ids, z_events = (
        compute_event_representations(
            h_all=h_all,
            event_ids_all=event_ids_all,
        )
    )

    print(f"Train events   : {z_events.shape[0]}")
    print(
        "z(E) shape     : "
        f"{tuple(z_events.shape)}"
    )

    if z_events.shape[0] < c2_codebook_size:
        raise ValueError(
            "Number of train events must be >= "
            "C2 codebook size. "
            f"events={z_events.shape[0]}, "
            "c2_codebook_size="
            f"{c2_codebook_size}"
        )

    print(
        "Running K-means "
        f"(k={c2_codebook_size})..."
    )

    kmeans = KMeans(
        n_clusters=c2_codebook_size,
        init="k-means++",
        n_init=kmeans_n_init,
        random_state=seed,
    )

    kmeans.fit(
        z_events.numpy()
    )

    centroids = torch.from_numpy(
        kmeans.cluster_centers_
    ).float()

    print(
        "Centroid shape : "
        f"{tuple(centroids.shape)}"
    )

    model.set_c2_codebook(
        centroids.to(device)
    )

    print("Q2 codebook initialized.")
    print("=" * 70 + "\n")

    return (
        unique_event_ids,
        z_events,
        centroids,
    )


# ============================================================
# Evaluation / loss printing
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    gumbel_t,
):
    """
    전체 dataloader에 대해 아래 loss의 dataset 평균을 계산한다.

    - loss: 최종 total loss
    - reconstruction_loss
    - codebook_loss
    - commitment_loss
    - rqvae_loss

    batch 크기가 마지막에 달라질 수 있으므로
    batch size로 가중해서 평균한다.
    """

    model.eval()

    sums = {
        "loss": 0.0,
        "reconstruction_loss": 0.0,
        "codebook_loss": 0.0,
        "commitment_loss": 0.0,
        "rqvae_loss": 0.0,
    }

    total_samples = 0

    for batch in dataloader:
        x, category_ids, _ = unpack_batch(batch)

        x = x.to(
            device=device,
            non_blocking=True,
        )

        category_ids = category_ids.to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        batch_size_now = x.shape[0]

        output = model(
            x=x,
            category_ids=category_ids,
            gumbel_t=gumbel_t,
        )

        sums["loss"] += (
            output.loss
            .detach()
            .float()
            .item()
            * batch_size_now
        )

        sums["reconstruction_loss"] += (
            output.reconstruction_loss
            .detach()
            .float()
            .item()
            * batch_size_now
        )

        sums["codebook_loss"] += (
            output.codebook_loss
            .detach()
            .float()
            .item()
            * batch_size_now
        )

        sums["commitment_loss"] += (
            output.commitment_loss
            .detach()
            .float()
            .item()
            * batch_size_now
        )

        sums["rqvae_loss"] += (
            output.rqvae_loss
            .detach()
            .float()
            .item()
            * batch_size_now
        )

        total_samples += batch_size_now

    if total_samples == 0:
        raise RuntimeError(
            "Evaluation dataloader is empty."
        )

    return {
        key: value / total_samples
        for key, value in sums.items()
    }


def print_loss_result(
    name: str,
    result,
) -> None:
    print(f"[{name}]")
    print(
        "  total loss          : "
        f"{result['loss']:.6f}"
    )
    print(
        "  reconstruction loss : "
        f"{result['reconstruction_loss']:.6f}"
    )
    print(
        "  codebook loss       : "
        f"{result['codebook_loss']:.6f}"
    )
    print(
        "  commitment loss     : "
        f"{result['commitment_loss']:.6f}"
    )
    print(
        "  rqvae loss          : "
        f"{result['rqvae_loss']:.6f}"
    )


def print_loss_snapshot(
    title: str,
    train_result,
    validation_result=None,
) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)

    print_loss_result(
        "TRAIN",
        train_result,
    )

    if validation_result is not None:
        print()
        print_loss_result(
            "VALIDATION",
            validation_result,
        )

    print("=" * 70 + "\n")


# ============================================================
# Checkpoint
# ============================================================

def build_checkpoint_state(
    accelerator,
    model,
    optimizer,
    epoch,
    global_step,
    lambda_rec,
    lambda_cb,
    lambda_com,
):
    unwrapped_model = accelerator.unwrap_model(
        model
    )

    state = {
        # 새 epoch 기반 정보
        "epoch": epoch,
        "global_step": global_step,

        # 기존 generate_semantic_ids.py 등과의 호환을 위해 유지
        # 기존 코드에서 iter는 0-based 마지막 update index였다.
        "iter": max(global_step - 1, -1),

        "model": unwrapped_model.state_dict(),
        "model_config": unwrapped_model.config,
        "optimizer": optimizer.state_dict(),
        "loss_weights": {
            "lambda_rec": lambda_rec,
            "lambda_cb": lambda_cb,
            "lambda_com": lambda_com,
        },
        "gin_config": gin.operative_config_str(),
    }

    return state


# ============================================================
# Training
# ============================================================

@gin.configurable
def train(
    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------
    epochs: int = 327,
    batch_size: int = 64,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    gradient_accumulate_every: int = 1,

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------
    dataset_folder: str = "datasets/ebnerd",

    # --------------------------------------------------------
    # Network
    # --------------------------------------------------------
    vae_input_dim: int = 768,
    vae_hidden_dims=[512, 256],
    vae_embed_dim: int = 128,
    vae_num_categories: int = 25,
    vae_c2_codebook_size: int = 256,
    vae_c3_codebook_size: int = 256,

    # --------------------------------------------------------
    # Quantization
    # --------------------------------------------------------
    vae_codebook_normalize: bool = False,
    vae_sim_vq: bool = False,
    vae_codebook_mode=QuantizeForwardMode.STE,

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    lambda_rec: float = 1.0,
    lambda_cb: float = 1.0,
    lambda_com: float = 0.25,

    # --------------------------------------------------------
    # Q2 K-means initialization
    # --------------------------------------------------------
    kmeans_encode_batch_size: int = 512,
    kmeans_n_init: int = 10,

    # --------------------------------------------------------
    # Gumbel temperature schedule
    # --------------------------------------------------------
    gumbel_t0: float = 1.0,
    gumbel_min_t: float = 0.1,
    gumbel_anneal_rate: float = 5.8e-5,

    # --------------------------------------------------------
    # Runtime
    # --------------------------------------------------------
    split_batches: bool = True,
    amp: bool = True,
    mixed_precision_type: str = "fp16",
    num_workers: int = 0,

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------
    do_eval: bool = True,
    eval_every_epochs: int = 10,

    # --------------------------------------------------------
    # Checkpoint
    # --------------------------------------------------------
    pretrained_rqvae_path=None,
    save_dir_root: str = "out/rqvae/ebnerd",
    save_model_every_epochs: int = 50,

    # --------------------------------------------------------
    # Logging
    # --------------------------------------------------------
    wandb_logging: bool = False,
    wandb_project: str = "news-rqvae-training",

    # --------------------------------------------------------
    # Misc
    # --------------------------------------------------------
    seed: int = 42,
):
    if epochs <= 0:
        raise ValueError(
            "epochs must be > 0."
        )

    if gradient_accumulate_every <= 0:
        raise ValueError(
            "gradient_accumulate_every must be > 0."
        )

    if eval_every_epochs <= 0:
        raise ValueError(
            "eval_every_epochs must be > 0."
        )

    if save_model_every_epochs <= 0:
        raise ValueError(
            "save_model_every_epochs must be > 0."
        )

    ensure_torch_serialization_compatibility()
    set_seed(seed)

    accelerator = Accelerator(
        split_batches=split_batches,
        mixed_precision=(
            mixed_precision_type
            if amp
            else "no"
        ),
    )

    device = accelerator.device
    print(f"Device: {device}")

    os.makedirs(
        save_dir_root,
        exist_ok=True,
    )

    if accelerator.is_main_process:
        test_torch_serialization(
            save_dir_root
        )

    accelerator.wait_for_everyone()

    print_train_config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        gradient_accumulate_every=(
            gradient_accumulate_every
        ),
        dataset_folder=dataset_folder,
        vae_input_dim=vae_input_dim,
        vae_hidden_dims=vae_hidden_dims,
        vae_embed_dim=vae_embed_dim,
        vae_num_categories=(
            vae_num_categories
        ),
        vae_c2_codebook_size=(
            vae_c2_codebook_size
        ),
        vae_c3_codebook_size=(
            vae_c3_codebook_size
        ),
        vae_codebook_normalize=(
            vae_codebook_normalize
        ),
        vae_sim_vq=vae_sim_vq,
        vae_codebook_mode=(
            vae_codebook_mode
        ),
        lambda_rec=lambda_rec,
        lambda_cb=lambda_cb,
        lambda_com=lambda_com,
        gumbel_t0=gumbel_t0,
        gumbel_min_t=gumbel_min_t,
        gumbel_anneal_rate=(
            gumbel_anneal_rate
        ),
        amp=amp,
        mixed_precision_type=(
            mixed_precision_type
        ),
        do_eval=do_eval,
        eval_every_epochs=(
            eval_every_epochs
        ),
        save_model_every_epochs=(
            save_model_every_epochs
        ),
        save_dir_root=save_dir_root,
        seed=seed,
    )

    # ========================================================
    # Dataset / DataLoader
    # ========================================================

    train_dataset = NewsArticleDataset(
        data_dir=dataset_folder,
        split="train",
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    # 학습 전/후 전체 Train loss를 안정적으로 계산하기 위한 loader
    train_eval_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type == "cuda"
        ),
    )

    print(
        "Train articles: "
        f"{len(train_dataset)}"
    )

    if do_eval:
        eval_dataset = NewsArticleDataset(
            data_dir=dataset_folder,
            split="validation",
        )

        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=(
                device.type == "cuda"
            ),
        )

        print(
            "Validation articles: "
            f"{len(eval_dataset)}"
        )
    else:
        eval_dataset = None
        eval_dataloader = None

    batches_per_epoch = len(
        train_dataloader
    )

    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch
        / gradient_accumulate_every
    )

    print(
        "Batches per epoch: "
        f"{batches_per_epoch}"
    )
    print(
        "Optimizer steps per epoch: "
        f"{optimizer_steps_per_epoch}"
    )
    print(
        "Planned optimizer updates: "
        f"{epochs * optimizer_steps_per_epoch}"
    )

    # ========================================================
    # Input sanity check
    # ========================================================

    first_item = train_dataset[0]
    first_x = first_item["x"]
    actual_input_dim = first_x.shape[-1]

    if actual_input_dim != vae_input_dim:
        raise ValueError(
            "Input dimension mismatch. "
            "Dataset x(a) dimension="
            f"{actual_input_dim}, "
            "vae_input_dim="
            f"{vae_input_dim}"
        )

    if (
        "category_id" not in first_item
        and "model_category_id" not in first_item
    ):
        raise KeyError(
            "Dataset item must contain "
            "'category_id' or "
            "'model_category_id'."
        )

    # ========================================================
    # Model
    # ========================================================

    model = RqVae(
        input_dim=vae_input_dim,
        embed_dim=vae_embed_dim,
        hidden_dims=vae_hidden_dims,
        num_categories=(
            vae_num_categories
        ),
        c2_codebook_size=(
            vae_c2_codebook_size
        ),
        c3_codebook_size=(
            vae_c3_codebook_size
        ),
        codebook_normalize=(
            vae_codebook_normalize
        ),
        codebook_sim_vq=(
            vae_sim_vq
        ),
        codebook_mode=(
            vae_codebook_mode
        ),
        lambda_rec=lambda_rec,
        lambda_cb=lambda_cb,
        lambda_com=lambda_com,
    )

    model = model.to(device)

    checkpoint_state = None
    start_epoch = 0
    global_step = 0

    # ========================================================
    # Resume or fresh Q2 initialization
    # ========================================================

    if pretrained_rqvae_path is not None:
        checkpoint_state = safe_torch_load(
            pretrained_rqvae_path,
            map_location=device,
        )

        model.load_state_dict(
            checkpoint_state["model"]
        )

        global_step = int(
            checkpoint_state.get(
                "global_step",
                checkpoint_state.get(
                    "iter",
                    -1,
                ) + 1,
            )
        )

        if "epoch" in checkpoint_state:
            start_epoch = int(
                checkpoint_state["epoch"]
            ) + 1
        else:
            # 옛 iteration checkpoint는 정확한 epoch 위치를 모르므로
            # global_step을 현재 steps/epoch 기준으로 환산한다.
            start_epoch = (
                global_step
                // optimizer_steps_per_epoch
            )

            print(
                "WARNING: old iteration-based "
                "checkpoint detected. "
                "start_epoch was estimated from "
                "global_step."
            )

        print(
            "Loaded checkpoint: "
            f"{pretrained_rqvae_path}"
        )
        print(
            "Resume epoch: "
            f"{start_epoch + 1}"
        )
        print(
            "Resume global_step: "
            f"{global_step}"
        )
    else:
        initialize_c2_codebook(
            model=model,
            train_dataset=train_dataset,
            device=device,
            c2_codebook_size=(
                vae_c2_codebook_size
            ),
            encode_batch_size=(
                kmeans_encode_batch_size
            ),
            kmeans_n_init=(
                kmeans_n_init
            ),
            seed=seed,
        )

    # ========================================================
    # Optimizer
    # ========================================================

    optimizer = AdamW(
        params=model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    if (
        checkpoint_state is not None
        and "optimizer" in checkpoint_state
    ):
        optimizer.load_state_dict(
            checkpoint_state["optimizer"]
        )

    # ========================================================
    # Accelerate prepare
    # ========================================================

    if do_eval:
        (
            model,
            optimizer,
            train_dataloader,
            train_eval_dataloader,
            eval_dataloader,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            train_eval_dataloader,
            eval_dataloader,
        )
    else:
        (
            model,
            optimizer,
            train_dataloader,
            train_eval_dataloader,
        ) = accelerator.prepare(
            model,
            optimizer,
            train_dataloader,
            train_eval_dataloader,
        )

    # ========================================================
    # W&B
    # ========================================================

    if (
        wandb_logging
        and accelerator.is_main_process
    ):
        wandb.login()

        wandb.init(
            project=wandb_project,
            config={
                "epochs": epochs,
                "batch_size": batch_size,
                "learning_rate": (
                    learning_rate
                ),
                "weight_decay": weight_decay,
                "gradient_accumulate_every": (
                    gradient_accumulate_every
                ),
                "vae_input_dim": vae_input_dim,
                "vae_hidden_dims": (
                    vae_hidden_dims
                ),
                "vae_embed_dim": vae_embed_dim,
                "num_categories": (
                    vae_num_categories
                ),
                "c2_codebook_size": (
                    vae_c2_codebook_size
                ),
                "c3_codebook_size": (
                    vae_c3_codebook_size
                ),
                "codebook_mode": str(
                    vae_codebook_mode
                ),
                "lambda_rec": lambda_rec,
                "lambda_cb": lambda_cb,
                "lambda_com": lambda_com,
                "gumbel_t0": gumbel_t0,
                "gumbel_min_t": (
                    gumbel_min_t
                ),
                "gumbel_anneal_rate": (
                    gumbel_anneal_rate
                ),
                "seed": seed,
            },
        )

        wandb.define_metric("epoch")
        wandb.define_metric(
            "train/*",
            step_metric="epoch",
        )
        wandb.define_metric(
            "eval/*",
            step_metric="epoch",
        )
        wandb.define_metric(
            "gumbel_temperature",
            step_metric="epoch",
        )
        wandb.define_metric(
            "global_step",
            step_metric="epoch",
        )

    # ========================================================
    # Save operative Gin config
    # ========================================================

    if accelerator.is_main_process:
        gin_config_path = os.path.join(
            save_dir_root,
            "operative_config.gin",
        )

        with open(
            gin_config_path,
            "w",
            encoding="utf-8",
        ) as f:
            f.write(
                gin.operative_config_str()
            )

    # ========================================================
    # BEFORE TRAINING LOSS
    # ========================================================

    accelerator.wait_for_everyone()

    initial_gumbel_t = get_gumbel_temperature(
        global_step=global_step,
        gumbel_t0=gumbel_t0,
        gumbel_min_t=gumbel_min_t,
        gumbel_anneal_rate=(
            gumbel_anneal_rate
        ),
    )

    initial_train_result = evaluate(
        model=model,
        dataloader=(
            train_eval_dataloader
        ),
        device=device,
        gumbel_t=initial_gumbel_t,
    )

    if do_eval:
        initial_eval_result = evaluate(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            gumbel_t=initial_gumbel_t,
        )
    else:
        initial_eval_result = None

    if accelerator.is_main_process:
        print_loss_snapshot(
            title=(
                "LOSS BEFORE TRAINING "
                "(after Q2 initialization / "
                "loaded checkpoint)"
            ),
            train_result=(
                initial_train_result
            ),
            validation_result=(
                initial_eval_result
            ),
        )

    # ========================================================
    # Epoch training loop
    # ========================================================

    if start_epoch >= epochs:
        print(
            "start_epoch is already >= epochs. "
            "No additional training will run."
        )

    for epoch in range(
        start_epoch,
        epochs,
    ):
        model.train()

        epoch_sums = {
            "loss": 0.0,
            "reconstruction_loss": 0.0,
            "codebook_loss": 0.0,
            "commitment_loss": 0.0,
            "rqvae_loss": 0.0,
        }

        epoch_samples = 0
        optimizer.zero_grad(
            set_to_none=True
        )

        batch_pbar = tqdm(
            enumerate(train_dataloader),
            total=len(train_dataloader),
            desc=(
                f"Epoch {epoch + 1}/{epochs}"
            ),
            leave=False,
            disable=(
                not accelerator.is_main_process
            ),
        )

        last_gumbel_t = get_gumbel_temperature(
            global_step=global_step,
            gumbel_t0=gumbel_t0,
            gumbel_min_t=gumbel_min_t,
            gumbel_anneal_rate=(
                gumbel_anneal_rate
            ),
        )

        for batch_idx, batch in batch_pbar:
            # 마지막 gradient accumulation group이 짧아도
            # gradient scale이 정확하도록 실제 group 크기를 계산한다.
            group_start = (
                batch_idx
                // gradient_accumulate_every
                * gradient_accumulate_every
            )

            group_end = min(
                group_start
                + gradient_accumulate_every,
                len(train_dataloader),
            )

            current_group_size = (
                group_end
                - group_start
            )

            last_gumbel_t = (
                get_gumbel_temperature(
                    global_step=global_step,
                    gumbel_t0=gumbel_t0,
                    gumbel_min_t=(
                        gumbel_min_t
                    ),
                    gumbel_anneal_rate=(
                        gumbel_anneal_rate
                    ),
                )
            )

            x, category_ids, _ = unpack_batch(
                batch
            )

            x = x.to(
                device=device,
                non_blocking=True,
            )

            category_ids = category_ids.to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )

            batch_size_now = x.shape[0]

            with accelerator.autocast():
                model_output = model(
                    x=x,
                    category_ids=(
                        category_ids
                    ),
                    gumbel_t=last_gumbel_t,
                )

                loss_for_backward = (
                    model_output.loss
                    / current_group_size
                )

            accelerator.backward(
                loss_for_backward
            )

            epoch_sums["loss"] += (
                model_output.loss
                .detach()
                .float()
                .item()
                * batch_size_now
            )

            epoch_sums[
                "reconstruction_loss"
            ] += (
                model_output
                .reconstruction_loss
                .detach()
                .float()
                .item()
                * batch_size_now
            )

            epoch_sums[
                "codebook_loss"
            ] += (
                model_output
                .codebook_loss
                .detach()
                .float()
                .item()
                * batch_size_now
            )

            epoch_sums[
                "commitment_loss"
            ] += (
                model_output
                .commitment_loss
                .detach()
                .float()
                .item()
                * batch_size_now
            )

            epoch_sums[
                "rqvae_loss"
            ] += (
                model_output
                .rqvae_loss
                .detach()
                .float()
                .item()
                * batch_size_now
            )

            epoch_samples += batch_size_now

            should_step = (
                (batch_idx + 1)
                % gradient_accumulate_every
                == 0
                or (batch_idx + 1)
                == len(train_dataloader)
            )

            if should_step:
                optimizer.step()
                optimizer.zero_grad(
                    set_to_none=True
                )
                global_step += 1

            current_avg_loss = (
                epoch_sums["loss"]
                / epoch_samples
            )

            current_avg_rec = (
                epoch_sums[
                    "reconstruction_loss"
                ]
                / epoch_samples
            )

            batch_pbar.set_postfix(
                loss=f"{current_avg_loss:.4f}",
                rec=f"{current_avg_rec:.4f}",
                t=f"{last_gumbel_t:.4f}",
            )

        epoch_result = {
            key: value / epoch_samples
            for key, value
            in epoch_sums.items()
        }

        if accelerator.is_main_process:
            print(
                f"[Epoch {epoch + 1:4d}/{epochs}] "
                f"loss={epoch_result['loss']:.6f} | "
                "rec="
                f"{epoch_result['reconstruction_loss']:.6f} | "
                "cb="
                f"{epoch_result['codebook_loss']:.6f} | "
                "com="
                f"{epoch_result['commitment_loss']:.6f} | "
                "rqvae="
                f"{epoch_result['rqvae_loss']:.6f} | "
                f"t={last_gumbel_t:.6f} | "
                f"global_step={global_step}"
            )

        if (
            wandb_logging
            and accelerator.is_main_process
        ):
            wandb_log = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "gumbel_temperature": (
                    last_gumbel_t
                ),
                "train/total_loss": (
                    epoch_result["loss"]
                ),
                "train/reconstruction_loss": (
                    epoch_result[
                        "reconstruction_loss"
                    ]
                ),
                "train/codebook_loss": (
                    epoch_result[
                        "codebook_loss"
                    ]
                ),
                "train/commitment_loss": (
                    epoch_result[
                        "commitment_loss"
                    ]
                ),
                "train/rqvae_loss": (
                    epoch_result[
                        "rqvae_loss"
                    ]
                ),
            }

            if hasattr(
                model_output,
                "p_unique_ids",
            ):
                wandb_log[
                    "train/p_unique_ids"
                ] = (
                    model_output
                    .p_unique_ids
                    .detach()
                    .float()
                    .cpu()
                    .item()
                )

            wandb.log(
                wandb_log
            )

        # ----------------------------------------------------
        # Periodic validation by epoch
        # ----------------------------------------------------

        should_eval = (
            do_eval
            and (
                (epoch + 1)
                % eval_every_epochs
                == 0
                or (epoch + 1) == epochs
            )
        )

        if should_eval:
            accelerator.wait_for_everyone()

            eval_gumbel_t = (
                get_gumbel_temperature(
                    global_step=global_step,
                    gumbel_t0=gumbel_t0,
                    gumbel_min_t=(
                        gumbel_min_t
                    ),
                    gumbel_anneal_rate=(
                        gumbel_anneal_rate
                    ),
                )
            )

            eval_result = evaluate(
                model=model,
                dataloader=eval_dataloader,
                device=device,
                gumbel_t=eval_gumbel_t,
            )

            if accelerator.is_main_process:
                print(
                    f"[Validation epoch {epoch + 1}] "
                    f"loss={eval_result['loss']:.6f} | "
                    "rec="
                    f"{eval_result['reconstruction_loss']:.6f} | "
                    "cb="
                    f"{eval_result['codebook_loss']:.6f} | "
                    "com="
                    f"{eval_result['commitment_loss']:.6f} | "
                    "rqvae="
                    f"{eval_result['rqvae_loss']:.6f}"
                )

                if wandb_logging:
                    wandb.log(
                        {
                            "epoch": epoch + 1,
                            "global_step": (
                                global_step
                            ),
                            "eval/total_loss": (
                                eval_result["loss"]
                            ),
                            "eval/reconstruction_loss": (
                                eval_result[
                                    "reconstruction_loss"
                                ]
                            ),
                            "eval/codebook_loss": (
                                eval_result[
                                    "codebook_loss"
                                ]
                            ),
                            "eval/commitment_loss": (
                                eval_result[
                                    "commitment_loss"
                                ]
                            ),
                            "eval/rqvae_loss": (
                                eval_result[
                                    "rqvae_loss"
                                ]
                            ),
                        }
                    )

        # ----------------------------------------------------
        # Checkpoint save by epoch
        # ----------------------------------------------------

        should_save = (
            (epoch + 1)
            % save_model_every_epochs
            == 0
        )

        if should_save:
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                state = build_checkpoint_state(
                    accelerator=accelerator,
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    global_step=global_step,
                    lambda_rec=lambda_rec,
                    lambda_cb=lambda_cb,
                    lambda_com=lambda_com,
                )

                checkpoint_path = os.path.join(
                    save_dir_root,
                    (
                        "checkpoint_epoch_"
                        f"{epoch + 1}.pt"
                    ),
                )

                safe_torch_save(
                    state,
                    checkpoint_path,
                )

                print(
                    "Saved checkpoint: "
                    f"{checkpoint_path}"
                )

    # ========================================================
    # AFTER TRAINING LOSS
    # ========================================================

    accelerator.wait_for_everyone()

    final_gumbel_t = get_gumbel_temperature(
        global_step=global_step,
        gumbel_t0=gumbel_t0,
        gumbel_min_t=gumbel_min_t,
        gumbel_anneal_rate=(
            gumbel_anneal_rate
        ),
    )

    final_train_result = evaluate(
        model=model,
        dataloader=train_eval_dataloader,
        device=device,
        gumbel_t=final_gumbel_t,
    )

    if do_eval:
        final_eval_result = evaluate(
            model=model,
            dataloader=eval_dataloader,
            device=device,
            gumbel_t=final_gumbel_t,
        )
    else:
        final_eval_result = None

    if accelerator.is_main_process:
        print_loss_snapshot(
            title="LOSS AFTER TRAINING",
            train_result=final_train_result,
            validation_result=(
                final_eval_result
            ),
        )

        print("=" * 70)
        print("LOSS CHANGE: BEFORE -> AFTER")
        print("=" * 70)

        for key, label in [
            ("loss", "total"),
            (
                "reconstruction_loss",
                "reconstruction",
            ),
            ("codebook_loss", "codebook"),
            (
                "commitment_loss",
                "commitment",
            ),
            ("rqvae_loss", "rqvae"),
        ]:
            before = initial_train_result[key]
            after = final_train_result[key]
            diff = after - before

            print(
                f"Train {label:14s}: "
                f"{before:.6f} -> "
                f"{after:.6f} "
                f"({diff:+.6f})"
            )

        if final_eval_result is not None:
            print()

            for key, label in [
                ("loss", "total"),
                (
                    "reconstruction_loss",
                    "reconstruction",
                ),
                (
                    "codebook_loss",
                    "codebook",
                ),
                (
                    "commitment_loss",
                    "commitment",
                ),
                ("rqvae_loss", "rqvae"),
            ]:
                before = initial_eval_result[key]
                after = final_eval_result[key]
                diff = after - before

                print(
                    f"Valid {label:14s}: "
                    f"{before:.6f} -> "
                    f"{after:.6f} "
                    f"({diff:+.6f})"
                )

        print("=" * 70 + "\n")

    # ========================================================
    # Final checkpoint
    # ========================================================

    accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        final_epoch_index = (
            epochs - 1
            if epochs > 0
            else -1
        )

        final_state = build_checkpoint_state(
            accelerator=accelerator,
            model=model,
            optimizer=optimizer,
            epoch=final_epoch_index,
            global_step=global_step,
            lambda_rec=lambda_rec,
            lambda_cb=lambda_cb,
            lambda_com=lambda_com,
        )

        final_path = os.path.join(
            save_dir_root,
            "checkpoint_final.pt",
        )

        safe_torch_save(
            final_state,
            final_path,
        )

        print("\n" + "=" * 70)
        print(
            "Final model saved: "
            f"{final_path}"
        )
        print(f"Final epoch: {epochs}")
        print(
            "Final global_step: "
            f"{global_step}"
        )
        print("=" * 70)

    if (
        wandb_logging
        and accelerator.is_main_process
    ):
        wandb.finish()


if __name__ == "__main__":
    parse_config()
    train()
