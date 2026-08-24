from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from common import resolve_master_paths


def load_model(checkpoint_path: str | Path, device):
    from modules.rqvae import RqVae

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )
    if "model_config" not in checkpoint:
        raise KeyError("checkpoint에 model_config가 없습니다.")
    if "model" not in checkpoint:
        raise KeyError("checkpoint에 model state가 없습니다.")

    model = RqVae(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    for p in model.parameters():
        p.requires_grad_(False)

    return model


@torch.inference_mode()
def compute_reconstruction_loss(
    model,
    master_path,
    embeddings_path,
    device,
    batch_size=512,
    gumbel_t=0.001,
):
    master = pd.read_parquet(master_path).reset_index(drop=True)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    required = {"embedding_row", "model_category_id"}
    missing = required - set(master.columns)
    if missing:
        raise KeyError(f"{master_path}에 필요한 column이 없습니다: {sorted(missing)}")

    encoder_param = next(model.encoder.parameters())
    model_dtype = encoder_param.dtype

    total_loss = 0.0
    total_n = 0

    for start in range(0, len(master), batch_size):
        batch_df = master.iloc[start:start + batch_size]
        rows = batch_df["embedding_row"].astype(np.int64).to_numpy()

        x_np = np.asarray(embeddings[rows], dtype=np.float32)
        x = torch.from_numpy(x_np).to(device=device, dtype=model_dtype)

        category_ids = torch.tensor(
            batch_df["model_category_id"].astype(np.int64).to_numpy(),
            dtype=torch.long,
            device=device,
        )

        try:
            output = model(
                x=x,
                category_ids=category_ids,
                gumbel_t=gumbel_t,
            )
        except TypeError:
            output = model(x, category_ids, gumbel_t)

        if not hasattr(output, "reconstruction_loss"):
            raise AttributeError(
                "model output에 reconstruction_loss가 없습니다. "
                "현재 modules/rqvae.py의 output field 이름을 확인해주세요."
            )

        rec = output.reconstruction_loss.detach().float()
        n = len(batch_df)

        if rec.ndim == 0:
            total_loss += rec.item() * n
        else:
            total_loss += rec.sum().item()

        total_n += n

    return total_loss / total_n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--gumbel-t", type=float, default=0.001)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train_path, valid_path, embeddings_path = resolve_master_paths(data_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = load_model(args.checkpoint, device)

    train_loss = compute_reconstruction_loss(
        model,
        train_path,
        embeddings_path,
        device,
        batch_size=args.batch_size,
        gumbel_t=args.gumbel_t,
    )
    valid_loss = compute_reconstruction_loss(
        model,
        valid_path,
        embeddings_path,
        device,
        batch_size=args.batch_size,
        gumbel_t=args.gumbel_t,
    )

    print(f"Train Rec Loss : {train_loss:.6f}")
    print(f"Valid Rec Loss : {valid_loss:.6f}")


if __name__ == "__main__":
    main()
