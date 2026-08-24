from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd


def resolve_master_paths(data_dir: str | Path):
    data_dir = Path(data_dir)

    train_candidates = [
        data_dir / "article_master.parquet",
        data_dir / "train_article_master.parquet",
    ]
    valid_candidates = [
        data_dir / "validation_article_master.parquet",
        data_dir / "valid_article_master.parquet",
        data_dir / "val_article_master.parquet",
    ]
    embeddings_path = data_dir / "article_embeddings.npy"

    train_path = next((p for p in train_candidates if p.exists()), None)
    valid_path = next((p for p in valid_candidates if p.exists()), None)

    if train_path is None:
        raise FileNotFoundError(
            f"Train master를 찾지 못했습니다: {train_candidates}"
        )
    if valid_path is None:
        raise FileNotFoundError(
            f"Validation master를 찾지 못했습니다: {valid_candidates}"
        )
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embedding 파일을 찾지 못했습니다: {embeddings_path}")

    return train_path, valid_path, embeddings_path


def load_sid_eval_df(
    sid_path: str | Path,
    data_dir: str | Path,
    split: str = "all",
):
    sid = pd.read_parquet(sid_path).copy()
    train_path, valid_path, embeddings_path = resolve_master_paths(data_dir)

    train_master = pd.read_parquet(train_path).copy()
    valid_master = pd.read_parquet(valid_path).copy()

    train_master["split"] = "train"
    valid_master["split"] = "validation"
    master = pd.concat([train_master, valid_master], ignore_index=True)

    required_sid = {"article_id", "c1", "c2", "c3"}
    missing = required_sid - set(sid.columns)
    if missing:
        raise KeyError(f"SID 파일에 필요한 column이 없습니다: {sorted(missing)}")

    sid["article_id"] = sid["article_id"].astype(str)
    master["article_id"] = master["article_id"].astype(str)

    if split != "all":
        if "split" not in sid.columns:
            raise KeyError("split별 평가를 하려면 SID 파일에 split column이 필요합니다.")
        sid = sid[sid["split"] == split].copy()

    # 중복 article_id가 있으면 먼저 알려줌
    if sid["article_id"].duplicated().any():
        dup = sid.loc[sid["article_id"].duplicated(keep=False), "article_id"].head(10).tolist()
        raise ValueError(f"SID 파일에 중복 article_id가 있습니다. 예: {dup}")

    meta_cols = [
        c for c in [
            "article_id",
            "event_id",
            "embedding_row",
            "model_category_id",
            "split",
        ]
        if c in master.columns
    ]

    df = sid.merge(
        master[meta_cols],
        on="article_id",
        how="left",
        suffixes=("", "_master"),
        validate="one_to_one",
    )

    for col in ["embedding_row", "event_id", "model_category_id", "split"]:
        master_col = f"{col}_master"
        if master_col in df.columns:
            if col in df.columns:
                df[col] = df[col].fillna(df[master_col])
                df = df.drop(columns=[master_col])
            else:
                df = df.rename(columns={master_col: col})

    return df.reset_index(drop=True), embeddings_path


def print_result(result: dict):
    print("=" * 70)
    for key, value in result.items():
        if isinstance(value, float):
            if "Rate" in key or "Utilization" in key:
                print(f"{key:<30}: {value:.2%}")
            else:
                print(f"{key:<30}: {value:.6f}")
        else:
            print(f"{key:<30}: {value}")
    print("=" * 70)
