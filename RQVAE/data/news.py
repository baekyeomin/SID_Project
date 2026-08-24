from pathlib import Path

import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class NewsArticleDataset(Dataset):
    REQUIRED_COLUMNS = {
        "article_id",
        "embedding_row",
        "model_category_id",
        "event_id",
    }

    def __init__(self, data_dir: str, split: str = "train"):
        self.data_dir = Path(data_dir)
        self.split = split

        if split == "train":
            master_path = self.data_dir / "article_master.parquet"
        elif split == "validation":
            master_path = self.data_dir / "validation_article_master.parquet"
        else:
            raise ValueError(
                f"split은 'train' 또는 'validation'이어야 합니다. "
                f"현재 입력: {split}"
            )

        embedding_path = self.data_dir / "article_embeddings.npy"

        if not master_path.exists():
            raise FileNotFoundError(
                f"article master 파일을 찾을 수 없습니다: {master_path}"
            )

        if not embedding_path.exists():
            raise FileNotFoundError(
                f"embedding 파일을 찾을 수 없습니다: {embedding_path}"
            )

        self.article_master = pd.read_parquet(master_path)
        self.article_master = self.article_master.reset_index(drop=True)

        missing_columns = self.REQUIRED_COLUMNS - set(self.article_master.columns)

        if missing_columns:
            raise ValueError(
                "article_master에 필요한 컬럼이 없습니다: "
                f"{sorted(missing_columns)}"
            )

        self.article_embeddings = np.load(
            embedding_path,
            mmap_mode="r",
        )

        if self.article_embeddings.ndim != 2:
            raise ValueError(
                "article_embeddings.npy는 "
                "(기사 수, embedding_dim) 형태여야 합니다."
            )

        if self.article_embeddings.shape[1] != 768:
            raise ValueError(
                "현재 RQ-VAE 입력은 768차원으로 가정합니다. "
                f"실제 shape: {self.article_embeddings.shape}"
            )

        embedding_rows = (
            self.article_master["embedding_row"]
            .astype(int)
            .to_numpy()
        )

        if len(embedding_rows) > 0:
            if embedding_rows.min() < 0:
                raise ValueError(
                    "embedding_row에 음수가 존재합니다."
                )

            if embedding_rows.max() >= len(self.article_embeddings):
                raise ValueError(
                    "article_master의 embedding_row가 "
                    "article_embeddings.npy 범위를 벗어났습니다."
                )

    def __len__(self):
        return len(self.article_master)

    def __getitem__(self, idx):
        row = self.article_master.iloc[idx]

        article_id = row["article_id"]
        embedding_row = int(row["embedding_row"])
        category_id = int(row["model_category_id"])
        event_id = row["event_id"]

        x = np.array(
            self.article_embeddings[embedding_row],
            dtype=np.float32,
            copy=True,
        )

        x = torch.from_numpy(x)

        return {
            "article_id": article_id,
            "x": x,
            "category_id": category_id,
            "event_id": event_id,
        }