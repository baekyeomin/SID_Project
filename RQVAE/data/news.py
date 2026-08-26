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

    def __init__(self, data_dir: str, split: str = "train",):
        """
        data_dir:
            EB-NeRD 또는 MIND 데이터 폴더 경로
            (예: datasets/ebnerd)

        split: "train" 또는 "validation"
        """

        self.data_dir = Path(data_dir)
        self.split = split

        # 1. split에 따라 사용할 article_master 결정
        if split == "train":
            master_path = self.data_dir / "article_master.parquet"

        elif split == "validation":
            master_path = self.data_dir / "validation_article_master.parquet"
        
        else:
            raise ValueError(
                f"split은 'train' 또는 'validation'이어야 합니다. "
                f"현재 입력: {split}"
            )

        embedding_path = (
            self.data_dir / "article_embeddings.npy"
        )

        # 2. 파일 존재 여부 확인
        if not master_path.exists():
            raise FileNotFoundError(f"article master 파일을 찾을 수 없습니다: {master_path}")

        if not embedding_path.exists():
            raise FileNotFoundError(f"embedding 파일을 찾을 수 없습니다: {embedding_path}")

        # 3. article master 불러오기
        self.article_master = pd.read_parquet(master_path)

        # index를 0, 1, 2, ... 로 다시 맞춤
        self.article_master = self.article_master.reset_index(drop=True)

        # 4. 필수 컬럼 확인
        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(self.article_master.columns)
        )

        if missing_columns:
            raise ValueError(
                "article_master에 필요한 컬럼이 없습니다: "
                f"{sorted(missing_columns)}"
            )

        # 5. E5 article embeddings 불러오기
        #
        # mmap_mode="r": 전체 npy를 RAM에 한꺼번에 올리지 않고 필요한 행만 읽을 수 있게 
        self.article_embeddings = np.load(embedding_path,mmap_mode="r",)

        # 6. embedding shape 확인
        if self.article_embeddings.ndim != 2:
            raise ValueError(
                "article_embeddings.npy는 "
                "(기사 수, embedding_dim) 형태여야 합니다."
            )

        # E5 embedding = 768차원
        if self.article_embeddings.shape[1] != 768:
            raise ValueError(
                "현재 RQ-VAE 입력은 768차원으로 가정합니다. "
                f"실제 shape: {self.article_embeddings.shape}"
            )

        # 7. embedding_row 범위 확인
        embedding_rows = (self.article_master["embedding_row"].astype(int).to_numpy())

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

    def __len__(self): #Dataset에 포함된 기사 수
        return len(self.article_master)

    def __getitem__(self, idx): #기사 하나를 RQVAE 학습용 형태로 반환
        row = self.article_master.iloc[idx]

        # article_master에서 metadata 읽기
        article_id = row["article_id"]
        embedding_row = int(row["embedding_row"])
        category_id = int(row["model_category_id"])
        event_id = row["event_id"]


        # embedding_row를 이용하여
        # 실제 E5 article embedding x(a) 조회
        x = np.array(self.article_embeddings[embedding_row], dtype=np.float32,copy=True,)

        x = torch.from_numpy(x)

        return {
            "article_id": article_id,
            "x": x,
            "category_id": category_id,
            "event_id": event_id,
        }