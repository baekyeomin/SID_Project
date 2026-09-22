from __future__ import annotations

from typing import Any, Dict, List, Optional

import gin
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


# ============================================================
# Constants
# ============================================================

PAD_SID_VALUE = 0

NUM_HISTORY_SID_LEVELS = 4      # c1, c2, c3, c4
NUM_CANDIDATE_SID_LEVELS = 3    # c1, c2, c3

NUM_CANDIDATES = 5              # positive 1 + negative 4
NUM_POSITIVES = 1


# ============================================================
# Utility functions
# ============================================================

def _to_list(value: Any) -> List:
    """
    parquet에서 읽은 값을 Python list로 변환한다.
    """
    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()

    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass

    return [value]


def _to_int_list(value: Any) -> List[int]:
    return [
        int(v)
        for v in _to_list(value)
    ]


def _to_float_list(value: Any) -> List[float]:
    return [
        float(v)
        for v in _to_list(value)
    ]


# ============================================================
# Dataset
# ============================================================

@gin.configurable
class NewsSequenceDataset(Dataset):

    def __init__(
        self,
        parquet_path: str,
        max_history_length: Optional[int] = None,
        drop_empty_history: bool = True,
        validate_data: bool = True,
    ) -> None:

        super().__init__()

        self.parquet_path = parquet_path
        self.max_history_length = max_history_length
        self.drop_empty_history = drop_empty_history
        self.validate_data = validate_data

        # ----------------------------------------------------
        # Transformer 학습에 반드시 필요한 column
        # ----------------------------------------------------

        required_columns = [
            "history_c1",
            "history_c2",
            "history_c3",
            "history_c4",
            "candidate_c1",
            "candidate_c2",
            "candidate_c3",
            "candidate_labels",
        ]

        # ----------------------------------------------------
        # 학습 자체에는 필수가 아니지만
        # 결과 확인/분석에 사용할 수 있는 column
        # ----------------------------------------------------

        possible_optional_columns = [
            "impression_id",
            "user_id",
            "impression_time",
            "history_article_ids",
            "target_article_ids",
            "candidate_article_ids",
        ]

        # ----------------------------------------------------
        # parquet 읽기
        # ----------------------------------------------------

        df = pd.read_parquet(parquet_path)

        parquet_columns = df.columns.tolist()

        missing_columns = [
            col
            for col in required_columns
            if col not in parquet_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in "
                f"{parquet_path}: {missing_columns}"
            )

        optional_columns = [
            col
            for col in possible_optional_columns
            if col in parquet_columns
        ]

        columns_to_keep = (
            required_columns
            + optional_columns
        )

        self.df = (
            df[columns_to_keep]
            .reset_index(drop=True)
        )

        # 실제 사용할 row index
        self.sample_indices: List[int] = []

        self._build_sample_index()

        print(
            f"[NewsSequenceDataset]\n"
            f"  path            : {parquet_path}\n"
            f"  original rows   : {len(self.df):,}\n"
            f"  usable samples  : {len(self.sample_indices):,}\n"
            f"  max history len : {self.max_history_length}\n"
            f"  candidates      : {NUM_CANDIDATES} "
            f"(1 positive + 4 negative)"
        )

    # ========================================================
    # 사용할 sample 결정
    # ========================================================

    def _build_sample_index(self) -> None:

        for row_idx, value in enumerate(
            self.df["history_c1"]
        ):

            history_c1 = _to_int_list(value)

            # history가 없는 사용자는 제외
            if (
                self.drop_empty_history
                and len(history_c1) == 0
            ):
                continue

            self.sample_indices.append(row_idx)

    # ========================================================
    # Dataset length
    # ========================================================

    def __len__(self) -> int:
        return len(self.sample_indices)

    # ========================================================
    # 한 sample 반환
    # ========================================================

    def __getitem__(
        self,
        index: int,
    ) -> Dict[str, Any]:

        row_idx = self.sample_indices[index]
        row = self.df.iloc[row_idx]

        # ====================================================
        # 1. History SID
        # ====================================================

        history_c1 = _to_int_list(
            row["history_c1"]
        )

        history_c2 = _to_int_list(
            row["history_c2"]
        )

        history_c3 = _to_int_list(
            row["history_c3"]
        )

        history_c4 = _to_int_list(
            row["history_c4"]
        )

        history_lengths = [
            len(history_c1),
            len(history_c2),
            len(history_c3),
            len(history_c4),
        ]

        # c1~c4 길이는 모두 같아야 함
        if (
            self.validate_data
            and len(set(history_lengths)) != 1
        ):
            raise ValueError(
                f"History SID lengths do not match "
                f"at parquet row {row_idx}: "
                f"{history_lengths}"
            )

        # 최근 history만 사용
        if self.max_history_length is not None:

            history_c1 = history_c1[
                -self.max_history_length:
            ]

            history_c2 = history_c2[
                -self.max_history_length:
            ]

            history_c3 = history_c3[
                -self.max_history_length:
            ]

            history_c4 = history_c4[
                -self.max_history_length:
            ]

        history_length = len(history_c1)

        # [history_length, 4]
        if history_length > 0:

            history_sids = torch.tensor(
                list(
                    zip(
                        history_c1,
                        history_c2,
                        history_c3,
                        history_c4,
                    )
                ),
                dtype=torch.long,
            )

        else:

            history_sids = torch.empty(
                (
                    0,
                    NUM_HISTORY_SID_LEVELS,
                ),
                dtype=torch.long,
            )

        # ====================================================
        # 2. Candidate SID
        #
        # candidate에서는 c1, c2, c3만 사용한다.
        # c4는 main score와 tie-breaking 모두 사용하지 않는다.
        # ====================================================

        candidate_c1 = _to_int_list(
            row["candidate_c1"]
        )

        candidate_c2 = _to_int_list(
            row["candidate_c2"]
        )

        candidate_c3 = _to_int_list(
            row["candidate_c3"]
        )

        candidate_labels = _to_float_list(
            row["candidate_labels"]
        )

        candidate_lengths = [
            len(candidate_c1),
            len(candidate_c2),
            len(candidate_c3),
            len(candidate_labels),
        ]

        # ====================================================
        # 3. Candidate 검증
        # ====================================================

        if self.validate_data:

            # 각 column 길이가 같아야 함
            if len(set(candidate_lengths)) != 1:

                raise ValueError(
                    f"Candidate lengths do not match "
                    f"at parquet row {row_idx}: "
                    f"{candidate_lengths}"
                )

            # 반드시 5개 후보
            if (
                candidate_lengths[0]
                != NUM_CANDIDATES
            ):

                raise ValueError(
                    f"Expected {NUM_CANDIDATES} candidates "
                    f"at parquet row {row_idx}, "
                    f"but found {candidate_lengths[0]}."
                )

            # label은 0 또는 1
            invalid_labels = [
                label
                for label in candidate_labels
                if label not in (0.0, 1.0)
            ]

            if invalid_labels:

                raise ValueError(
                    f"Candidate labels must be 0 or 1 "
                    f"at parquet row {row_idx}: "
                    f"{invalid_labels}"
                )

            # positive는 정확히 1개
            num_positives = sum(
                int(label == 1.0)
                for label in candidate_labels
            )

            if num_positives != NUM_POSITIVES:

                raise ValueError(
                    f"Expected exactly "
                    f"{NUM_POSITIVES} positive candidate "
                    f"at parquet row {row_idx}, "
                    f"but found {num_positives}."
                )

            # (c1,c2,c3)가 candidate 안에서
            # 서로 겹치지 않아야 함
            candidate_triplets = list(
                zip(
                    candidate_c1,
                    candidate_c2,
                    candidate_c3,
                )
            )

            if (
                len(set(candidate_triplets))
                != NUM_CANDIDATES
            ):

                raise ValueError(
                    f"Duplicate (c1,c2,c3) candidate "
                    f"found at parquet row {row_idx}: "
                    f"{candidate_triplets}"
                )

        # [5, 3]
        candidate_sids = torch.tensor(
            list(
                zip(
                    candidate_c1,
                    candidate_c2,
                    candidate_c3,
                )
            ),
            dtype=torch.long,
        )

        # [5]
        candidate_labels_tensor = torch.tensor(
            candidate_labels,
            dtype=torch.float32,
        )

        # ====================================================
        # 4. SID 값 검증
        # ====================================================

        if self.validate_data:

            if (
                history_sids.numel() > 0
                and (history_sids < 0).any()
            ):
                raise ValueError(
                    f"Negative SID found in history "
                    f"at parquet row {row_idx}."
                )

            if (candidate_sids < 0).any():

                raise ValueError(
                    f"Negative SID found in candidate "
                    f"at parquet row {row_idx}."
                )

        # ====================================================
        # 5. Optional metadata
        # ====================================================

        impression_id = (
            row["impression_id"]
            if "impression_id" in self.df.columns
            else None
        )

        user_id = (
            row["user_id"]
            if "user_id" in self.df.columns
            else None
        )

        impression_time = (
            row["impression_time"]
            if "impression_time" in self.df.columns
            else None
        )

        # ----------------------------------------------------
        # History article IDs
        # ----------------------------------------------------

        history_article_ids = None

        if "history_article_ids" in self.df.columns:

            history_article_ids = _to_list(
                row["history_article_ids"]
            )

            if self.max_history_length is not None:

                history_article_ids = (
                    history_article_ids[
                        -self.max_history_length:
                    ]
                )

            if (
                self.validate_data
                and len(history_article_ids)
                != history_length
            ):
                raise ValueError(
                    f"history_article_ids length does not "
                    f"match history SID length at row "
                    f"{row_idx}: "
                    f"{len(history_article_ids)} vs "
                    f"{history_length}"
                )

        # ----------------------------------------------------
        # Target article ID
        # ----------------------------------------------------

        target_article_ids = None

        if "target_article_ids" in self.df.columns:

            target_article_ids = _to_list(
                row["target_article_ids"]
            )

        # ----------------------------------------------------
        # Candidate article IDs
        # ----------------------------------------------------

        candidate_article_ids = None

        if "candidate_article_ids" in self.df.columns:

            candidate_article_ids = _to_list(
                row["candidate_article_ids"]
            )

            if (
                self.validate_data
                and len(candidate_article_ids)
                != NUM_CANDIDATES
            ):
                raise ValueError(
                    f"candidate_article_ids length does not "
                    f"match candidate SID length at row "
                    f"{row_idx}: "
                    f"{len(candidate_article_ids)} vs "
                    f"{NUM_CANDIDATES}"
                )

        # ====================================================
        # 반환
        # ====================================================

        return {
            "history_sids": history_sids,

            "candidate_sids": candidate_sids,
            "candidate_labels": candidate_labels_tensor,

            "history_length": history_length,

            "impression_id": impression_id,
            "user_id": user_id,
            "impression_time": impression_time,

            "history_article_ids": history_article_ids,
            "target_article_ids": target_article_ids,
            "candidate_article_ids": candidate_article_ids,
        }


# ============================================================
# Collate function
# ============================================================

def collate_news_sequences(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:

    if len(batch) == 0:
        raise ValueError(
            "Empty batch received."
        )

    batch_size = len(batch)

    # ========================================================
    # 1. History padding
    # ========================================================

    history_lengths = [
        sample["history_sids"].shape[0]
        for sample in batch
    ]

    max_history_length = max(
        history_lengths
    )

    # [B, max_history, 4]
    history_sids = torch.full(
        (
            batch_size,
            max_history_length,
            NUM_HISTORY_SID_LEVELS,
        ),
        fill_value=PAD_SID_VALUE,
        dtype=torch.long,
    )

    # [B, max_history]
    history_mask = torch.zeros(
        (
            batch_size,
            max_history_length,
        ),
        dtype=torch.bool,
    )

    for batch_idx, sample in enumerate(batch):

        seq = sample["history_sids"]
        seq_len = seq.shape[0]

        if seq_len == 0:
            continue

        history_sids[
            batch_idx,
            :seq_len,
            :,
        ] = seq

        history_mask[
            batch_idx,
            :seq_len,
        ] = True

    history_lengths_tensor = torch.tensor(
        history_lengths,
        dtype=torch.long,
    )

    # ========================================================
    # 2. Candidate
    #
    # 모든 sample이 이미 후보 5개이므로
    # candidate padding / mask가 필요 없다.
    # ========================================================

    # [B, 5, 3]
    candidate_sids = torch.stack(
        [
            sample["candidate_sids"]
            for sample in batch
        ],
        dim=0,
    )

    # [B, 5]
    candidate_labels = torch.stack(
        [
            sample["candidate_labels"]
            for sample in batch
        ],
        dim=0,
    )

    # ========================================================
    # 3. Metadata
    # ========================================================

    impression_ids = [
        sample["impression_id"]
        for sample in batch
    ]

    user_ids = [
        sample["user_id"]
        for sample in batch
    ]

    impression_times = [
        sample["impression_time"]
        for sample in batch
    ]

    history_article_ids = [
        sample["history_article_ids"]
        for sample in batch
    ]

    target_article_ids = [
        sample["target_article_ids"]
        for sample in batch
    ]

    candidate_article_ids = [
        sample["candidate_article_ids"]
        for sample in batch
    ]

    # ========================================================
    # 4. Batch 반환
    # ========================================================

    return {
        "history_sids": history_sids,
        "history_mask": history_mask,
        "history_lengths": history_lengths_tensor,

        "candidate_sids": candidate_sids,
        "candidate_labels": candidate_labels,

        "impression_ids": impression_ids,
        "user_ids": user_ids,
        "impression_times": impression_times,

        "history_article_ids": history_article_ids,
        "target_article_ids": target_article_ids,
        "candidate_article_ids": candidate_article_ids,
    }


# ============================================================
# Simple test
# ============================================================

if __name__ == "__main__":

    from torch.utils.data import DataLoader

    dataset_path = (
        "datasets/ebnerd/"
        "train_sequences_1pos4neg.parquet"
    )

    dataset = NewsSequenceDataset(
        parquet_path=dataset_path,
        max_history_length=20,
        drop_empty_history=True,
        validate_data=True,
    )

    print()
    print(
        "Dataset size:",
        len(dataset),
    )

    if len(dataset) > 0:

        # ----------------------------------------------------
        # Single sample
        # ----------------------------------------------------

        sample = dataset[0]

        print()
        print("===== SINGLE SAMPLE =====")

        print(
            "history_sids shape:",
            sample["history_sids"].shape,
        )

        print(
            "candidate_sids shape:",
            sample["candidate_sids"].shape,
        )

        print(
            "candidate_sids:",
            sample["candidate_sids"],
        )

        print(
            "candidate_labels:",
            sample["candidate_labels"],
        )

        print(
            "candidate_article_ids:",
            sample["candidate_article_ids"],
        )

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_news_sequences,
        )

        batch = next(
            iter(loader)
        )

        print()
        print("===== BATCH =====")

        print(
            "history_sids:",
            batch["history_sids"].shape,
        )

        print(
            "history_mask:",
            batch["history_mask"].shape,
        )

        print(
            "candidate_sids:",
            batch["candidate_sids"].shape,
        )

        print(
            "candidate_labels:",
            batch["candidate_labels"].shape,
        )

        print()
        print(
            "candidate_sids[0]:",
            batch["candidate_sids"][0],
        )

        print(
            "candidate_labels[0]:",
            batch["candidate_labels"][0],
        )