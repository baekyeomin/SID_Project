from __future__ import annotations

from typing import Any, Dict, List, Optional

import gin
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

PAD_SID_VALUE = 0
PAD_C4_VALUE = 0

NUM_HISTORY_SID_LEVELS = 4
NUM_CANDIDATE_SID_LEVELS = 3


def _to_list(value: Any) -> List:
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
    return [int(v) for v in _to_list(value)]


def _to_float_list(value: Any) -> List[float]:
    return [float(v) for v in _to_list(value)]


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

        required_columns = [
            "history_c1",
            "history_c2",
            "history_c3",
            "history_c4",
            "candidate_c1",
            "candidate_c2",
            "candidate_c3",
            "candidate_c4",
            "candidate_labels",
        ]

        possible_optional_columns = [
            "impression_id",
            "user_id",
            "impression_time",
            "history_article_ids",
            "target_article_ids",
            "candidate_article_ids",
        ]

        parquet_columns = pd.read_parquet(parquet_path).columns.tolist()

        missing_columns = [
            col for col in required_columns
            if col not in parquet_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {parquet_path}: {missing_columns}"
            )

        optional_columns = [
            col for col in possible_optional_columns
            if col in parquet_columns
        ]

        self.df = pd.read_parquet(
            parquet_path,
            columns=required_columns + optional_columns,
        ).reset_index(drop=True)

        self.sample_indices: List[int] = []
        self._build_sample_index()

        print(
            f"[NewsSequenceDataset]\n"
            f"  path            : {parquet_path}\n"
            f"  original rows   : {len(self.df):,}\n"
            f"  training samples: {len(self.sample_indices):,}\n"
            f"  max history len : {self.max_history_length}"
        )

    def _build_sample_index(self) -> None:
        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]

            history_c1 = _to_int_list(row["history_c1"])

            if self.drop_empty_history and len(history_c1) == 0:
                continue

            candidate_c1 = _to_int_list(row["candidate_c1"])
            candidate_labels = _to_float_list(row["candidate_labels"])

            if len(candidate_c1) == 0 or len(candidate_labels) == 0:
                continue

            self.sample_indices.append(row_idx)

    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row_idx = self.sample_indices[index]
        row = self.df.iloc[row_idx]

        history_c1 = _to_int_list(row["history_c1"])
        history_c2 = _to_int_list(row["history_c2"])
        history_c3 = _to_int_list(row["history_c3"])
        history_c4 = _to_int_list(row["history_c4"])

        history_lengths = [
            len(history_c1),
            len(history_c2),
            len(history_c3),
            len(history_c4),
        ]

        if self.validate_data and len(set(history_lengths)) != 1:
            raise ValueError(
                f"History SID lengths do not match at parquet row "
                f"{row_idx}: {history_lengths}"
            )

        if self.max_history_length is not None:
            history_c1 = history_c1[-self.max_history_length:]
            history_c2 = history_c2[-self.max_history_length:]
            history_c3 = history_c3[-self.max_history_length:]
            history_c4 = history_c4[-self.max_history_length:]

        history_length = len(history_c1)

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

        candidate_c1 = _to_int_list(row["candidate_c1"])
        candidate_c2 = _to_int_list(row["candidate_c2"])
        candidate_c3 = _to_int_list(row["candidate_c3"])
        candidate_c4 = _to_int_list(row["candidate_c4"])
        candidate_labels = _to_float_list(row["candidate_labels"])

        candidate_lengths = [
            len(candidate_c1),
            len(candidate_c2),
            len(candidate_c3),
            len(candidate_c4),
            len(candidate_labels),
        ]

        if self.validate_data:
            if min(candidate_lengths) == 0:
                raise ValueError(
                    f"Empty candidate found at parquet row {row_idx}."
                )

            if len(set(candidate_lengths)) != 1:
                raise ValueError(
                    f"Candidate lengths do not match at parquet row "
                    f"{row_idx}: {candidate_lengths}"
                )

            invalid_labels = [
                label for label in candidate_labels
                if label not in (0.0, 1.0)
            ]

            if invalid_labels:
                raise ValueError(
                    f"Candidate labels must be 0 or 1 at parquet row "
                    f"{row_idx}: {invalid_labels[:10]}"
                )

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

        candidate_c4_tensor = torch.tensor(
            candidate_c4,
            dtype=torch.long,
        )

        candidate_labels_tensor = torch.tensor(
            candidate_labels,
            dtype=torch.float32,
        )

        if self.validate_data:
            if (history_sids < 0).any():
                raise ValueError(
                    f"Negative SID found in history at parquet row {row_idx}."
                )

            if (candidate_sids < 0).any():
                raise ValueError(
                    f"Negative SID found in candidate at parquet row {row_idx}."
                )

            if (candidate_c4_tensor < 0).any():
                raise ValueError(
                    f"Negative c4 found in candidate at parquet row {row_idx}."
                )

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

        history_article_ids = None

        if "history_article_ids" in self.df.columns:
            history_article_ids = _to_list(
                row["history_article_ids"]
            )

            if self.max_history_length is not None:
                history_article_ids = history_article_ids[
                    -self.max_history_length:
                ]

        target_article_ids = None

        if "target_article_ids" in self.df.columns:
            target_article_ids = _to_list(
                row["target_article_ids"]
            )

        candidate_article_ids = None

        if "candidate_article_ids" in self.df.columns:
            candidate_article_ids = _to_list(
                row["candidate_article_ids"]
            )

            if (
                self.validate_data
                and len(candidate_article_ids) != candidate_sids.shape[0]
            ):
                raise ValueError(
                    f"candidate_article_ids length does not match "
                    f"candidate SID length at row {row_idx}: "
                    f"{len(candidate_article_ids)} vs "
                    f"{candidate_sids.shape[0]}"
                )

        return {
            "history_sids": history_sids,
            "candidate_sids": candidate_sids,
            "candidate_c4": candidate_c4_tensor,
            "candidate_labels": candidate_labels_tensor,
            "history_length": history_length,
            "impression_id": impression_id,
            "user_id": user_id,
            "impression_time": impression_time,
            "history_article_ids": history_article_ids,
            "target_article_ids": target_article_ids,
            "candidate_article_ids": candidate_article_ids,
        }


def collate_news_sequences(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    if len(batch) == 0:
        raise ValueError("Empty batch received.")

    batch_size = len(batch)

    history_lengths = [
        sample["history_sids"].shape[0]
        for sample in batch
    ]

    max_history_length = max(history_lengths)

    history_sids = torch.full(
        (
            batch_size,
            max_history_length,
            NUM_HISTORY_SID_LEVELS,
        ),
        fill_value=PAD_SID_VALUE,
        dtype=torch.long,
    )

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

    candidate_lengths = [
        sample["candidate_sids"].shape[0]
        for sample in batch
    ]

    max_candidate_length = max(candidate_lengths)

    candidate_sids = torch.full(
        (
            batch_size,
            max_candidate_length,
            NUM_CANDIDATE_SID_LEVELS,
        ),
        fill_value=PAD_SID_VALUE,
        dtype=torch.long,
    )

    candidate_c4 = torch.full(
        (
            batch_size,
            max_candidate_length,
        ),
        fill_value=PAD_C4_VALUE,
        dtype=torch.long,
    )

    candidate_labels = torch.zeros(
        (
            batch_size,
            max_candidate_length,
        ),
        dtype=torch.float32,
    )

    candidate_mask = torch.zeros(
        (
            batch_size,
            max_candidate_length,
        ),
        dtype=torch.bool,
    )

    for batch_idx, sample in enumerate(batch):
        candidates = sample["candidate_sids"]
        c4_values = sample["candidate_c4"]
        labels = sample["candidate_labels"]

        num_candidates = candidates.shape[0]

        candidate_sids[
            batch_idx,
            :num_candidates,
            :,
        ] = candidates

        candidate_c4[
            batch_idx,
            :num_candidates,
        ] = c4_values

        candidate_labels[
            batch_idx,
            :num_candidates,
        ] = labels

        candidate_mask[
            batch_idx,
            :num_candidates,
        ] = True

    candidate_lengths_tensor = torch.tensor(
        candidate_lengths,
        dtype=torch.long,
    )

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

    return {
        "history_sids": history_sids,
        "history_mask": history_mask,
        "candidate_sids": candidate_sids,
        "candidate_c4": candidate_c4,
        "candidate_labels": candidate_labels,
        "candidate_mask": candidate_mask,
        "history_lengths": history_lengths_tensor,
        "candidate_lengths": candidate_lengths_tensor,
        "impression_ids": impression_ids,
        "user_ids": user_ids,
        "impression_times": impression_times,
        "history_article_ids": history_article_ids,
        "target_article_ids": target_article_ids,
        "candidate_article_ids": candidate_article_ids,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    dataset_path = "datasets/ebnerd/train_sequences.parquet"

    dataset = NewsSequenceDataset(
        parquet_path=dataset_path,
        max_history_length=20,
    )

    print("Dataset size:", len(dataset))

    if len(dataset) > 0:
        sample = dataset[0]

        print("history_sids shape:", sample["history_sids"].shape)
        print("history_sids:", sample["history_sids"])
        print("candidate_sids shape:", sample["candidate_sids"].shape)
        print("candidate_sids:", sample["candidate_sids"])
        print("candidate_c4:", sample["candidate_c4"])
        print("candidate_labels:", sample["candidate_labels"])
        print("candidate_article_ids:", sample["candidate_article_ids"])

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_news_sequences,
        )

        batch = next(iter(loader))

        print("history_sids:", batch["history_sids"].shape)
        print("history_mask:", batch["history_mask"].shape)
        print("candidate_sids:", batch["candidate_sids"].shape)
        print("candidate_c4:", batch["candidate_c4"].shape)
        print("candidate_labels:", batch["candidate_labels"].shape)
        print("candidate_mask:", batch["candidate_mask"].shape)

        print("history_sids[0]:", batch["history_sids"][0])
        print("history_mask[0]:", batch["history_mask"][0])
        print("candidate_sids[0]:", batch["candidate_sids"][0])
        print("candidate_c4[0]:", batch["candidate_c4"][0])
        print("candidate_labels[0]:", batch["candidate_labels"][0])
        print("candidate_mask[0]:", batch["candidate_mask"][0])