# train valid sequence parquet 두개를 읽어서 
# history sid / target sid / user id / impression id를 반환하도록함

from __future__ import annotations

from typing import Any, Dict, List, Optional

import gin
import numpy as np
import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import Dataset

# Padding (빈자리 0) 적용함 -> 사용자마다 history 길이가 다르기 때문
# 패딩된 0은 실제 SID code로도 사용되지 않도록 history_mask=False(0)인 위치는 Transformer attention에서 무시
# 예를 들어 A mask = [1,1,1,0,0]면 index 3 4는 패당이니까 어텐션에서 무시
PAD_SID_VALUE = 0

NUM_SID_LEVELS = 4

#parquet에서 읽은 값을 list로 변환
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

    # scalar NaN 처리
    try:
        if pd.isna(value):
            return []
    except (TypeError, ValueError):
        pass
    
    # scalar 값이면 하나짜리 list로 변환
    return [value]


def _to_int_list(value: Any) -> List[int]:
    values = _to_list(value)
    return [int(v) for v in values]


@gin.configurable
class NewsSequenceDataset(Dataset):
    """
    Example
    
    parquet:

        history_c1 = [1, 1, 1]
        history_c2 = [2, 3, 5]
        history_c3 = [3, 2, 5]
        history_c4 = [0, 0, 1]

        target_c1 = [2]
        target_c2 = [6]
        target_c3 = [7]
        target_c4 = [0]

    output:

        history_sids =
        [
            [1, 2, 3, 0],
            [1, 3, 2, 0],
            [1, 5, 5, 1],
        ]

        target_sid =
        [2, 6, 7, 0]

 
    한 impression에서 클릭 target이 여러 개라면
    동일 history를 가진 별도의 training sample들로 둠

    예:

        history = A, B, C
        target  = [D, E] 라면

        sample 1: A,B,C -> D
        sample 2: A,B,C -> E
    """

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
            "target_c1",
            "target_c2",
            "target_c3",
            "target_c4",
        ]

        # optional metadata columns
        possible_optional_columns = [
            "impression_id",
            "user_id",
            "impression_time",
            "history_article_ids",
            "target_article_ids",
        ]

        # parquet schema 먼저 확인
        parquet_columns = pd.read_parquet(
            parquet_path,
        ).columns.tolist()

        # required column 확인
        missing_columns = [
            col
            for col in required_columns
            if col not in parquet_columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns in {parquet_path}: "
                f"{missing_columns}"
            )

        optional_columns = [
            col
            for col in possible_optional_columns
            if col in parquet_columns
        ]

        columns_to_load = required_columns + optional_columns

        # 실제 필요한 column만 로드 (메모리 절감)
        self.df = pd.read_parquet(
            parquet_path,
            columns=columns_to_load,
        ).reset_index(drop=True)

        # 한 row에 target이 여러 개 있을 수 있으므로
        # (row_index, target_index) 형태로 training sample 구성

        self.sample_indices: List[tuple[int, int]] = []

        self._build_sample_index()

        print(
            f"[NewsSequenceDataset]\n"
            f"  path            : {parquet_path}\n"
            f"  original rows   : {len(self.df):,}\n"
            f"  training samples: {len(self.sample_indices):,}\n"
            f"  max history len : {self.max_history_length}"
        )

    def _build_sample_index(self) -> None:
        """
        parquet row마다 target 개수를 확인하고
        실제 Dataset sample index를 만든다.

        target이 여러 개인 경우: 한 row → 여러 sample
        """

        for row_idx in range(len(self.df)):
            row = self.df.iloc[row_idx]
            history_c1 = _to_int_list(row["history_c1"])

            # history가 없는 sample 제거
            if self.drop_empty_history and len(history_c1) == 0:
                continue

            target_c1 = _to_int_list(row["target_c1"])
            target_c2 = _to_int_list(row["target_c2"])
            target_c3 = _to_int_list(row["target_c3"])
            target_c4 = _to_int_list(row["target_c4"])

            target_lengths = [
                len(target_c1),
                len(target_c2),
                len(target_c3),
                len(target_c4),
            ]

            # target이 하나도 없는 경우
            if min(target_lengths) == 0:
                continue

            # target level들의 길이가 모두 같아야 함 (4)
            if self.validate_data:
                if len(set(target_lengths)) != 1:
                    raise ValueError(
                        f"Target SID lengths do not match "
                        f"at row {row_idx}: "
                        f"{target_lengths}"
                    )

            num_targets = min(target_lengths)

            for target_idx in range(num_targets):
                self.sample_indices.append(
                    (row_idx, target_idx)
                )


    def __len__(self) -> int:
        return len(self.sample_indices)

    def __getitem__(self, index: int) -> Dict[str, Any]:

        row_idx, target_idx = self.sample_indices[index]

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

        if self.validate_data:
            if len(set(history_lengths)) != 1:
                raise ValueError(
                    f"History SID lengths do not match "
                    f"at parquet row {row_idx}: "
                    f"{history_lengths}"
                )

        history_length = min(history_lengths)

        # 최근 history만 사용할 건지 다할건지 ?

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

        # Target SID

        target_c1 = _to_int_list(row["target_c1"])
        target_c2 = _to_int_list(row["target_c2"])
        target_c3 = _to_int_list(row["target_c3"])
        target_c4 = _to_int_list(row["target_c4"])

        target_sid = torch.tensor(
            [
                target_c1[target_idx],
                target_c2[target_idx],
                target_c3[target_idx],
                target_c4[target_idx],
            ],
            dtype=torch.long,
        )

        # SID는 nn.Embedding index가 되므로 음수 있으면 안됨
        if self.validate_data:

            if (history_sids < 0).any():
                raise ValueError(
                    f"Negative SID found in history "
                    f"at parquet row {row_idx}."
                )

            if (target_sid < 0).any():
                raise ValueError(
                    f"Negative SID found in target "
                    f"at parquet row {row_idx}."
                )

        # Metadata

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

        # history article IDs
        history_article_ids = None

        if "history_article_ids" in self.df.columns:
            history_article_ids = _to_list(
                row["history_article_ids"]
            )

            if self.max_history_length is not None:
                history_article_ids = history_article_ids[
                    -self.max_history_length:
                ]

        # target article ID
        target_article_id = None

        if "target_article_ids" in self.df.columns:

            target_article_ids = _to_list(
                row["target_article_ids"]
            )

            if target_idx < len(target_article_ids):
                target_article_id = target_article_ids[
                    target_idx
                ]

        return {
            "history_sids": history_sids,
            "target_sid": target_sid,
            "history_length": history_length,

            # 나중에 prediction / evaluation에 사용
            "impression_id": impression_id,
            "user_id": user_id,
            "impression_time": impression_time,

            "history_article_ids": history_article_ids,
            "target_article_id": target_article_id,
        }



def collate_news_sequences(
    batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    길이가 서로 다른 user history를 하나의 batch로 padding
    [Input]
    sample 1 history shape : [5, 4] > 여기에 [0,0,0,0] 패딩이 3개 붙을 것 
    sample 2 history shape :  [8, 4]

    [Output]

    history_sids: [B, max_history_length, 4]
    history_mask: [B, max_history_length]

    target_sids:  [B, 4]

    history_mask=True(1): 실제 기사
    history_mask=False(0): padding
    """
    if len(batch) == 0:
        raise ValueError("Empty batch received.")

    batch_size = len(batch)

    history_lengths = [
        sample["history_sids"].shape[0]
        for sample in batch
    ]

    max_history_length = max(history_lengths)

    # 패딩 텐서
    history_sids = torch.full(
        (
            batch_size,
            max_history_length,
            NUM_SID_LEVELS,
        ),
        fill_value=PAD_SID_VALUE,
        dtype=torch.long,
    )

    # Attention mask
    # True  = 실제 기사
    # False = padding
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

    # Target  [B, 4]

    target_sids = torch.stack(
        [
            sample["target_sid"]
            for sample in batch
        ],
        dim=0,
    )

    history_lengths_tensor = torch.tensor(
        history_lengths,
        dtype=torch.long,
    )

    # Metadata는 Tensor로 바꾸지 않고 list로 유지
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
        sample["target_article_id"]
        for sample in batch
    ]

    return {
        # Transformer 입력
        "history_sids": history_sids,

        # Encoder attention mask
        "history_mask": history_mask,

        # 정답 next-news SID
        "target_sids": target_sids,

        "history_lengths": history_lengths_tensor,

        # prediction / evaluation용 metadata
        "impression_ids": impression_ids,
        "user_ids": user_ids,
        "impression_times": impression_times,
        "history_article_ids": history_article_ids,
        "target_article_ids": target_article_ids,
    }

# Debug / sanity check
if __name__ == "__main__":

    from torch.utils.data import DataLoader

    # 필요하면 테스트할 때 경로 수정
    dataset_path = "datasets/mind/train_sequences.parquet"

    dataset = NewsSequenceDataset(
        parquet_path=dataset_path,
        max_history_length=20,
    )

    print()
    print("Dataset size:", len(dataset))

    if len(dataset) > 0:

        sample = dataset[0]

        print()
        print("===== Single Sample =====")
        print(
            "history_sids shape:",
            sample["history_sids"].shape,
        )
        print(
            "history_sids:",
            sample["history_sids"],
        )
        print(
            "target_sid:",
            sample["target_sid"],
        )
        print(
            "target_article_id:",
            sample["target_article_id"],
        )

        loader = DataLoader(
            dataset,
            batch_size=4,
            shuffle=False,
            collate_fn=collate_news_sequences,
        )

        batch = next(iter(loader))

        print()
        print("===== Batch =====")

        print(
            "history_sids:",
            batch["history_sids"].shape,
        )

        print(
            "history_mask:",
            batch["history_mask"].shape,
        )

        print(
            "target_sids:",
            batch["target_sids"].shape,
        )

        print()
        print(
            "history_sids[0]:",
            batch["history_sids"][0],
        )

        print(
            "history_mask[0]:",
            batch["history_mask"][0],
        )

        print(
            "target_sids[0]:",
            batch["target_sids"][0],
        )