import argparse
import importlib
from pathlib import Path
from typing import Optional, Set

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from modules.rqvae import RqVae


# ============================================================
# PyTorch serialization compatibility
# ============================================================

def ensure_torch_serialization_compatibility() -> None:
    if not hasattr(torch, "Tensor"):
        tensor_class = None

        try:
            tensor_module = importlib.import_module(
                "torch._tensor"
            )
            tensor_class = getattr(
                tensor_module,
                "Tensor",
                None,
            )
        except Exception:
            tensor_class = None

        if tensor_class is None:
            try:
                tensor_class = type(
                    torch.empty(0)
                )
            except Exception as exc:
                raise RuntimeError(
                    "torch.Tensor attribute를 "
                    "복구하지 못했습니다."
                ) from exc

        setattr(
            torch,
            "Tensor",
            tensor_class,
        )

    if not hasattr(
        torch,
        "_utils",
    ):
        try:
            torch_utils = (
                importlib.import_module(
                    "torch._utils"
                )
            )

            setattr(
                torch,
                "_utils",
                torch_utils,
            )

        except Exception as exc:
            raise RuntimeError(
                "torch._utils attribute를 "
                "복구하지 못했습니다."
            ) from exc


def safe_torch_load(
    checkpoint_path,
    map_location,
):
    ensure_torch_serialization_compatibility()

    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


# ============================================================
# Dataset
# ============================================================

class SemanticIdDataset(Dataset):

    REQUIRED_COLUMNS = {
        "article_id",
        "embedding_row",
        "model_category_id",
        "event_id",
    }

    def __init__(
        self,
        master_path: str,
        embeddings_path: str,
    ) -> None:

        super().__init__()

        self.master_path = Path(
            master_path
        )

        self.embeddings_path = Path(
            embeddings_path
        )

        self.master = pd.read_parquet(
            self.master_path
        ).reset_index(
            drop=True
        )

        missing_columns = (
            self.REQUIRED_COLUMNS
            - set(
                self.master.columns
            )
        )

        if missing_columns:
            raise ValueError(
                f"{self.master_path} "
                "is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        if (
            self.master[
                "event_id"
            ]
            .isna()
            .any()
        ):
            missing_count = int(
                self.master[
                    "event_id"
                ]
                .isna()
                .sum()
            )

            raise ValueError(
                f"{self.master_path} contains "
                f"{missing_count} missing "
                "event_id values."
            )

        self.embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r",
        )

        if (
            self.embeddings.ndim
            != 2
        ):
            raise ValueError(
                "article_embeddings.npy "
                "must have shape "
                "[num_articles, embedding_dim]. "
                f"Got {self.embeddings.shape}."
            )

        rows = (
            self.master[
                "embedding_row"
            ]
            .to_numpy()
        )

        if len(rows) > 0:
            min_row = int(
                rows.min()
            )

            max_row = int(
                rows.max()
            )

            if min_row < 0:
                raise ValueError(
                    "embedding_row contains "
                    "negative values."
                )

            if (
                max_row
                >= len(
                    self.embeddings
                )
            ):
                raise ValueError(
                    "embedding_row points outside "
                    "article_embeddings.npy. "
                    f"Maximum embedding_row="
                    f"{max_row}, "
                    f"but embeddings contain "
                    f"{len(self.embeddings)} rows."
                )

    def __len__(
        self,
    ) -> int:

        return len(
            self.master
        )

    def __getitem__(
        self,
        index: int,
    ):

        row = (
            self.master
            .iloc[index]
        )

        embedding_row = int(
            row[
                "embedding_row"
            ]
        )

        category_id = int(
            row[
                "model_category_id"
            ]
        )

        event_id = int(
            row[
                "event_id"
            ]
        )

        embedding = np.asarray(
            self.embeddings[
                embedding_row
            ],
            dtype=np.float32,
        ).copy()

        x = torch.from_numpy(
            embedding
        )

        return {
            "x": x,

            "category_id": torch.tensor(
                category_id,
                dtype=torch.long,
            ),

            "event_id": torch.tensor(
                event_id,
                dtype=torch.long,
            ),

            "article_id": str(
                row[
                    "article_id"
                ]
            ),

            "embedding_row": torch.tensor(
                embedding_row,
                dtype=torch.long,
            ),
        }


# ============================================================
# Load trained RQ-VAE
# ============================================================

def load_rqvae(
    checkpoint_path: str,
    device: torch.device,
) -> RqVae:

    checkpoint_path = Path(
        checkpoint_path
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            "RQ-VAE checkpoint "
            f"not found: {checkpoint_path}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "LOADING TRAINED RQ-VAE"
    )

    print(
        "=" * 70
    )

    checkpoint = safe_torch_load(
        checkpoint_path=(
            checkpoint_path
        ),
        map_location=device,
    )

    if (
        "model_config"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'model_config'."
        )

    if (
        "model"
        not in checkpoint
    ):
        raise KeyError(
            "Checkpoint does not contain "
            "'model'."
        )

    model_config = (
        checkpoint[
            "model_config"
        ]
    )

    print(
        "Model config:"
    )

    for (
        key,
        value,
    ) in (
        model_config.items()
    ):
        print(
            f"  {key}: {value}"
        )

    model = RqVae(
        **model_config
    )

    model.load_state_dict(
        checkpoint[
            "model"
        ]
    )

    model = model.to(
        device
    )

    # --------------------------------------------------------
    # 학습 완료된 모델 전체 고정
    # --------------------------------------------------------

    model.eval()

    for parameter in (
        model.parameters()
    ):
        parameter.requires_grad_(
            False
        )

    print(
        f"\nLoaded checkpoint: "
        f"{checkpoint_path}"
    )

    if (
        "epoch"
        in checkpoint
    ):
        print(
            "Training epoch: "
            f"{checkpoint['epoch'] + 1}"
        )

    if (
        "global_step"
        in checkpoint
    ):
        print(
            "Training global step: "
            f"{checkpoint['global_step']}"
        )

    elif (
        "iter"
        in checkpoint
    ):
        print(
            "Training iteration: "
            f"{checkpoint['iter']}"
        )

    print(
        "RQ-VAE frozen."
    )

    print(
        "=" * 70
        + "\n"
    )

    return model


# ============================================================
# Dataset sanity check
# ============================================================

def validate_dataset_against_model(
    dataset: SemanticIdDataset,
    model: RqVae,
) -> None:

    embedding_dim = (
        dataset
        .embeddings
        .shape[1]
    )

    if (
        embedding_dim
        != model.input_dim
    ):
        raise ValueError(
            "Article embedding dimension "
            "does not match RQ-VAE "
            "input_dim. "
            "article_embeddings.npy="
            f"{embedding_dim}, "
            "RQ-VAE input_dim="
            f"{model.input_dim}"
        )

    if (
        len(
            dataset.master
        )
        > 0
    ):
        category_values = (
            dataset.master[
                "model_category_id"
            ]
            .astype(int)
        )

        min_category = int(
            category_values.min()
        )

        max_category = int(
            category_values.max()
        )

        if (
            min_category
            < 0
        ):
            raise ValueError(
                "model_category_id "
                "must be >= 0. "
                "Found minimum="
                f"{min_category}."
            )

        if (
            max_category
            >= model.num_categories
        ):
            raise ValueError(
                "model_category_id exceeds "
                "Q1 codebook range. "
                "Maximum category ID="
                f"{max_category}, "
                "Q1 size="
                f"{model.num_categories}. "
                "Category IDs must be in "
                f"[0, "
                f"{model.num_categories - 1}]."
            )


# ============================================================
# Train article SID
# ============================================================

@torch.inference_mode()
def generate_train_semantic_ids(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
) -> pd.DataFrame:
    """
    학습이 완료된 final frozen RQ-VAE를 이용하여
    Train article의 최종 (c1, c2, c3)를 한 번 확정한다.

    Train article은 기존 RQ-VAE 방식 그대로:
        c1 = category ID
        Q2 = article-level nearest
        Q3 = article-level nearest

    Validation inference 단계에서는 이 Train article SID를
    다시 생성하지 않고 재사용할 수 있다.
    """

    dataset = SemanticIdDataset(
        master_path=master_path,
        embeddings_path=embeddings_path,
    )

    validate_dataset_against_model(
        dataset=dataset,
        model=model,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    article_ids_all = []
    embedding_rows_all = []
    event_ids_all = []

    c1_all = []
    c2_all = []
    c3_all = []

    print(
        "\nGenerating final "
        "Train Semantic IDs..."
    )

    print(
        f"Articles: {len(dataset)}"
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        dataloader
    ):

        x = (
            batch[
                "x"
            ]
            .to(
                device=device,
                dtype=next(
                    model
                    .encoder
                    .parameters()
                ).dtype,
                non_blocking=True,
            )
        )

        category_ids = (
            batch[
                "category_id"
            ]
            .to(
                device=device,
                dtype=torch.long,
                non_blocking=True,
            )
        )

        # ----------------------------------------------------
        # Train:
        # fixed_c2_ids 없음
        #
        # final frozen model 기준
        # article-level C2 / C3 결정
        # ----------------------------------------------------

        output = (
            model.get_semantic_ids(
                x=x,
                category_ids=(
                    category_ids
                ),
            )
        )

        semantic_ids = (
            output
            .sem_ids
            .detach()
            .cpu()
        )

        article_ids_all.extend(
            batch[
                "article_id"
            ]
        )

        embedding_rows_all.extend(
            batch[
                "embedding_row"
            ]
            .cpu()
            .tolist()
        )

        event_ids_all.extend(
            batch[
                "event_id"
            ]
            .cpu()
            .tolist()
        )

        c1_all.extend(
            semantic_ids[
                :,
                0,
            ].tolist()
        )

        c2_all.extend(
            semantic_ids[
                :,
                1,
            ].tolist()
        )

        c3_all.extend(
            semantic_ids[
                :,
                2,
            ].tolist()
        )

        if (
            (
                batch_index
                + 1
            )
            % 20
            == 0
        ):
            processed = min(
                (
                    batch_index
                    + 1
                )
                * batch_size,
                len(dataset),
            )

            print(
                f"  {processed}/"
                f"{len(dataset)}"
            )

    result = pd.DataFrame({
        "article_id": (
            article_ids_all
        ),
        "embedding_row": (
            embedding_rows_all
        ),
        "event_id": (
            event_ids_all
        ),
        "split": "train",
        "c1": c1_all,
        "c2": c2_all,
        "c3": c3_all,
    })

    if (
        len(result)
        != len(dataset)
    ):
        raise RuntimeError(
            "Number of generated "
            "Train Semantic IDs does "
            "not match number of articles. "
            f"articles={len(dataset)}, "
            f"semantic_ids={len(result)}"
        )

    if (
        result[
            "article_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Train result contains "
            "duplicated article_id."
        )

    print(
        f"Train: "
        f"{len(result)} final SID "
        "rows generated."
    )

    return result


# ============================================================
# Event representation
#
# z(E) = mean h(a)
#
# 교수님 설계:
# event representation은 Q2 input인 r1을 그대로
# 흉내내기 위한 값이 아니라,
# category 의미까지 포함한 사건 자체를 잘 표현하는 역할.
#
# 따라서 h(a)를 평균낸다.
# ============================================================

@torch.inference_mode()
def compute_event_c2_from_mean_h(
    model: RqVae,
    dataset: SemanticIdDataset,
    device: torch.device,
    allowed_event_ids: Optional[
        Set[int]
    ] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    source: str = "mean_h_frozen_q2",
) -> pd.DataFrame:
    """
    event별 representation:

        z(E) = mean_{a in E} h(a)

    를 만든 뒤 학습 완료된 frozen Q2 codebook에서
    nearest search하여 event-level C2를 결정한다.

    allowed_event_ids:
        None -> dataset의 모든 event 처리
        set  -> 해당 event들만 처리
    """

    validate_dataset_against_model(
        dataset=dataset,
        model=model,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    event_sums = {}
    event_counts = {}

    print(
        "\nBuilding event representations "
        "z(E) = mean(h(a))..."
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        dataloader
    ):

        batch_event_ids = [
            int(event_id)
            for event_id
            in (
                batch[
                    "event_id"
                ]
                .cpu()
                .tolist()
            )
        ]

        # ----------------------------------------------------
        # 필요한 event의 기사 position만 선택
        # ----------------------------------------------------

        if (
            allowed_event_ids
            is None
        ):
            selected_positions = list(
                range(
                    len(
                        batch_event_ids
                    )
                )
            )

        else:
            selected_positions = [
                index
                for (
                    index,
                    event_id,
                )
                in enumerate(
                    batch_event_ids
                )
                if (
                    event_id
                    in allowed_event_ids
                )
            ]

        if (
            len(
                selected_positions
            )
            == 0
        ):
            continue

        position_tensor = (
            torch.tensor(
                selected_positions,
                dtype=torch.long,
            )
        )

        x = (
            batch[
                "x"
            ][
                position_tensor
            ]
            .to(
                device=device,
                dtype=next(
                    model
                    .encoder
                    .parameters()
                ).dtype,
                non_blocking=True,
            )
        )

        selected_event_ids = [
            batch_event_ids[
                position
            ]
            for position
            in selected_positions
        ]

        # ----------------------------------------------------
        # 중요:
        #
        # r1 = h - q1 아님
        #
        # 교수님 설계대로
        # event representation은 h(a)를 평균
        # ----------------------------------------------------

        h = model.encode(
            x
        )

        h_cpu = (
            h
            .detach()
            .float()
            .cpu()
        )

        for (
            event_id,
            vector,
        ) in zip(
            selected_event_ids,
            h_cpu,
        ):

            if (
                event_id
                not in event_sums
            ):
                event_sums[
                    event_id
                ] = (
                    vector.clone()
                )

                event_counts[
                    event_id
                ] = 1

            else:
                event_sums[
                    event_id
                ].add_(
                    vector
                )

                event_counts[
                    event_id
                ] += 1

        if (
            (
                batch_index
                + 1
            )
            % 20
            == 0
        ):
            processed = min(
                (
                    batch_index
                    + 1
                )
                * batch_size,
                len(dataset),
            )

            print(
                f"  {processed}/"
                f"{len(dataset)}"
            )

    if (
        len(
            event_sums
        )
        == 0
    ):
        return pd.DataFrame(
            columns=[
                "event_id",
                "event_c2",
                "num_articles",
                "source",
            ]
        )

    event_ids = sorted(
        event_sums.keys()
    )

    mapping_rows = []

    print(
        f"Events represented: "
        f"{len(event_ids)}"
    )

    print(
        "Searching nearest code "
        "in frozen Q2 codebook..."
    )

    # --------------------------------------------------------
    # event representation을 batch 단위로 Q2 nearest search
    # --------------------------------------------------------

    for start in range(
        0,
        len(event_ids),
        batch_size,
    ):

        batch_event_ids = (
            event_ids[
                start:
                start
                + batch_size
            ]
        )

        z_events = torch.stack(
            [
                (
                    event_sums[
                        event_id
                    ]
                    / float(
                        event_counts[
                            event_id
                        ]
                    )
                )
                for event_id
                in batch_event_ids
            ],
            dim=0,
        ).to(
            device=device,
            dtype=next(
                model
                .encoder
                .parameters()
            ).dtype,
        )

        # ----------------------------------------------------
        # model.eval() 상태:
        # quantizer_2는 deterministic nearest search
        # ----------------------------------------------------

        q2_out = (
            model.quantizer_2(
                x=z_events,
                temperature=1.0,
            )
        )

        c2_ids = (
            q2_out
            .ids
            .detach()
            .cpu()
            .tolist()
        )

        for (
            event_id,
            c2,
        ) in zip(
            batch_event_ids,
            c2_ids,
        ):

            mapping_rows.append({
                "event_id": int(
                    event_id
                ),

                "event_c2": int(
                    c2
                ),

                "num_articles": int(
                    event_counts[
                        event_id
                    ]
                ),

                "source": source,
            })

    result = pd.DataFrame(
        mapping_rows
    )

    if (
        result[
            "event_id"
        ]
        .duplicated()
        .any()
    ):
        raise RuntimeError(
            "Event C2 mapping contains "
            "duplicated event_id."
        )

    return result


# ============================================================
# Train EventCode
#
# Train event의 공식 EventCode:
#
# same event h(a)
#       ↓
# mean
#       ↓
# z_train(E)
#       ↓
# frozen Q2 nearest
#       ↓
# event_c2
# ============================================================

def build_train_event_c2_mapping(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
) -> pd.DataFrame:

    train_dataset = (
        SemanticIdDataset(
            master_path=(
                master_path
            ),
            embeddings_path=(
                embeddings_path
            ),
        )
    )

    mapping = (
        compute_event_c2_from_mean_h(
            model=model,
            dataset=train_dataset,
            device=device,
            allowed_event_ids=None,
            batch_size=batch_size,
            num_workers=num_workers,
            source=(
                "train_mean_h_frozen_q2"
            ),
        )
    )

    print(
        "\nTrain EventCode mapping "
        "complete."
    )

    print(
        "Train events: "
        f"{len(mapping)}"
    )

    return mapping


# ============================================================
# Validation new event EventCode
#
# Train event -> 기존 Train EventCode 상속
#
# Validation-only new event:
#       h(a)
#        ↓
# same event mean
#        ↓
# z_val(E)
#        ↓
# frozen Q2 nearest
#        ↓
# event C2
# ============================================================

def build_validation_event_c2_mapping(
    model: RqVae,
    validation_dataset: SemanticIdDataset,
    train_event_mapping: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
) -> pd.DataFrame:

    train_event_to_c2 = {
        int(row.event_id):
        int(row.event_c2)

        for row
        in (
            train_event_mapping
            .itertuples(
                index=False
            )
        )
    }

    train_event_ids = set(
        train_event_to_c2.keys()
    )

    validation_event_ids = {
        int(event_id)

        for event_id
        in (
            validation_dataset
            .master[
                "event_id"
            ]
            .tolist()
        )
    }

    # --------------------------------------------------------
    # Validation에서 처음 등장한 event
    # --------------------------------------------------------

    new_event_ids = (
        validation_event_ids
        - train_event_ids
    )

    print(
        "\nValidation event analysis"
    )

    print(
        "Validation total events : "
        f"{len(validation_event_ids)}"
    )

    print(
        "Existing Train events   : "
        f"{len(validation_event_ids & train_event_ids)}"
    )

    print(
        "Validation-only events  : "
        f"{len(new_event_ids)}"
    )

    # --------------------------------------------------------
    # 새 event만 mean(h) → frozen Q2
    # --------------------------------------------------------

    new_event_mapping = (
        compute_event_c2_from_mean_h(
            model=model,
            dataset=(
                validation_dataset
            ),
            device=device,
            allowed_event_ids=(
                new_event_ids
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            source=(
                "validation_new_"
                "event_mean_h_frozen_q2"
            ),
        )
    )

    new_event_to_c2 = {
        int(row.event_id):
        int(row.event_c2)

        for row
        in (
            new_event_mapping
            .itertuples(
                index=False
            )
        )
    }

    validation_event_counts = (
        validation_dataset
        .master
        .groupby(
            "event_id"
        )
        .size()
        .to_dict()
    )

    mapping_rows = []

    # --------------------------------------------------------
    # Validation의 모든 event에 최종 C2 지정
    # --------------------------------------------------------

    for event_id in sorted(
        validation_event_ids
    ):

        if (
            event_id
            in train_event_to_c2
        ):

            event_c2 = (
                train_event_to_c2[
                    event_id
                ]
            )

            source = (
                "inherited_train_event"
            )

        else:

            if (
                event_id
                not in new_event_to_c2
            ):
                raise RuntimeError(
                    "New validation event "
                    "does not have assigned C2. "
                    f"event_id={event_id}"
                )

            event_c2 = (
                new_event_to_c2[
                    event_id
                ]
            )

            source = (
                "new_validation_event"
            )

        mapping_rows.append({
            "event_id": int(
                event_id
            ),

            "event_c2": int(
                event_c2
            ),

            "num_articles": int(
                validation_event_counts[
                    event_id
                ]
            ),

            "source": source,
        })

    result = pd.DataFrame(
        mapping_rows
    )

    return result


# ============================================================
# Validation article SID
# ============================================================

@torch.inference_mode()
def generate_validation_semantic_ids(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    train_result: pd.DataFrame,
    train_event_mapping: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
):
    """
    Validation 처리:

    [이미 Train에서 등장한 article]
        -> Train SID 그대로 재사용

    [Validation에서 새롭게 등장한 article]

        event가 Train에 존재:
            -> 기존 Train EventCode(C2) 상속

        event가 Train에 존재하지 않음:
            -> same event validation articles의
               h(a) 평균
            -> z(E)
            -> frozen Q2 nearest
            -> event-level C2 결정

    이후 article마다:
        c1 = category ID
        q1 = Q1[c1]
        r1 = h - q1

        q2 = Q2[event C2]
        r2 = r1 - q2

        Q3 nearest
        -> c3
    """

    dataset = SemanticIdDataset(
        master_path=master_path,
        embeddings_path=embeddings_path,
    )

    validate_dataset_against_model(
        dataset=dataset,
        model=model,
    )

    # --------------------------------------------------------
    # Train article SID lookup
    # --------------------------------------------------------

    train_article_lookup = {}

    for row in (
        train_result
        .itertuples(
            index=False
        )
    ):
        article_id = str(
            row.article_id
        )

        if (
            article_id
            in train_article_lookup
        ):
            raise RuntimeError(
                "Duplicated Train article_id "
                f"found: {article_id}"
            )

        train_article_lookup[
            article_id
        ] = (
            int(row.c1),
            int(row.c2),
            int(row.c3),
        )

    # --------------------------------------------------------
    # Validation event → C2 mapping
    # --------------------------------------------------------

    validation_event_mapping = (
        build_validation_event_c2_mapping(
            model=model,
            validation_dataset=dataset,
            train_event_mapping=(
                train_event_mapping
            ),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    validation_event_to_c2 = {
        int(row.event_id):
        int(row.event_c2)

        for row
        in (
            validation_event_mapping
            .itertuples(
                index=False
            )
        )
    }

    # --------------------------------------------------------
    # Validation article inference
    # --------------------------------------------------------

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    article_ids_all = []
    embedding_rows_all = []
    event_ids_all = []

    c1_all = []
    c2_all = []
    c3_all = []

    reused_train_articles = 0
    inferred_validation_articles = 0

    print(
        "\nGenerating Validation "
        "Semantic IDs..."
    )

    print(
        f"Validation master articles: "
        f"{len(dataset)}"
    )

    for (
        batch_index,
        batch,
    ) in enumerate(
        dataloader
    ):

        batch_article_ids = [
            str(article_id)

            for article_id
            in batch[
                "article_id"
            ]
        ]

        batch_event_ids = [
            int(event_id)

            for event_id
            in (
                batch[
                    "event_id"
                ]
                .cpu()
                .tolist()
            )
        ]

        batch_size_now = len(
            batch_article_ids
        )

        batch_c1 = [
            None
        ] * batch_size_now

        batch_c2 = [
            None
        ] * batch_size_now

        batch_c3 = [
            None
        ] * batch_size_now

        inference_positions = []

        # ----------------------------------------------------
        # Train에서 이미 본 article은 SID 그대로 재사용
        # ----------------------------------------------------

        for index in range(
            batch_size_now
        ):

            article_id = (
                batch_article_ids[
                    index
                ]
            )

            if (
                article_id
                in train_article_lookup
            ):

                (
                    c1,
                    c2,
                    c3,
                ) = (
                    train_article_lookup[
                        article_id
                    ]
                )

                batch_c1[
                    index
                ] = c1

                batch_c2[
                    index
                ] = c2

                batch_c3[
                    index
                ] = c3

                reused_train_articles += 1

            else:
                inference_positions.append(
                    index
                )

        # ----------------------------------------------------
        # validation-only article만 inference
        # ----------------------------------------------------

        if (
            len(
                inference_positions
            )
            > 0
        ):

            position_tensor = (
                torch.tensor(
                    inference_positions,
                    dtype=torch.long,
                )
            )

            x = (
                batch[
                    "x"
                ][
                    position_tensor
                ]
                .to(
                    device=device,
                    dtype=next(
                        model
                        .encoder
                        .parameters()
                    ).dtype,
                    non_blocking=True,
                )
            )

            category_ids = (
                batch[
                    "category_id"
                ][
                    position_tensor
                ]
                .to(
                    device=device,
                    dtype=torch.long,
                    non_blocking=True,
                )
            )

            selected_event_ids = [
                batch_event_ids[
                    position
                ]

                for position
                in inference_positions
            ]

            # ------------------------------------------------
            # event-level로 이미 결정된 C2를 강제
            #
            # 기존 Train event:
            #     Train EventCode 상속
            #
            # 새 Validation event:
            #     mean(h) -> frozen Q2로 결정한 C2
            # ------------------------------------------------

            fixed_c2_ids = (
                torch.tensor(
                    [
                        validation_event_to_c2[
                            event_id
                        ]

                        for event_id
                        in selected_event_ids
                    ],
                    dtype=torch.long,
                    device=device,
                )
            )

            output = (
                model.get_semantic_ids(
                    x=x,
                    category_ids=(
                        category_ids
                    ),
                    fixed_c2_ids=(
                        fixed_c2_ids
                    ),
                )
            )

            semantic_ids = (
                output
                .sem_ids
                .detach()
                .cpu()
            )

            for (
                local_index,
                original_position,
            ) in enumerate(
                inference_positions
            ):

                batch_c1[
                    original_position
                ] = int(
                    semantic_ids[
                        local_index,
                        0,
                    ].item()
                )

                batch_c2[
                    original_position
                ] = int(
                    semantic_ids[
                        local_index,
                        1,
                    ].item()
                )

                batch_c3[
                    original_position
                ] = int(
                    semantic_ids[
                        local_index,
                        2,
                    ].item()
                )

            inferred_validation_articles += (
                len(
                    inference_positions
                )
            )

        # ----------------------------------------------------
        # batch 결과 확인
        # ----------------------------------------------------

        if (
            any(
                value is None
                for value
                in (
                    batch_c1
                    + batch_c2
                    + batch_c3
                )
            )
        ):
            raise RuntimeError(
                "Some Validation article "
                "did not receive SID."
            )

        article_ids_all.extend(
            batch_article_ids
        )

        embedding_rows_all.extend(
            batch[
                "embedding_row"
            ]
            .cpu()
            .tolist()
        )

        event_ids_all.extend(
            batch_event_ids
        )

        c1_all.extend(
            batch_c1
        )

        c2_all.extend(
            batch_c2
        )

        c3_all.extend(
            batch_c3
        )

        if (
            (
                batch_index
                + 1
            )
            % 20
            == 0
        ):
            processed = min(
                (
                    batch_index
                    + 1
                )
                * batch_size,
                len(dataset),
            )

            print(
                f"  {processed}/"
                f"{len(dataset)}"
            )

    result = pd.DataFrame({
        "article_id": (
            article_ids_all
        ),
        "embedding_row": (
            embedding_rows_all
        ),
        "event_id": (
            event_ids_all
        ),
        "split": "validation",
        "c1": c1_all,
        "c2": c2_all,
        "c3": c3_all,
    })

    if (
        len(result)
        != len(dataset)
    ):
        raise RuntimeError(
            "Number of generated "
            "Validation Semantic IDs "
            "does not match "
            "number of articles."
        )

    # --------------------------------------------------------
    # validation-only inferred article에 대해서
    # same event C2 일관성 확인
    #
    # Train article 재사용 row는 기존 Train article SID의
    # article-level C2를 그대로 쓰므로 여기서 제외
    # --------------------------------------------------------

    validation_only_result = (
        result[
            ~result[
                "article_id"
            ].isin(
                train_article_lookup.keys()
            )
        ]
        .copy()
    )

    if (
        len(
            validation_only_result
        )
        > 0
    ):

        event_c2_counts = (
            validation_only_result
            .groupby(
                "event_id"
            )[
                "c2"
            ]
            .nunique()
        )

        inconsistent_events = (
            event_c2_counts[
                event_c2_counts
                > 1
            ]
        )

        if (
            len(
                inconsistent_events
            )
            > 0
        ):
            raise RuntimeError(
                "Validation event-level "
                "C2 assignment failed. "
                f"{len(inconsistent_events)} "
                "events received "
                "multiple C2 codes."
            )

    print(
        "\nValidation SID generation "
        "complete."
    )

    print(
        "Reused Train articles     : "
        f"{reused_train_articles}"
    )

    print(
        "New articles inferred     : "
        f"{inferred_validation_articles}"
    )

    inherited_event_count = int(
        (
            validation_event_mapping[
                "source"
            ]
            == "inherited_train_event"
        ).sum()
    )

    new_event_count = int(
        (
            validation_event_mapping[
                "source"
            ]
            == "new_validation_event"
        ).sum()
    )

    print(
        "Existing Train events     : "
        f"{inherited_event_count}"
    )

    print(
        "Validation-only new events: "
        f"{new_event_count}"
    )

    return (
        result,
        validation_event_mapping,
    )


# ============================================================
# C4
#
# 같은 (c1, c2, c3)를 가지는 기사 구분
#
# Train + Validation 전체 unique article에 대해
# 마지막에 부여한다.
# ============================================================

def assign_disambiguation_c4(
    train_result: pd.DataFrame,
    validation_result: pd.DataFrame,
):
    """
    최종 unique article 전체에 대해
    동일한 (c1,c2,c3)를 가진 기사들을

        c4 = 0,1,2,...

    로 구분한다.

    Validation master에 Train article이 다시 등장했다면
    같은 article_id는 하나의 article로 취급하고,
    동일한 c4를 Train/Validation 결과에 재사용한다.
    """

    combined = pd.concat(
        [
            train_result,
            validation_result,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # 같은 article_id가 Train/Validation에 중복 등장한다면
    # 기존 SID가 정말 동일한지 검사
    # --------------------------------------------------------

    duplicated_article_ids = (
        combined[
            combined[
                "article_id"
            ].duplicated(
                keep=False
            )
        ][
            "article_id"
        ]
        .unique()
    )

    for article_id in (
        duplicated_article_ids
    ):

        article_rows = (
            combined[
                combined[
                    "article_id"
                ]
                == article_id
            ]
        )

        unique_sid = (
            article_rows[
                [
                    "c1",
                    "c2",
                    "c3",
                ]
            ]
            .drop_duplicates()
        )

        if (
            len(
                unique_sid
            )
            != 1
        ):
            raise RuntimeError(
                "Same article_id has "
                "different SID between "
                "Train and Validation. "
                f"article_id={article_id}"
            )

    # --------------------------------------------------------
    # 전체 article universe에서는 article_id당 한 행만 유지
    #
    # Train row를 먼저 concat했으므로
    # overlap이면 Train row가 유지됨
    # --------------------------------------------------------

    all_result = (
        combined
        .drop_duplicates(
            subset=[
                "article_id"
            ],
            keep="first",
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # c4 부여
    # --------------------------------------------------------

    all_result[
        "c4"
    ] = (
        all_result
        .groupby(
            [
                "c1",
                "c2",
                "c3",
            ],
            sort=False,
        )
        .cumcount()
        .astype(
            np.int64
        )
    )

    # --------------------------------------------------------
    # article_id -> c4
    # --------------------------------------------------------

    article_to_c4 = {
        str(row.article_id):
        int(row.c4)

        for row
        in (
            all_result
            .itertuples(
                index=False
            )
        )
    }

    train_result = (
        train_result.copy()
    )

    validation_result = (
        validation_result.copy()
    )

    train_result[
        "c4"
    ] = (
        train_result[
            "article_id"
        ]
        .astype(str)
        .map(
            article_to_c4
        )
        .astype(
            np.int64
        )
    )

    validation_result[
        "c4"
    ] = (
        validation_result[
            "article_id"
        ]
        .astype(str)
        .map(
            article_to_c4
        )
        .astype(
            np.int64
        )
    )

    return (
        train_result,
        validation_result,
        all_result,
    )


# ============================================================
# Statistics
# ============================================================

def print_sid_statistics(
    result: pd.DataFrame,
    split: str,
) -> None:

    unique_c123 = (
        result[
            [
                "c1",
                "c2",
                "c3",
            ]
        ]
        .drop_duplicates()
        .shape[0]
    )

    if (
        "c4"
        in result.columns
    ):
        unique_c1234 = (
            result[
                [
                    "c1",
                    "c2",
                    "c3",
                    "c4",
                ]
            ]
            .drop_duplicates()
            .shape[0]
        )

    else:
        unique_c1234 = None

    duplicated_c123_articles = int(
        result
        .duplicated(
            subset=[
                "c1",
                "c2",
                "c3",
            ],
            keep=False,
        )
        .sum()
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        f"{split.upper()} "
        "SID STATISTICS"
    )

    print(
        "=" * 70
    )

    print(
        "Articles             : "
        f"{len(result)}"
    )

    print(
        "Events               : "
        f"{result['event_id'].nunique()}"
    )

    print(
        "Unique c1            : "
        f"{result['c1'].nunique()}"
    )

    print(
        "Unique c2            : "
        f"{result['c2'].nunique()}"
    )

    print(
        "Unique c3            : "
        f"{result['c3'].nunique()}"
    )

    print(
        "Unique (c1,c2,c3)    : "
        f"{unique_c123}"
    )

    print(
        "Articles in dup c123 : "
        f"{duplicated_c123_articles}"
    )

    if (
        unique_c1234
        is not None
    ):
        print(
            "Unique final SID     : "
            f"{unique_c1234}"
        )

        print(
            "Final SID uniqueness : "
            f"{unique_c1234 / max(len(result), 1):.4f}"
        )

        print(
            "Max c4               : "
            f"{int(result['c4'].max()) if len(result) else 0}"
        )

    print(
        "=" * 70
    )


# ============================================================
# Coverage
# ============================================================

def print_coverage_summary(
    train_result: pd.DataFrame,
    validation_result: pd.DataFrame,
    all_result: pd.DataFrame,
    embeddings_path: str,
) -> None:

    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    num_embedding_rows = len(
        embeddings
    )

    referenced_rows = set(
        all_result[
            "embedding_row"
        ]
        .astype(int)
        .tolist()
    )

    all_embedding_rows = set(
        range(
            num_embedding_rows
        )
    )

    missing_rows = (
        all_embedding_rows
        - referenced_rows
    )

    validation_train_overlap = (
        set(
            train_result[
                "article_id"
            ]
            .astype(str)
        )
        & set(
            validation_result[
                "article_id"
            ]
            .astype(str)
        )
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEMANTIC ID COVERAGE"
    )

    print(
        "=" * 70
    )

    print(
        "article_embeddings.npy rows : "
        f"{num_embedding_rows}"
    )

    print(
        "Train SID rows              : "
        f"{len(train_result)}"
    )

    print(
        "Validation SID rows         : "
        f"{len(validation_result)}"
    )

    print(
        "Train/Validation overlap    : "
        f"{len(validation_train_overlap)}"
    )

    print(
        "Unique final articles       : "
        f"{len(all_result)}"
    )

    print(
        "Unique referenced rows      : "
        f"{len(referenced_rows)}"
    )

    if (
        len(
            missing_rows
        )
        == 0
    ):
        print(
            "\nAll rows in "
            "article_embeddings.npy "
            "are referenced."
        )

    else:
        print(
            f"\nWARNING: "
            f"{len(missing_rows)} "
            "embedding rows are "
            "not referenced."
        )

        print(
            "Example missing rows: "
            f"{sorted(missing_rows)[:20]}"
        )

    print(
        "=" * 70
        + "\n"
    )


# ============================================================
# Input file resolution
# ============================================================

def resolve_input_files(
    data_dir: Path,
):

    preferred_train_master = (
        data_dir
        / "article_master.parquet"
    )

    legacy_train_master = (
        data_dir
        / "train_article_master.parquet"
    )

    if (
        preferred_train_master
        .exists()
    ):
        train_master_path = (
            preferred_train_master
        )

    elif (
        legacy_train_master
        .exists()
    ):
        train_master_path = (
            legacy_train_master
        )

        print(
            "WARNING: Using legacy "
            "train_article_master.parquet"
        )

    else:
        raise FileNotFoundError(
            "Train master file not found. "
            "Expected one of:\n"
            f"  {preferred_train_master}\n"
            f"  {legacy_train_master}"
        )

    validation_master_path = (
        data_dir
        / "validation_article_master.parquet"
    )

    if not (
        validation_master_path
        .exists()
    ):
        raise FileNotFoundError(
            "Validation master file "
            "not found: "
            f"{validation_master_path}"
        )

    embeddings_path = (
        data_dir
        / "article_embeddings.npy"
    )

    if not (
        embeddings_path
        .exists()
    ):
        raise FileNotFoundError(
            "Embedding file not found: "
            f"{embeddings_path}"
        )

    return (
        train_master_path,
        validation_master_path,
        embeddings_path,
    )


# ============================================================
# Main Semantic ID generation
# ============================================================

def generate_semantic_ids(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
):

    data_dir = Path(
        data_dir
    )

    checkpoint_path = Path(
        checkpoint_path
    )

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    (
        train_master_path,
        validation_master_path,
        embeddings_path,
    ) = resolve_input_files(
        data_dir
    )

    if not (
        checkpoint_path
        .exists()
    ):
        raise FileNotFoundError(
            "Checkpoint not found: "
            f"{checkpoint_path}"
        )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEMANTIC ID GENERATION CONFIG"
    )

    print(
        "=" * 70
    )

    print(
        "Train master      : "
        f"{train_master_path}"
    )

    print(
        "Validation master : "
        f"{validation_master_path}"
    )

    print(
        "Embeddings        : "
        f"{embeddings_path}"
    )

    print(
        "Checkpoint        : "
        f"{checkpoint_path}"
    )

    print(
        "Output directory  : "
        f"{output_dir}"
    )

    print(
        "Batch size        : "
        f"{batch_size}"
    )

    print(
        "=" * 70
        + "\n"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"Device: {device}"
    )

    # ========================================================
    # 1. Load / freeze trained model
    # ========================================================

    model = load_rqvae(
        checkpoint_path=str(
            checkpoint_path
        ),
        device=device,
    )

    # ========================================================
    # 2. Train article final SID 확정
    #
    # 학습 중 계산되던 SID를
    # final frozen model 기준으로 한 번 확정하여 저장
    # ========================================================

    train_result = (
        generate_train_semantic_ids(
            model=model,
            master_path=str(
                train_master_path
            ),
            embeddings_path=str(
                embeddings_path
            ),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    # ========================================================
    # 3. Train event EventCode 확정
    #
    # z_train(E) = mean(h(a))
    #        ↓
    # frozen Q2 nearest
    #
    # Validation에서 동일 event가 다시 등장하면
    # 이 EventCode를 그대로 상속
    # ========================================================

    train_event_mapping = (
        build_train_event_c2_mapping(
            model=model,
            master_path=str(
                train_master_path
            ),
            embeddings_path=str(
                embeddings_path
            ),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    train_event_mapping_path = (
        output_dir
        / "train_event_c2_mapping.parquet"
    )

    train_event_mapping.to_parquet(
        train_event_mapping_path,
        index=False,
    )

    # ========================================================
    # 4. Validation
    #
    # Train에서 본 article:
    #      기존 SID 재사용
    #
    # validation-only article:
    #
    #   기존 Train event:
    #      기존 EventCode 상속
    #
    #   새로운 event:
    #      mean(h)
    #      -> z(E)
    #      -> frozen Q2 nearest
    #
    #   그 뒤 fixed C2로 article-level Q3 결정
    # ========================================================

    (
        validation_result,
        validation_event_mapping,
    ) = (
        generate_validation_semantic_ids(
            model=model,
            master_path=str(
                validation_master_path
            ),
            embeddings_path=str(
                embeddings_path
            ),
            train_result=(
                train_result
            ),
            train_event_mapping=(
                train_event_mapping
            ),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    validation_event_mapping_path = (
        output_dir
        / "validation_event_c2_mapping.parquet"
    )

    validation_event_mapping.to_parquet(
        validation_event_mapping_path,
        index=False,
    )

    # ========================================================
    # 5. C4
    #
    # 모든 c1,c2,c3가 확정된 뒤
    # Train + Validation 전체 unique article 기준으로 부여
    # ========================================================

    (
        train_result,
        validation_result,
        all_result,
    ) = assign_disambiguation_c4(
        train_result=(
            train_result
        ),
        validation_result=(
            validation_result
        ),
    )

    # ========================================================
    # 6. Statistics
    # ========================================================

    print_sid_statistics(
        result=train_result,
        split="train",
    )

    print_sid_statistics(
        result=validation_result,
        split="validation",
    )

    print_sid_statistics(
        result=all_result,
        split=(
            "train+validation unique"
        ),
    )

    # ========================================================
    # 7. Save
    # ========================================================

    train_output_path = (
        output_dir
        / "train_article_semantic_ids.parquet"
    )

    validation_output_path = (
        output_dir
        / "validation_article_semantic_ids.parquet"
    )

    all_output_path = (
        output_dir
        / "article_semantic_ids.parquet"
    )

    train_result.to_parquet(
        train_output_path,
        index=False,
    )

    validation_result.to_parquet(
        validation_output_path,
        index=False,
    )

    all_result.to_parquet(
        all_output_path,
        index=False,
    )

    # ========================================================
    # 8. Coverage
    # ========================================================

    print_coverage_summary(
        train_result=(
            train_result
        ),
        validation_result=(
            validation_result
        ),
        all_result=(
            all_result
        ),
        embeddings_path=str(
            embeddings_path
        ),
    )

    # ========================================================
    # Final summary
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "SEMANTIC ID GENERATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        "\nTrain Event C2 mapping:"
        f"\n  {train_event_mapping_path}"
    )

    print(
        "\nValidation Event C2 mapping:"
        f"\n  {validation_event_mapping_path}"
    )

    print(
        "\nTrain SID file:"
        f"\n  {train_output_path}"
    )

    print(
        "\nValidation SID file:"
        f"\n  {validation_output_path}"
    )

    print(
        "\nCombined final SID file:"
        f"\n  {all_output_path}"
    )

    print(
        "\nTrain SID rows       : "
        f"{len(train_result)}"
    )

    print(
        "Validation SID rows  : "
        f"{len(validation_result)}"
    )

    print(
        "Unique final articles: "
        f"{len(all_result)}"
    )

    print(
        "\nPreview:"
    )

    print(
        all_result.head(
            10
        )
    )

    print(
        "=" * 70
    )

    return (
        train_result,
        validation_result,
        all_result,
        train_event_mapping,
        validation_event_mapping,
    )


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":

    parser = (
        argparse.ArgumentParser()
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Directory containing "
            "article_master.parquet, "
            "validation_article_master.parquet, "
            "and article_embeddings.npy"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help=(
            "Path to trained "
            "checkpoint_final.pt"
        ),
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=(
            "out/semantic_ids"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = (
        parser.parse_args()
    )

    generate_semantic_ids(
        data_dir=(
            args.data_dir
        ),
        checkpoint_path=(
            args.checkpoint
        ),
        output_dir=(
            args.output_dir
        ),
        batch_size=(
            args.batch_size
        ),
        num_workers=(
            args.num_workers
        ),
    )