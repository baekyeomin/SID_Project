# validation 단계에서 eventid 사용없이 각 기사를 독립적으로 train 기사에 매칭하도록 변경한 신버전
#
# Train 사건 구성
# - 이전: event_id로 그룹화
# - 변경: 유지
#
# Train 사건 대표
# - 이전: z(E) = mean(h(a))
# - 변경: 유지
#
# Validation-only 매칭 단위
# - 이전: 사건 단위
# - 변경: 기사 단위
#
# 매칭에 사용하는 임베딩
# - 이전: 사건의 첫 기사 h(a1)
# - 변경: 각 기사 h(a)
#
# Train 사건 선택
# - 이전: 코사인 유사도 최대
# - 변경: 유지
#
# 72시간 조건
# - 이전: 사건의 첫 기사 발행 시각 기준
# - 변경: 각 기사 발행 시각 기준
#**기사에 발행 시각 정보가 없거나 만족하는 시간 조건이 없다면 코사인 유사도로만 판별
#
# C2 부여
# - 이전: 동일 Validation 사건의 모든 기사에 동일 C2
# - 변경: 기사별로 매칭된 Train 사건의 C2 상속
#
# 기존 Train 기사
# - 이전: 기존 SID 재사용
# - 변경: 유지


import argparse
import importlib
from pathlib import Path
from typing import Optional, Set

import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from modules.rqvae import RqVae


def ensure_torch_serialization_compatibility() -> None:
    if not hasattr(torch, "Tensor"):
        tensor_class = None

        try:
            tensor_module = importlib.import_module("torch._tensor")
            tensor_class = getattr(tensor_module, "Tensor", None, )
        except Exception:
            tensor_class = None

        if tensor_class is None:
            try:
                tensor_class = type(torch.empty(0))
            except Exception as exc:
                raise RuntimeError("torch.Tensor attribute를 " "복구하지 못했습니다.") from exc

        setattr(torch, "Tensor", tensor_class, )

    if not hasattr(torch, "_utils", ):
        try:
            torch_utils = (importlib.import_module("torch._utils"))

            setattr(torch, "_utils", torch_utils, )

        except Exception as exc:
            raise RuntimeError("torch._utils attribute를 " "복구하지 못했습니다.") from exc


def safe_torch_load(checkpoint_path, map_location, ):
    ensure_torch_serialization_compatibility()

    return torch.load(checkpoint_path, map_location=map_location, weights_only=False, )


class SemanticIdDataset(Dataset):

    REQUIRED_COLUMNS = { "article_id", "embedding_row", "model_category_id", "event_id", }

    def __init__(self, master_path: str, embeddings_path: str, ) -> None:

        super().__init__()

        self.master_path = Path(master_path)

        self.embeddings_path = Path(embeddings_path)

        self.master = pd.read_parquet(self.master_path).reset_index(drop=True)

        missing_columns = (self.REQUIRED_COLUMNS - set(self.master.columns))

        if missing_columns:
            raise ValueError(f"{self.master_path} " "is missing required columns: " f"{sorted(missing_columns)}")

        if ("event_id" in self.master.columns and self.master["event_id"].isna().any()):
            missing_count = int(self.master["event_id"].isna().sum())

            raise ValueError(f"{self.master_path} contains " f"{missing_count} missing " "event_id values.")

        self.embeddings = np.load(self.embeddings_path, mmap_mode="r", )

        if (self.embeddings.ndim != 2):
            raise ValueError(
                "article_embeddings.npy "
                "must have shape "
                "[num_articles, embedding_dim]. "
                f"Got {self.embeddings.shape}."
            )

        rows = (self.master["embedding_row"].to_numpy())

        if len(rows) > 0:
            min_row = int(rows.min())

            max_row = int(rows.max())

            if min_row < 0:
                raise ValueError("embedding_row contains " "negative values.")

            if (max_row >= len(self.embeddings)):
                raise ValueError(
                    "embedding_row points outside "
                    "article_embeddings.npy. "
                    f"Maximum embedding_row="
                    f"{max_row}, "
                    f"but embeddings contain "
                    f"{len(self.embeddings)} rows."
                )

    def __len__(self, ) -> int:

        return len(self.master)

    def __getitem__(self, index: int, ):

        row = (self.master.iloc[index])

        embedding_row = int(row["embedding_row"])

        category_id = int(row["model_category_id"])

        event_id = int(row["event_id"])

        embedding = np.asarray(self.embeddings[embedding_row], dtype=np.float32, ).copy()

        x = torch.from_numpy(embedding)

        return {
            "x": x,

            "category_id": torch.tensor(category_id, dtype=torch.long, ),

            "event_id": torch.tensor(event_id, dtype=torch.long, ),

            "article_id": str(row["article_id"]),

            "embedding_row": torch.tensor(embedding_row, dtype=torch.long, ),
        }


def load_rqvae(checkpoint_path: str, device: torch.device, ) -> RqVae:

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError("RQ-VAE checkpoint " f"not found: {checkpoint_path}")

    print("\n" + "=" * 70)

    print("LOADING TRAINED RQ-VAE")

    print("=" * 70)

    checkpoint = safe_torch_load(checkpoint_path=(checkpoint_path), map_location=device, )

    if ("model_config" not in checkpoint):
        raise KeyError("Checkpoint does not contain " "'model_config'.")

    if ("model" not in checkpoint):
        raise KeyError("Checkpoint does not contain " "'model'.")

    model_config = (checkpoint["model_config"])

    print("Model config:")

    for (key, value, ) in (model_config.items()):
        print(f"  {key}: {value}")

    model = RqVae(**model_config)

    model.load_state_dict(checkpoint["model"])

    model = model.to(device)


    model.eval()

    for parameter in (model.parameters()):
        parameter.requires_grad_(False)

    print(f"\nLoaded checkpoint: " f"{checkpoint_path}")

    if ("epoch" in checkpoint):
        print("Training epoch: " f"{checkpoint['epoch'] + 1}")

    if ("global_step" in checkpoint):
        print("Training global step: " f"{checkpoint['global_step']}")

    elif ("iter" in checkpoint):
        print("Training iteration: " f"{checkpoint['iter']}")

    print("RQ-VAE frozen.")

    print("=" * 70 + "\n")

    return model


def validate_dataset_against_model(dataset: SemanticIdDataset, model: RqVae, ) -> None:

    embedding_dim = (dataset.embeddings.shape[1])

    if (embedding_dim != model.input_dim):
        raise ValueError(
            "Article embedding dimension "
            "does not match RQ-VAE "
            "input_dim. "
            "article_embeddings.npy="
            f"{embedding_dim}, "
            "RQ-VAE input_dim="
            f"{model.input_dim}"
        )

    if (len(dataset.master) > 0):
        category_values = (dataset.master["model_category_id"].astype(int))

        min_category = int(category_values.min())

        max_category = int(category_values.max())

        if (min_category < 0):
            raise ValueError(
                "model_category_id "
                "must be >= 0. "
                "Found minimum="
                f"{min_category}."
            )

        if (max_category >= model.num_categories):
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


@torch.inference_mode()
def generate_train_semantic_ids(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
) -> pd.DataFrame:


    dataset = SemanticIdDataset(master_path=master_path, embeddings_path=embeddings_path, )

    validate_dataset_against_model(dataset=dataset, model=model, )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=(device.type == "cuda"), )

    article_ids_all = []
    embedding_rows_all = []
    event_ids_all = []

    c1_all = []
    c2_all = []
    c3_all = []

    print("\nGenerating final " "Train Semantic IDs...")

    print(f"Articles: {len(dataset)}")

    for (batch_index, batch, ) in enumerate(dataloader):

        x = (batch["x"].to(device=device, dtype=next(model.encoder.parameters()).dtype, non_blocking=True, ))

        category_ids = (batch["category_id"].to(device=device, dtype=torch.long, non_blocking=True, ))


        output = (model.get_semantic_ids(x=x, category_ids=(category_ids), ))

        semantic_ids = (output.sem_ids.detach().cpu())

        article_ids_all.extend(batch["article_id"])

        embedding_rows_all.extend(batch["embedding_row"].cpu().tolist())

        event_ids_all.extend(batch["event_id"].cpu().tolist())

        c1_all.extend(semantic_ids[:, 0, ].tolist())

        c2_all.extend(semantic_ids[:, 1, ].tolist())

        c3_all.extend(semantic_ids[:, 2, ].tolist())

        if ((batch_index + 1) % 20 == 0):
            processed = min((batch_index + 1) * batch_size, len(dataset), )

            print(f"  {processed}/" f"{len(dataset)}")

    result = pd.DataFrame({
        "article_id": (article_ids_all),
        "embedding_row": (embedding_rows_all),
        "event_id": (event_ids_all),
        "split": "train",
        "c1": c1_all,
        "c2": c2_all,
        "c3": c3_all,
    })

    if (len(result) != len(dataset)):
        raise RuntimeError(
            "Number of generated "
            "Train Semantic IDs does "
            "not match number of articles. "
            f"articles={len(dataset)}, "
            f"semantic_ids={len(result)}"
        )

    if (result["article_id"].duplicated().any()):
        raise RuntimeError("Train result contains " "duplicated article_id.")

    print(f"Train: " f"{len(result)} final SID " "rows generated.")

    return result


@torch.inference_mode()
def compute_event_c2_from_mean_h(
    model: RqVae,
    dataset: SemanticIdDataset,
    device: torch.device,
    allowed_event_ids: Optional[Set[int]] = None,
    batch_size: int = 512,
    num_workers: int = 0,
    source: str = "mean_h_frozen_q2",
    return_representations: bool = False,
) -> pd.DataFrame:


    validate_dataset_against_model(dataset=dataset, model=model, )

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=(device.type == "cuda"), )

    event_sums = {}
    event_counts = {}

    print("\nBuilding event representations " "z(E) = mean(h(a))...")

    for (batch_index, batch, ) in enumerate(dataloader):

        batch_event_ids = [int(event_id) for event_id in (batch["event_id"].cpu().tolist())]


        if (allowed_event_ids is None):
            selected_positions = list(range(len(batch_event_ids)))

        else:
            selected_positions = [index for (index, event_id, ) in enumerate(batch_event_ids) if (event_id in allowed_event_ids)]

        if (len(selected_positions) == 0):
            continue

        position_tensor = (torch.tensor(selected_positions, dtype=torch.long, ))

        x = (batch["x"][position_tensor].to(device=device, dtype=next(model.encoder.parameters()).dtype, non_blocking=True, ))

        selected_event_ids = [batch_event_ids[position] for position in selected_positions]


        h = model.encode(x)

        h_cpu = (h.detach().float().cpu())

        for (event_id, vector, ) in zip(selected_event_ids, h_cpu, ):

            if (event_id not in event_sums):
                event_sums[event_id] = (vector.clone())

                event_counts[event_id] = 1

            else:
                event_sums[event_id].add_(vector)

                event_counts[event_id] += 1

        if ((batch_index + 1) % 20 == 0):
            processed = min((batch_index + 1) * batch_size, len(dataset), )

            print(f"  {processed}/" f"{len(dataset)}")

    if (len(event_sums) == 0):
        empty = pd.DataFrame(columns=["event_id", "event_c2", "num_articles", "source"])
        return (empty, {}) if return_representations else empty

    event_ids = sorted(event_sums.keys())

    mapping_rows = []
    representations = {}

    print(f"Events represented: " f"{len(event_ids)}")

    print("Searching nearest code " "in frozen Q2 codebook...")


    for start in range(0, len(event_ids), batch_size, ):

        batch_event_ids = (event_ids[start: start + batch_size])

        z_events = torch.stack([(event_sums[event_id] / float(event_counts[event_id])) for event_id in batch_event_ids], dim=0, ).to(
            device=device,
            dtype=next(model.encoder.parameters()).dtype,
        )


        if return_representations:
            representations.update({ int(event_id): vector.detach().float().cpu().numpy().copy() for event_id, vector in zip(batch_event_ids, z_events) })


        q2_out = (model.quantizer_2(x=z_events, temperature=1.0, ))

        c2_ids = (q2_out.ids.detach().cpu().tolist())

        for (event_id, c2, ) in zip(batch_event_ids, c2_ids, ):

            mapping_rows.append({ "event_id": int(event_id),  "event_c2": int(c2),  "num_articles": int(event_counts[event_id]),  "source": source, })

    result = pd.DataFrame(mapping_rows)

    if (result["event_id"].duplicated().any()):
        raise RuntimeError("Event C2 mapping contains " "duplicated event_id.")

    # 변경: Train 사건 대표 벡터를 Validation 매칭에 재사용
    return (result, representations) if return_representations else result


def build_train_event_c2_mapping(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
    return_representations: bool = False,
) -> pd.DataFrame:

    train_dataset = (SemanticIdDataset(master_path=(master_path), embeddings_path=(embeddings_path), ))

    mapping = (
        compute_event_c2_from_mean_h(
            model=model,
            dataset=train_dataset,
            device=device,
            allowed_event_ids=None,
            batch_size=batch_size,
            num_workers=num_workers,
            source=("train_mean_h_frozen_q2"),
            return_representations=return_representations,
        )
    )

    print("\nTrain EventCode mapping " "complete.")

    print("Train events: " f"{len(mapping[0]) if return_representations else len(mapping)}")

    return mapping


# 변경: Validation 신규 사건의 첫 기사 및 Train 사건 시각 확인
TIME_COLUMN_CANDIDATES = (
    "published_time", "published_at", "publication_time", "publication_date",
    "published_date", "pub_time", "publish_time", "publication_timestamp",
    "published", "first_published_at", "article_published_time",
)


def get_publication_times(master: pd.DataFrame) -> tuple[pd.Series, Optional[str]]:

    for column in TIME_COLUMN_CANDIDATES:
        if column in master.columns:
            values = pd.to_datetime(master[column], errors="coerce", utc=True)
            if values.notna().any():
                return values, column
    return pd.Series(pd.NaT, index=master.index, dtype="datetime64[ns, UTC]"), None


@torch.inference_mode()
def encode_selected_articles(model: RqVae, dataset: SemanticIdDataset, positions: list[int], device: torch.device, batch_size: int = 512, ) -> np.ndarray:

    vectors = []
    dtype = next(model.encoder.parameters()).dtype
    for start in range(0, len(positions), batch_size):
        chunk = dataset.master.iloc[positions[start:start + batch_size]]
        rows = chunk["embedding_row"].astype(int).to_numpy()
        x = torch.from_numpy(np.asarray(dataset.embeddings[rows], dtype=np.float32).copy())
        h = model.encode(x.to(device=device, dtype=dtype))
        vectors.append(h.detach().float().cpu().numpy())
    return np.concatenate(vectors) if vectors else np.empty((0, model.embed_dim), dtype=np.float32)


# Validation-only 기사를 각각 독립적으로 Train 사건에 매칭
def match_validation_articles_to_train(model: RqVae, validation_dataset: SemanticIdDataset, positions: list[int], train_event_representations: dict[int, np.ndarray], train_master: pd.DataFrame, device: torch.device, batch_size: int = 512, event_time_window_days: Optional[float] = 3.0) -> pd.DataFrame:
    if not positions:
        return pd.DataFrame(columns=["article_id", "matched_train_event_id", "event_c2", "match_cosine", "match_used_time_window"])
    if not train_event_representations:
        raise ValueError("Train 사건 대표 벡터가 없습니다.")
    valid_master = validation_dataset.master
    valid_times, valid_time_column = get_publication_times(valid_master)
    train_times, train_time_column = get_publication_times(train_master)
    if event_time_window_days is not None and (valid_time_column is None or train_time_column is None):
        warnings.warn("발행 시각 컬럼 부족: 시간 조건 없이 코사인 유사도로 매칭합니다.")
    train_event_ids = sorted(train_event_representations)
    train_vectors = np.stack([train_event_representations[e] for e in train_event_ids]).astype(np.float32)
    train_vectors /= np.maximum(np.linalg.norm(train_vectors, axis=1, keepdims=True), 1e-12)
    train_last_times = {}
    if train_time_column is not None:
        train_last_times = pd.DataFrame({"event_id": train_master["event_id"].to_numpy(), "timestamp": train_times.to_numpy()}).groupby("event_id")["timestamp"].max().to_dict()
    article_vectors = encode_selected_articles(model, validation_dataset, positions, device, batch_size)
    article_vectors /= np.maximum(np.linalg.norm(article_vectors, axis=1, keepdims=True), 1e-12)
    rows = []
    for i, position in enumerate(positions):
        similarity = train_vectors @ article_vectors[i]
        first_time = valid_times.iloc[position]
        eligible = np.ones(len(train_event_ids), dtype=bool)
        used_window = False
        if event_time_window_days is not None and pd.notna(first_time) and train_last_times:
            hours = np.array([(first_time - train_last_times[e]).total_seconds() / 3600 if pd.notna(train_last_times.get(e)) else np.inf for e in train_event_ids])
            eligible = (hours >= 0) & (hours <= event_time_window_days * 24)
            used_window = bool(eligible.any())
            if not used_window:
                eligible[:] = True
        best = int(np.argmax(np.where(eligible, similarity, -np.inf)))
        rows.append({"article_id": str(valid_master.iloc[position]["article_id"]), "matched_train_event_id": int(train_event_ids[best]), "match_cosine": float(similarity[best]), "match_used_time_window": used_window})
    return pd.DataFrame(rows)


@torch.inference_mode()
def generate_validation_semantic_ids(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    train_result: pd.DataFrame,
    train_event_mapping: pd.DataFrame,
    train_event_representations: dict[int, np.ndarray],
    train_master: pd.DataFrame,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
    event_time_window_days: Optional[float] = 3.0,
):


    dataset = SemanticIdDataset(master_path=master_path, embeddings_path=embeddings_path, )

    validate_dataset_against_model(dataset=dataset, model=model, )


    train_article_lookup = {}

    for row in (train_result.itertuples(index=False)):
        article_id = str(row.article_id)

        if (article_id in train_article_lookup):
            raise RuntimeError("Duplicated Train article_id " f"found: {article_id}")

        train_article_lookup[article_id] = (int(row.c1), int(row.c2), int(row.c3), )


    inference_master_positions = [i for i, article_id in enumerate(dataset.master["article_id"].astype(str)) if article_id not in train_article_lookup]
    validation_article_mapping = match_validation_articles_to_train(model, dataset, inference_master_positions, train_event_representations, train_master, device, batch_size, event_time_window_days)
    train_event_to_c2 = dict(zip(train_event_mapping["event_id"].astype(int), train_event_mapping["event_c2"].astype(int)))
    validation_article_mapping["event_c2"] = validation_article_mapping["matched_train_event_id"].map(train_event_to_c2).astype(int) if len(validation_article_mapping) else pd.Series(dtype=int)
    validation_article_to_c2 = dict(zip(validation_article_mapping["article_id"], validation_article_mapping["event_c2"]))

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=num_workers, pin_memory=(device.type == "cuda"), )

    article_ids_all = []
    embedding_rows_all = []
    event_ids_all = []

    c1_all = []
    c2_all = []
    c3_all = []

    reused_train_articles = 0
    inferred_validation_articles = 0

    print("\nGenerating Validation " "Semantic IDs...")

    print(f"Validation master articles: " f"{len(dataset)}")

    for (batch_index, batch, ) in enumerate(dataloader):

        batch_article_ids = [str(article_id)  for article_id in batch["article_id"]]

        batch_event_ids = [int(event_id)  for event_id in (batch["event_id"].cpu().tolist())]

        batch_size_now = len(batch_article_ids)

        batch_c1 = [None] * batch_size_now

        batch_c2 = [None] * batch_size_now

        batch_c3 = [None] * batch_size_now

        inference_positions = []


        for index in range(batch_size_now):

            article_id = (batch_article_ids[index])

            if (article_id in train_article_lookup):

                (c1, c2, c3, ) = (train_article_lookup[article_id])

                batch_c1[index] = c1

                batch_c2[index] = c2

                batch_c3[index] = c3

                reused_train_articles += 1

            else:
                inference_positions.append(index)


        if (len(inference_positions) > 0):

            position_tensor = (torch.tensor(inference_positions, dtype=torch.long, ))

            x = (batch["x"][position_tensor].to(device=device, dtype=next(model.encoder.parameters()).dtype, non_blocking=True, ))

            category_ids = (batch["category_id"][position_tensor].to(device=device, dtype=torch.long, non_blocking=True, ))


            fixed_c2_ids = (torch.tensor([validation_article_to_c2[batch_article_ids[position]] for position in inference_positions], dtype=torch.long, device=device, ))

            output = (model.get_semantic_ids(x=x, category_ids=(category_ids), fixed_c2_ids=(fixed_c2_ids), ))

            semantic_ids = (output.sem_ids.detach().cpu())

            for (local_index, original_position, ) in enumerate(inference_positions):

                batch_c1[original_position] = int(semantic_ids[local_index, 0, ].item())

                batch_c2[original_position] = int(semantic_ids[local_index, 1, ].item())

                batch_c3[original_position] = int(semantic_ids[local_index, 2, ].item())

            inferred_validation_articles += (len(inference_positions))


        if (any(value is None for value in (batch_c1 + batch_c2 + batch_c3))):
            raise RuntimeError("Some Validation article " "did not receive SID.")

        article_ids_all.extend(batch_article_ids)

        embedding_rows_all.extend(batch["embedding_row"].cpu().tolist())

        event_ids_all.extend(batch_event_ids)

        c1_all.extend(batch_c1)

        c2_all.extend(batch_c2)

        c3_all.extend(batch_c3)

        if ((batch_index + 1) % 20 == 0):
            processed = min((batch_index + 1) * batch_size, len(dataset), )

            print(f"  {processed}/" f"{len(dataset)}")

    result = pd.DataFrame({
        "article_id": (article_ids_all),
        "embedding_row": (embedding_rows_all),
        "event_id": (event_ids_all),
        "split": "validation",
        "c1": c1_all,
        "c2": c2_all,
        "c3": c3_all,
    })

    if (len(result) != len(dataset)):
        raise RuntimeError("Number of generated " "Validation Semantic IDs " "does not match " "number of articles.")


    print("\nValidation SID generation " "complete.")

    print("Reused Train articles     : " f"{reused_train_articles}")

    print("New articles inferred     : " f"{inferred_validation_articles}")

    print("Validation-only articles matched:", len(validation_article_mapping))

    return (result, validation_article_mapping, )


def assign_disambiguation_c4(train_result: pd.DataFrame, validation_result: pd.DataFrame, ):


    combined = pd.concat([train_result, validation_result, ], ignore_index=True, )


    duplicated_article_ids = (combined[combined["article_id"].duplicated(keep=False)]["article_id"].unique())

    for article_id in (duplicated_article_ids):

        article_rows = (combined[combined["article_id"] == article_id])

        unique_sid = (article_rows[["c1", "c2", "c3", ]].drop_duplicates())

        if (len(unique_sid) != 1):
            raise RuntimeError(
                "Same article_id has "
                "different SID between "
                "Train and Validation. "
                f"article_id={article_id}"
            )


    all_result = (combined.drop_duplicates(subset=["article_id"], keep="first", ).reset_index(drop=True))


    all_result["c4"] = (all_result.groupby(["c1", "c2", "c3", ], sort=False, ).cumcount().astype(np.int64))


    article_to_c4 = { str(row.article_id): int(row.c4)  for row in (all_result.itertuples(index=False)) }

    train_result = (train_result.copy())

    validation_result = (validation_result.copy())

    train_result["c4"] = (train_result["article_id"].astype(str).map(article_to_c4).astype(np.int64))

    validation_result["c4"] = (validation_result["article_id"].astype(str).map(article_to_c4).astype(np.int64))

    return (train_result, validation_result, all_result, )


def print_sid_statistics(result: pd.DataFrame, split: str, ) -> None:

    unique_c123 = (result[["c1", "c2", "c3", ]].drop_duplicates().shape[0])

    if ("c4" in result.columns):
        unique_c1234 = (result[["c1", "c2", "c3", "c4", ]].drop_duplicates().shape[0])

    else:
        unique_c1234 = None

    duplicated_c123_articles = int(result.duplicated(subset=["c1", "c2", "c3", ], keep=False, ).sum())

    print("\n" + "=" * 70)

    print(f"{split.upper()} " "SID STATISTICS")

    print("=" * 70)

    print("Articles             : " f"{len(result)}")

    print("Events               : " f"{result['event_id'].nunique()}")

    print("Unique c1            : " f"{result['c1'].nunique()}")

    print("Unique c2            : " f"{result['c2'].nunique()}")

    print("Unique c3            : " f"{result['c3'].nunique()}")

    print(
        "Unique (c1,c2,c3)    : "
        f"{unique_c123}"
    )

    print("Articles in dup c123 : " f"{duplicated_c123_articles}")

    if (unique_c1234 is not None):
        print("Unique final SID     : " f"{unique_c1234}")

        print("Final SID uniqueness : " f"{unique_c1234 / max(len(result), 1):.4f}")

        print("Max c4               : " f"{int(result['c4'].max()) if len(result) else 0}")

    print("=" * 70)


def print_coverage_summary(train_result: pd.DataFrame, validation_result: pd.DataFrame, all_result: pd.DataFrame, embeddings_path: str, ) -> None:

    embeddings = np.load(embeddings_path, mmap_mode="r", )

    num_embedding_rows = len(embeddings)

    referenced_rows = set(all_result["embedding_row"].astype(int).tolist())

    all_embedding_rows = set(range(num_embedding_rows))

    missing_rows = (all_embedding_rows - referenced_rows)

    validation_train_overlap = (set(train_result["article_id"].astype(str)) & set(validation_result["article_id"].astype(str)))

    print("\n" + "=" * 70)

    print("SEMANTIC ID COVERAGE")

    print("=" * 70)

    print("article_embeddings.npy rows : " f"{num_embedding_rows}")

    print("Train SID rows              : " f"{len(train_result)}")

    print("Validation SID rows         : " f"{len(validation_result)}")

    print("Train/Validation overlap    : " f"{len(validation_train_overlap)}")

    print("Unique final articles       : " f"{len(all_result)}")

    print("Unique referenced rows      : " f"{len(referenced_rows)}")

    if (len(missing_rows) == 0):
        print("\nAll rows in " "article_embeddings.npy " "are referenced.")

    else:
        print(f"\nWARNING: " f"{len(missing_rows)} " "embedding rows are " "not referenced.")

        print("Example missing rows: " f"{sorted(missing_rows)[:20]}")

    print("=" * 70 + "\n")


def resolve_input_files(data_dir: Path, ):

    preferred_train_master = (data_dir / "article_master.parquet")

    legacy_train_master = (data_dir / "train_article_master.parquet")

    if (preferred_train_master.exists()):
        train_master_path = (preferred_train_master)

    elif (legacy_train_master.exists()):
        train_master_path = (legacy_train_master)

        print("WARNING: Using legacy " "train_article_master.parquet")

    else:
        raise FileNotFoundError(
            "Train master file not found. "
            "Expected one of:\n"
            f"  {preferred_train_master}\n"
            f"  {legacy_train_master}"
        )

    validation_master_path = (data_dir / "validation_article_master.parquet")

    if not (validation_master_path.exists()):
        raise FileNotFoundError("Validation master file " "not found: " f"{validation_master_path}")

    embeddings_path = (data_dir / "article_embeddings.npy")

    if not (embeddings_path.exists()):
        raise FileNotFoundError("Embedding file not found: " f"{embeddings_path}")

    return (train_master_path, validation_master_path, embeddings_path, )


def generate_semantic_ids(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
    event_time_window_days: Optional[float] = 3.0,
):

    data_dir = Path(data_dir)

    checkpoint_path = Path(checkpoint_path)

    output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True, )

    (train_master_path, validation_master_path, embeddings_path, ) = resolve_input_files(data_dir)

    if not (checkpoint_path.exists()):
        raise FileNotFoundError("Checkpoint not found: " f"{checkpoint_path}")

    print("\n" + "=" * 70)

    print("SEMANTIC ID GENERATION CONFIG")

    print("=" * 70)

    print("Train master      : " f"{train_master_path}")

    print("Validation master : " f"{validation_master_path}")

    print("Embeddings        : " f"{embeddings_path}")

    print("Checkpoint        : " f"{checkpoint_path}")

    print("Output directory  : " f"{output_dir}")

    print("Batch size        : " f"{batch_size}")

    print("=" * 70 + "\n")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")


    model = load_rqvae(checkpoint_path=str(checkpoint_path), device=device, )


    train_result = (
        generate_train_semantic_ids(
            model=model,
            master_path=str(train_master_path),
            embeddings_path=str(embeddings_path),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )


    train_event_mapping, train_event_representations = (
        build_train_event_c2_mapping(
            model=model,
            master_path=str(train_master_path),
            embeddings_path=str(embeddings_path),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            return_representations=True,
        )
    )

    train_event_mapping_path = (output_dir / "train_event_c2_mapping.parquet")

    train_event_mapping.to_parquet(train_event_mapping_path, index=False, )


    (validation_result, validation_article_mapping, ) = (
        generate_validation_semantic_ids(
            model=model,
            master_path=str(validation_master_path),
            embeddings_path=str(embeddings_path),
            train_result=(train_result),
            train_event_mapping=(train_event_mapping),
            train_event_representations=train_event_representations,
            train_master=pd.read_parquet(train_master_path),
            device=device,
            batch_size=batch_size,
            num_workers=num_workers,
            event_time_window_days=event_time_window_days,
        )
    )

    validation_article_mapping_path = (output_dir / "validation_article_train_event_mapping.parquet")

    validation_article_mapping.to_parquet(validation_article_mapping_path, index=False, )


    (train_result, validation_result, all_result, ) = assign_disambiguation_c4(train_result=(train_result), validation_result=(validation_result), )


    print_sid_statistics(result=train_result, split="train", )

    print_sid_statistics(result=validation_result, split="validation", )

    print_sid_statistics(result=all_result, split=("train+validation unique"), )


    train_output_path = (output_dir / "train_article_semantic_ids.parquet")

    validation_output_path = (output_dir / "validation_article_semantic_ids.parquet")

    all_output_path = (output_dir / "article_semantic_ids.parquet")

    train_result.to_parquet(train_output_path, index=False, )

    validation_result.to_parquet(validation_output_path, index=False, )

    all_result.to_parquet(all_output_path, index=False, )


    print_coverage_summary(
        train_result=(train_result),
        validation_result=(validation_result),
        all_result=(all_result),
        embeddings_path=str(embeddings_path),
    )


    print("\n" + "=" * 70)

    print("SEMANTIC ID GENERATION COMPLETE")

    print("=" * 70)

    print("\nTrain Event C2 mapping:" f"\n  {train_event_mapping_path}")

    print("\nValidation article-to-Train event mapping:" f"\n  {validation_article_mapping_path}")

    print("\nTrain SID file:" f"\n  {train_output_path}")

    print("\nValidation SID file:" f"\n  {validation_output_path}")

    print("\nCombined final SID file:" f"\n  {all_output_path}")

    print("\nTrain SID rows       : " f"{len(train_result)}")

    print("Validation SID rows  : " f"{len(validation_result)}")

    print("Unique final articles: " f"{len(all_result)}")

    print("\nPreview:")

    print(all_result.head(10))

    print("=" * 70)

    return (train_result, validation_result, all_result, train_event_mapping, validation_article_mapping, )


if __name__ == "__main__":

    parser = (argparse.ArgumentParser())

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=("Directory containing " "article_master.parquet, " "validation_article_master.parquet, " "and article_embeddings.npy"),
    )

    parser.add_argument("--checkpoint", type=str, required=True, help=("Path to trained " "checkpoint_final.pt"), )

    parser.add_argument("--output_dir", type=str, default=("out/semantic_ids"), )

    parser.add_argument("--batch_size", type=int, default=512, )

    parser.add_argument("--num_workers", type=int, default=0, )

    # 변경: 시간 매칭 기본값 3일(72시간)
    parser.add_argument(
        "--event_time_window_days",
        type=float,
        default=3.0,
        help="Train 사건 시간 비교 범위(일). 기본 3일(72시간), 음수이면 시간 필터 미사용.",
    )

    args = (parser.parse_args())

    generate_semantic_ids(
        data_dir=(args.data_dir),
        checkpoint_path=(args.checkpoint),
        output_dir=(args.output_dir),
        batch_size=(args.batch_size),
        num_workers=(args.num_workers),
        event_time_window_days=(args.event_time_window_days if args.event_time_window_days >= 0 else None),
    ) 
