import argparse
import importlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from modules.rqvae import RqVae


def ensure_torch_serialization_compatibility() -> None:
    # 일부 Colab 환경에서 사라지는 torch.Tensor / torch._utils 속성 복구
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
                raise RuntimeError("torch.Tensor attribute를 복구하지 못했습니다.") from exc

        setattr(torch, "Tensor", tensor_class)

    if not hasattr(torch, "_utils"):
        try:
            torch_utils = importlib.import_module("torch._utils")
            setattr(torch, "_utils", torch_utils)
        except Exception as exc:
            raise RuntimeError("torch._utils attribute를 복구하지 못했습니다.") from exc


def safe_torch_load(checkpoint_path, map_location):
    # checkpoint load 전 PyTorch serialization 상태 확인
    ensure_torch_serialization_compatibility()

    return torch.load(
        checkpoint_path,
        map_location=map_location,
        weights_only=False,
    )


class SemanticIdDataset(Dataset):
    # SID 생성에 필요한 article 정보
    REQUIRED_COLUMNS = {
        "article_id",
        "embedding_row",
        "model_category_id",
    }

    def __init__(
        self,
        master_path: str,
        embeddings_path: str,
    ) -> None:
        super().__init__()

        self.master_path = Path(master_path)
        self.embeddings_path = Path(embeddings_path)

        # article 정보 로드
        self.master = pd.read_parquet(
            self.master_path
        ).reset_index(drop=True)

        missing_columns = self.REQUIRED_COLUMNS - set(self.master.columns)

        if missing_columns:
            raise ValueError(
                f"{self.master_path} is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        # 전체 embedding을 RAM에 올리지 않고 필요한 row만 읽음
        self.embeddings = np.load(
            self.embeddings_path,
            mmap_mode="r",
        )

        if self.embeddings.ndim != 2:
            raise ValueError(
                "article_embeddings.npy must have shape "
                "[num_articles, embedding_dim]. "
                f"Got {self.embeddings.shape}."
            )

        # embedding_row가 실제 embedding 범위 안에 있는지 확인
        rows = self.master["embedding_row"].to_numpy()

        if len(rows) > 0:
            min_row = int(rows.min())
            max_row = int(rows.max())

            if min_row < 0:
                raise ValueError("embedding_row contains negative values.")

            if max_row >= len(self.embeddings):
                raise ValueError(
                    "embedding_row points outside article_embeddings.npy. "
                    f"Maximum embedding_row={max_row}, "
                    f"but embeddings contain {len(self.embeddings)} rows."
                )

    def __len__(self) -> int:
        return len(self.master)

    def __getitem__(self, index: int):
        row = self.master.iloc[index]

        embedding_row = int(row["embedding_row"])
        category_id = int(row["model_category_id"])

        # 해당 article의 E5 embedding x(a) 로드
        embedding = np.asarray(
            self.embeddings[embedding_row],
            dtype=np.float32,
        ).copy()

        x = torch.from_numpy(embedding)

        return {
            "x": x,
            "category_id": torch.tensor(category_id, dtype=torch.long),
            "article_id": str(row["article_id"]),
            "embedding_row": torch.tensor(embedding_row, dtype=torch.long),
        }


def load_rqvae(
    checkpoint_path: str,
    device: torch.device,
) -> RqVae:
    # 학습 완료된 checkpoint에서 frozen RQ-VAE 복원
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"RQ-VAE checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 70)
    print("LOADING TRAINED RQ-VAE")
    print("=" * 70)

    checkpoint = safe_torch_load(
        checkpoint_path=checkpoint_path,
        map_location=device,
    )

    if "model_config" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model_config'.")

    if "model" not in checkpoint:
        raise KeyError("Checkpoint does not contain 'model'.")

    model_config = checkpoint["model_config"]

    print("Model config:")
    for key, value in model_config.items():
        print(f"  {key}: {value}")

    # checkpoint와 동일한 구조의 RQ-VAE 생성
    model = RqVae(**model_config)

    # 학습된 Encoder / Decoder / Q1 / Q2 / Q3 parameter 복원
    model.load_state_dict(checkpoint["model"])
    model = model.to(device)

    # inference 전용으로 고정
    model.eval()

    for parameter in model.parameters():
        parameter.requires_grad_(False)

    print(f"\nLoaded checkpoint: {checkpoint_path}")

    if "iter" in checkpoint:
        print(f"Training iteration: {checkpoint['iter']}")

    print("RQ-VAE frozen.")
    print("=" * 70 + "\n")

    return model


@torch.inference_mode()
def generate_split_semantic_ids(
    model: RqVae,
    master_path: str,
    embeddings_path: str,
    split: str,
    device: torch.device,
    batch_size: int = 512,
    num_workers: int = 0,
) -> pd.DataFrame:
    # 한 split의 모든 기사에 모델 기반 [c1, c2, c3] 생성 (c4는 이후 후처리)
    dataset = SemanticIdDataset(
        master_path=master_path,
        embeddings_path=embeddings_path,
    )

    # 입력 embedding 차원이 학습된 RQ-VAE와 같은지 확인
    embedding_dim = dataset.embeddings.shape[1]

    if embedding_dim != model.input_dim:
        raise ValueError(
            "Article embedding dimension does not match RQ-VAE input_dim. "
            f"article_embeddings.npy={embedding_dim}, "
            f"RQ-VAE input_dim={model.input_dim}"
        )

    # category ID가 Q1 codebook 범위 안에 있는지 확인
    if len(dataset.master) > 0:
        category_values = dataset.master["model_category_id"].astype(int)

        min_category = int(category_values.min())
        max_category = int(category_values.max())

        if min_category < 0:
            raise ValueError(
                "model_category_id must be >= 0. "
                f"Found minimum={min_category}."
            )

        if max_category >= model.num_categories:
            raise ValueError(
                "model_category_id exceeds Q1 codebook range. "
                f"Maximum category ID={max_category}, "
                f"Q1 size={model.num_categories}. "
                f"Category IDs must be in [0, {model.num_categories - 1}]."
            )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )

    article_ids_all = []
    embedding_rows_all = []
    c1_all = []
    c2_all = []
    c3_all = []

    print(f"\nGenerating Semantic IDs for {split}...")
    print(f"Articles: {len(dataset)}")

    for batch_index, batch in enumerate(dataloader):
        # 기사 embedding을 모델 device와 dtype에 맞춤
        x = batch["x"].to(
            device=device,
            dtype=next(model.encoder.parameters()).dtype,
            non_blocking=True,
        )

        # Q1에서 그대로 사용할 category ID
        category_ids = batch["category_id"].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        # frozen RQ-VAE로 [c1, c2, c3] 생성 (c4는 모델 밖에서 후처리)
        output = model.get_semantic_ids(
            x=x,
            category_ids=category_ids,
        )

        semantic_ids = output.sem_ids.detach().cpu()

        article_ids_all.extend(batch["article_id"])
        embedding_rows_all.extend(
            batch["embedding_row"].cpu().tolist()
        )
        c1_all.extend(semantic_ids[:, 0].tolist())
        c2_all.extend(semantic_ids[:, 1].tolist())
        c3_all.extend(semantic_ids[:, 2].tolist())

        # 20 batch마다 진행 상황 출력
        if (batch_index + 1) % 20 == 0:
            processed = min(
                (batch_index + 1) * batch_size,
                len(dataset),
            )
            print(f"  {processed}/{len(dataset)}")

    result = pd.DataFrame({
        "article_id": article_ids_all,
        "embedding_row": embedding_rows_all,
        "split": split,
        "c1": c1_all,
        "c2": c2_all,
        "c3": c3_all,
    })

    if len(result) != len(dataset):
        raise RuntimeError(
            "Number of generated Semantic IDs does not match number of articles. "
            f"articles={len(dataset)}, semantic_ids={len(result)}"
        )

    print(f"{split}: {len(result)} Semantic IDs generated.")

    return result



def assign_disambiguation_c4(
    result: pd.DataFrame,
) -> pd.DataFrame:
    """
    같은 (c1, c2, c3)를 가진 기사들을 구분하기 위한 후처리용 c4를 부여한다.

    - c4는 RQ-VAE codebook / 학습 / residual 계산과 무관하다.
    - 같은 (c1, c2, c3) 그룹 안에서 등장 순서대로 0, 1, 2, ...를 부여한다.
    - 중복이 없는 기사도 c4=0을 갖는다.
    - Train + Validation을 합친 뒤 적용해야 전체 데이터 기준 최종 SID가 유일해진다.
    """
    result = result.copy()

    result["c4"] = (
        result
        .groupby(["c1", "c2", "c3"], sort=False)
        .cumcount()
        .astype(np.int64)
    )

    return result


def print_sid_statistics(
    result: pd.DataFrame,
    split: str,
) -> None:
    # 생성된 Semantic ID 사용 분포 및 c4 구분 결과 확인
    unique_c123 = (
        result[["c1", "c2", "c3"]]
        .drop_duplicates()
        .shape[0]
    )

    unique_full_sid = (
        result[["c1", "c2", "c3", "c4"]]
        .drop_duplicates()
        .shape[0]
    )

    duplicated_c123_articles = int(
        result.duplicated(
            subset=["c1", "c2", "c3"],
            keep=False,
        ).sum()
    )

    print("\n" + "=" * 70)
    print(f"{split.upper()} SID STATISTICS")
    print("=" * 70)
    print(f"Articles             : {len(result)}")
    print(f"Unique c1            : {result['c1'].nunique()}")
    print(f"Unique c2            : {result['c2'].nunique()}")
    print(f"Unique c3            : {result['c3'].nunique()}")
    print(f"Unique (c1,c2,c3)    : {unique_c123}")
    print(f"Articles in dup c123 : {duplicated_c123_articles}")
    print(f"Unique final SID     : {unique_full_sid}")
    print(
        f"Final SID uniqueness : "
        f"{unique_full_sid / max(len(result), 1):.4f}"
    )
    print(f"Max c4               : {int(result['c4'].max()) if len(result) else 0}")
    print("=" * 70)

def print_coverage_summary(
    train_result: pd.DataFrame,
    validation_result: pd.DataFrame,
    embeddings_path: str,
) -> None:
    # Train+Validation이 전체 embedding row를 모두 참조하는지 확인
    embeddings = np.load(
        embeddings_path,
        mmap_mode="r",
    )

    num_embedding_rows = len(embeddings)

    all_results = pd.concat(
        [train_result, validation_result],
        ignore_index=True,
    )

    referenced_rows = set(
        all_results["embedding_row"].astype(int).tolist()
    )

    all_embedding_rows = set(range(num_embedding_rows))
    missing_rows = all_embedding_rows - referenced_rows
    duplicate_reference_count = len(all_results) - len(referenced_rows)

    print("\n" + "=" * 70)
    print("SEMANTIC ID COVERAGE")
    print("=" * 70)
    print(f"article_embeddings.npy rows : {num_embedding_rows}")
    print(f"Train master articles       : {len(train_result)}")
    print(f"Validation master articles  : {len(validation_result)}")
    print(f"Total SID rows              : {len(all_results)}")
    print(f"Unique referenced rows      : {len(referenced_rows)}")
    print(f"Duplicate references        : {duplicate_reference_count}")

    if len(missing_rows) == 0:
        print("\nAll rows in article_embeddings.npy are referenced by Train/Validation.")
        print("=> Every referenced article embedding received a Semantic ID.")
    else:
        print(
            f"\nWARNING: {len(missing_rows)} embedding rows are not referenced "
            "by either master file."
        )
        print("=> Those rows did NOT receive SID.")
        print(f"Example missing rows: {sorted(missing_rows)[:20]}")

    print("=" * 70 + "\n")


def resolve_input_files(data_dir: Path):
    # EB-NeRD의 Train / Validation master 및 embedding 파일 경로 확인
    preferred_train_master = data_dir / "article_master.parquet"
    legacy_train_master = data_dir / "train_article_master.parquet"

    if preferred_train_master.exists():
        train_master_path = preferred_train_master
    elif legacy_train_master.exists():
        train_master_path = legacy_train_master
        print("WARNING: Using legacy train_article_master.parquet")
    else:
        raise FileNotFoundError(
            "Train master file not found. Expected one of:\n"
            f"  {preferred_train_master}\n"
            f"  {legacy_train_master}"
        )

    validation_master_path = (
        data_dir / "validation_article_master.parquet"
    )

    if not validation_master_path.exists():
        raise FileNotFoundError(
            f"Validation master file not found: {validation_master_path}"
        )

    embeddings_path = data_dir / "article_embeddings.npy"

    if not embeddings_path.exists():
        raise FileNotFoundError(
            f"Embedding file not found: {embeddings_path}"
        )

    return (
        train_master_path,
        validation_master_path,
        embeddings_path,
    )


def generate_semantic_ids(
    data_dir: str,
    checkpoint_path: str,
    output_dir: str,
    batch_size: int = 512,
    num_workers: int = 0,
):
    # 최종 frozen RQ-VAE로 Train/Validation 전체 기사 SID 생성
    data_dir = Path(data_dir)
    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_master_path, validation_master_path, embeddings_path = (
        resolve_input_files(data_dir)
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n" + "=" * 70)
    print("SEMANTIC ID GENERATION CONFIG")
    print("=" * 70)
    print(f"Train master      : {train_master_path}")
    print(f"Validation master : {validation_master_path}")
    print(f"Embeddings        : {embeddings_path}")
    print(f"Checkpoint        : {checkpoint_path}")
    print(f"Output directory  : {output_dir}")
    print(f"Batch size        : {batch_size}")
    print("=" * 70 + "\n")

    # GPU가 있으면 CUDA, 없으면 CPU 사용
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Device: {device}")

    # checkpoint_final.pt에서 최종 RQ-VAE 복원
    model = load_rqvae(
        checkpoint_path=str(checkpoint_path),
        device=device,
    )

    # Train 전체 기사 SID 생성
    train_result = generate_split_semantic_ids(
        model=model,
        master_path=str(train_master_path),
        embeddings_path=str(embeddings_path),
        split="train",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Validation 전체 기사 SID 생성
    validation_result = generate_split_semantic_ids(
        model=model,
        master_path=str(validation_master_path),
        embeddings_path=str(embeddings_path),
        split="validation",
        device=device,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    # Train + Validation을 먼저 합친 뒤,
    # 전체 데이터 기준 동일한 (c1, c2, c3)를 c4=0,1,2,...로 구분
    all_result = pd.concat(
        [train_result, validation_result],
        ignore_index=True,
    )

    all_result = assign_disambiguation_c4(
        result=all_result,
    )

    # c4가 부여된 결과를 다시 split별로 분리
    train_result = (
        all_result[all_result["split"] == "train"]
        .reset_index(drop=True)
    )

    validation_result = (
        all_result[all_result["split"] == "validation"]
        .reset_index(drop=True)
    )

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
        split="train+validation",
    )

    # c4까지 포함된 Train/Validation 결과를 각각 저장
    train_output_path = (
        output_dir / "train_article_semantic_ids.parquet"
    )

    validation_output_path = (
        output_dir / "validation_article_semantic_ids.parquet"
    )

    train_result.to_parquet(
        train_output_path,
        index=False,
    )

    validation_result.to_parquet(
        validation_output_path,
        index=False,
    )

    # Train + Validation을 합친 전체 SID 파일 저장
    all_output_path = (
        output_dir / "article_semantic_ids.parquet"
    )

    all_result.to_parquet(
        all_output_path,
        index=False,
    )

    # 모든 embedding row에 SID가 생성되었는지 확인
    print_coverage_summary(
        train_result=train_result,
        validation_result=validation_result,
        embeddings_path=str(embeddings_path),
    )

    print("\n" + "=" * 70)
    print("SEMANTIC ID GENERATION COMPLETE")
    print("=" * 70)
    print(f"\nTrain SID file:\n  {train_output_path}")
    print(f"\nValidation SID file:\n  {validation_output_path}")
    print(f"\nCombined SID file:\n  {all_output_path}")
    print(f"\nTrain articles      : {len(train_result)}")
    print(f"Validation articles : {len(validation_result)}")
    print(f"Total articles      : {len(all_result)}")
    print("\nPreview:")
    print(all_result.head(10))
    print("=" * 70)

    return train_result, validation_result, all_result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()  # command line argument 설정

    parser.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help=(
            "Directory containing article_master.parquet, "
            "validation_article_master.parquet, "
            "and article_embeddings.npy"
        ),
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained checkpoint_final.pt",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/semantic_ids",
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

    args = parser.parse_args()

    generate_semantic_ids(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )