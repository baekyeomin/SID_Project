from pathlib import Path

import polars as pl


BASE_DIR = Path(__file__).resolve().parent

input_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "validation_sequences_1pos4neg.parquet"
)

validation_output_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "validation_sequences_1pos4neg_half.parquet"
)

test_output_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "test_sequences_1pos4neg.parquet"
)


# 원본 validation
df = pl.read_parquet(input_path)

print("Original validation rows:", df.height)


# 필수 컬럼 확인
required_columns = [
    "impression_id",
    "impression_time",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"{col} column이 없습니다.")


# impression 단위 정보
impressions = (
    df
    .select(
        [
            "impression_id",
            "impression_time",
        ]
    )
    .unique(
        subset=["impression_id"],
        keep="first",
    )
    .sort(
        [
            "impression_time",
            "impression_id",
        ]
    )
)


print("Unique impressions:", impressions.height)


# impression 기준 시간순 50:50
split_idx = impressions.height // 2

validation_ids = (
    impressions
    .slice(0, split_idx)
    .select("impression_id")
)

test_ids = (
    impressions
    .slice(
        split_idx,
        impressions.height - split_idx,
    )
    .select("impression_id")
)


# 원본 row 다시 분리
validation_df = (
    df
    .join(
        validation_ids,
        on="impression_id",
        how="semi",
    )
    .sort(
        [
            "impression_time",
            "impression_id",
        ]
    )
)

test_df = (
    df
    .join(
        test_ids,
        on="impression_id",
        how="semi",
    )
    .sort(
        [
            "impression_time",
            "impression_id",
        ]
    )
)


# overlap 확인
overlap = (
    validation_df
    .select("impression_id")
    .unique()
    .join(
        test_df
        .select("impression_id")
        .unique(),
        on="impression_id",
        how="inner",
    )
)

if overlap.height > 0:
    raise ValueError(
        f"Validation/Test impression overlap: {overlap.height}"
    )


# Polars + zstd로 저장
validation_df.write_parquet(
    validation_output_path,
    compression="zstd",
)

test_df.write_parquet(
    test_output_path,
    compression="zstd",
)


print()
print("New validation rows:", validation_df.height)
print("Test rows          :", test_df.height)

print()
print(
    "Validation impressions:",
    validation_df["impression_id"].n_unique(),
)

print(
    "Test impressions      :",
    test_df["impression_id"].n_unique(),
)

print()
print(
    "Validation/Test impression overlap:",
    overlap.height,
)


# 파일 크기
original_size = input_path.stat().st_size / (1024 ** 2)
validation_size = validation_output_path.stat().st_size / (1024 ** 2)
test_size = test_output_path.stat().st_size / (1024 ** 2)

print()
print(f"Original validation size : {original_size:.2f} MB")
print(f"Validation half size     : {validation_size:.2f} MB")
print(f"Test size                : {test_size:.2f} MB")

print()
print("Saved:")
print(validation_output_path)
print(test_output_path)