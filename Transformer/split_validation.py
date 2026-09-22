from pathlib import Path

import pandas as pd


# ============================================================
# 경로 설정
# ============================================================

# 현재 파일 위치 기준 Transformer/
BASE_DIR = Path(__file__).resolve().parent

input_path = (
    BASE_DIR
    / "datasets"
    / "mind"
    / "validation_sequences_1pos4neg.parquet"
)

validation_output_path = (
    BASE_DIR
    / "datasets"
    / "mind"
    / "validation_sequences_1pos4neg_half.parquet"
)

test_output_path = (
    BASE_DIR
    / "datasets"
    / "mind"
    / "test_sequences_1pos4neg.parquet"
)


# ============================================================
# 1:4 전처리 완료된 validation 불러오기
# ============================================================

df = pd.read_parquet(input_path)

print("Original validation rows:", len(df))


# ============================================================
# 필수 컬럼 확인
# ============================================================

required_columns = [
    "impression_id",
    "impression_time",
]

for col in required_columns:
    if col not in df.columns:
        raise ValueError(
            f"{col} column이 없습니다."
        )


# ============================================================
# impression_time datetime 변환
# ============================================================

df["impression_time"] = pd.to_datetime(
    df["impression_time"]
)


# ============================================================
# impression 단위 정보 생성
#
# positive가 2개 이상인 impression은 여러 row로 분리되어 있으므로
# 같은 impression이 validation/test 양쪽에 들어가지 않도록
# impression_id 기준으로 split
# ============================================================

impressions = (
    df[
        [
            "impression_id",
            "impression_time",
        ]
    ]
    .drop_duplicates(
        subset=["impression_id"]
    )
    .sort_values(
        [
            "impression_time",
            "impression_id",
        ]
    )
    .reset_index(drop=True)
)


print(
    "Unique impressions:",
    len(impressions),
)


# ============================================================
# 시간순 split
#
# 앞 50% impression = validation
# 뒤 50% impression = test
# ============================================================

split_idx = len(impressions) // 2

validation_impression_ids = set(
    impressions.iloc[:split_idx][
        "impression_id"
    ]
)

test_impression_ids = set(
    impressions.iloc[split_idx:][
        "impression_id"
    ]
)


# ============================================================
# 원래 row 기준으로 다시 분리
# ============================================================

validation_df = df[
    df["impression_id"].isin(
        validation_impression_ids
    )
].copy()

test_df = df[
    df["impression_id"].isin(
        test_impression_ids
    )
].copy()


# ============================================================
# 시간순으로 다시 정렬
# ============================================================

validation_df = (
    validation_df
    .sort_values(
        [
            "impression_time",
            "impression_id",
        ]
    )
    .reset_index(drop=True)
)

test_df = (
    test_df
    .sort_values(
        [
            "impression_time",
            "impression_id",
        ]
    )
    .reset_index(drop=True)
)


# ============================================================
# validation / test impression 중복 확인
# ============================================================

overlap = (
    set(validation_df["impression_id"])
    & set(test_df["impression_id"])
)

if overlap:
    raise ValueError(
        f"Validation/Test에 중복된 impression이 있습니다: "
        f"{len(overlap)}개"
    )


# ============================================================
# 시간 순서 검증
# ============================================================

validation_max_time = (
    validation_df["impression_time"].max()
)

test_min_time = (
    test_df["impression_time"].min()
)

if validation_max_time > test_min_time:
    print(
        "Warning: validation/test 시간 구간이 일부 겹칩니다."
    )


# ============================================================
# 저장
# ============================================================

validation_df.to_parquet(
    validation_output_path,
    index=False,
)

test_df.to_parquet(
    test_output_path,
    index=False,
)


# ============================================================
# 결과 확인
# ============================================================

print()

print(
    "New validation rows:",
    len(validation_df),
)

print(
    "Test rows          :",
    len(test_df),
)

print()

print(
    "Validation impressions:",
    validation_df["impression_id"].nunique(),
)

print(
    "Test impressions      :",
    test_df["impression_id"].nunique(),
)

print()

print(
    "Validation time:",
    validation_df["impression_time"].min(),
    "~",
    validation_df["impression_time"].max(),
)

print(
    "Test time      :",
    test_df["impression_time"].min(),
    "~",
    test_df["impression_time"].max(),
)

print()

print(
    "Validation/Test impression overlap:",
    len(overlap),
)

print()

print("Saved:")
print(validation_output_path)
print(test_output_path)