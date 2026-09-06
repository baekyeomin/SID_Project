from pathlib import Path

import pandas as pd


# 현재 파일 위치 기준 Transformer/
BASE_DIR = Path(__file__).resolve().parent

input_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "validation_sequences.parquet"
)

validation_output_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "validation_sequences_half.parquet"
)

test_output_path = (
    BASE_DIR
    / "datasets"
    / "ebnerd"
    / "test_sequences.parquet"
)


# ============================================================
# 기존 validation 불러오기
# ============================================================

df = pd.read_parquet(input_path)

print("Original validation rows:", len(df))


# ============================================================
# impression_time 확인
# ============================================================

if "impression_time" not in df.columns:
    raise ValueError(
        "impression_time column이 없습니다. "
        "시간 기준 split을 하려면 impression_time이 필요합니다."
    )


# datetime 형식으로 변환
df["impression_time"] = pd.to_datetime(
    df["impression_time"]
)


# ============================================================
# 시간순 정렬
# ============================================================

df = df.sort_values(
    "impression_time"
).reset_index(drop=True)


# ============================================================
# 앞 50% = validation
# 뒤 50% = test
# ============================================================

split_idx = len(df) // 2

validation_df = df.iloc[:split_idx].copy()
test_df = df.iloc[split_idx:].copy()


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
print("New validation rows:", len(validation_df))
print("Test rows          :", len(test_df))

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
print("Saved:")
print(validation_output_path)
print(test_output_path)