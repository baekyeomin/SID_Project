import pandas as pd
from pathlib import Path

path = Path(__file__).resolve().parent / "datasets/ebnerd/validation_article_master.parquet"
df = pd.read_parquet(path)

time_cols = [col for col in df.columns if any(key in col.lower() for key in ["time", "date", "publish"])]

print("시간 관련 컬럼:", time_cols)

for col in time_cols:
    times = pd.to_datetime(df[col], errors="coerce", utc=True)

    print(f"\n컬럼: {col}")
    print(f"전체 기사 수: {len(df):,}")
    print(f"발행 시각 누락: {times.isna().sum():,}")
    print(f"시간순 정렬 여부: {times.dropna().is_monotonic_increasing}")
    print(f"최초 발행 시각: {times.min()}")
    print(f"최종 발행 시각: {times.max()}")
    print("처음 5개:", times.head().tolist())
    print("마지막 5개:", times.tail().tolist())