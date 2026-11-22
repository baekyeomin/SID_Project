import pandas as pd
from pathlib import Path

data_dir = Path(__file__).resolve().parent / "datasets"

for path in sorted(data_dir.rglob("*.parquet")):
    df = pd.read_parquet(path)

    print(f"\n=== {path.relative_to(data_dir)} ===")
    print("컬럼:", df.columns.tolist())
    print(df.head(3).to_string())