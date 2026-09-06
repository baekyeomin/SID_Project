from pathlib import Path
import pandas as pd

# 현재 practice.py가 있는 Transformer 폴더
BASE_DIR = Path(__file__).resolve().parent

path = BASE_DIR / "datasets" / "ebnerd" / "article_semantic_ids.parquet"

print("찾는 경로:", path)
print("파일 존재 여부:", path.exists())

df = pd.read_parquet(
    path,
    columns=["c4"],
)

c4_vocab_size = int(df["c4"].max()) + 1

print("c4 min       :", df["c4"].min())
print("c4 max       :", df["c4"].max())
print("c4 vocab size:", c4_vocab_size)