from __future__ import annotations

import argparse

from common import load_sid_eval_df, print_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
    args = parser.parse_args()

    df, _ = load_sid_eval_df(args.sid, args.data_dir, split=args.split)

    if "model_category_id" not in df.columns:
        raise KeyError("model_category_id가 없어 C1 sanity check를 할 수 없습니다.")

    accuracy = (
        df["c1"].astype(int)
        == df["model_category_id"].astype(int)
    ).mean()

    print_result({
        "C1 Category Accuracy": float(accuracy),
    })


if __name__ == "__main__":
    main()
