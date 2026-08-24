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

    total_articles = int(len(df))
    unique_c123 = int(df[["c1", "c2", "c3"]].drop_duplicates().shape[0])

    collision_mask = df.duplicated(
        subset=["c1", "c2", "c3"],
        keep=False,
    )
    collision_articles = int(collision_mask.sum())
    collision_rate = (
        collision_articles / total_articles if total_articles else float("nan")
    )

    result = {
        "Total Articles": total_articles,
        "Unique c123": unique_c123,
        "Collision Articles": collision_articles,
        "Collision Rate": collision_rate,
    }
    print_result(result)


if __name__ == "__main__":
    main()
