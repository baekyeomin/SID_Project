from __future__ import annotations

import argparse

from common import load_sid_eval_df, print_result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
    parser.add_argument("--q2-size", type=int, required=True)
    parser.add_argument("--q3-size", type=int, required=True)
    args = parser.parse_args()

    df, _ = load_sid_eval_df(args.sid, args.data_dir, split=args.split)

    q2_used = int(df["c2"].nunique())
    q3_used = int(df["c3"].nunique())

    result = {
        "Total Articles": int(len(df)),
        "Q2 Used Codes": q2_used,
        "Q2 Utilization": q2_used / args.q2_size,
        "Q3 Used Codes": q3_used,
        "Q3 Utilization": q3_used / args.q3_size,
    }
    print_result(result)


if __name__ == "__main__":
    main()
