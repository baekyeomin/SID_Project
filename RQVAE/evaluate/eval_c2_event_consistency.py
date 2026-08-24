from __future__ import annotations

import argparse
import numpy as np

from common import load_sid_eval_df, print_result


def compute_c2_event_consistency(df, exclude_event_ids=None):
    if "event_id" not in df.columns:
        raise KeyError("event_id column이 필요합니다.")

    work = df.dropna(subset=["event_id", "c2"]).copy()

    if exclude_event_ids:
        work = work[~work["event_id"].isin(exclude_event_ids)]

    event_scores = []

    for _, group in work.groupby("event_id", sort=False):
        # 기사 하나짜리 event는 consistency를 평가할 수 없어 제외
        if len(group) < 2:
            continue

        dominant_ratio = group["c2"].value_counts().iloc[0] / len(group)
        event_scores.append(dominant_ratio)

    if not event_scores:
        return np.nan, 0

    return float(np.mean(event_scores)), len(event_scores)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
    parser.add_argument(
        "--exclude-event-id",
        type=int,
        action="append",
        default=[-1],
        help="제외할 event_id. 여러 개면 옵션을 반복해서 사용.",
    )
    args = parser.parse_args()

    df, _ = load_sid_eval_df(args.sid, args.data_dir, split=args.split)

    score, n_events = compute_c2_event_consistency(
        df,
        exclude_event_ids=set(args.exclude_event_id),
    )

    result = {
        "C2 Event Consistency": score,
        "Evaluated Multi-article Events": n_events,
    }
    print_result(result)


if __name__ == "__main__":
    main()
