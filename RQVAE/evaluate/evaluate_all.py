from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run(script, extra_args):
    cmd = [sys.executable, str(Path(__file__).parent / script)] + extra_args
    print()
    print("#" * 90)
    print("RUN:", " ".join(cmd))
    print("#" * 90)
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
    parser.add_argument("--q2-size", type=int, required=True)
    parser.add_argument("--q3-size", type=int, required=True)
    parser.add_argument("--pairs", type=int, default=50000)
    args = parser.parse_args()

    common_sid = [
        "--sid", args.sid,
        "--data-dir", args.data_dir,
        "--split", args.split,
    ]

    run(
        "eval_reconstruction.py",
        [
            "--data-dir", args.data_dir,
            "--checkpoint", args.checkpoint,
        ],
    )

    run(
        "eval_c2_event_consistency.py",
        common_sid,
    )

    run(
        "eval_codebook_usage.py",
        common_sid + [
            "--q2-size", str(args.q2_size),
            "--q3-size", str(args.q3_size),
        ],
    )

    run(
        "eval_sid_collision.py",
        common_sid,
    )

    run(
        "eval_semantic_similarity.py",
        common_sid + [
            "--pairs", str(args.pairs),
        ],
    )

    run(
        "eval_c1_sanity.py",
        common_sid,
    )


if __name__ == "__main__":
    main()
