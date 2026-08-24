from __future__ import annotations

import argparse
import numpy as np

from common import load_sid_eval_df, print_result


def group_indices(df, cols):
    return {
        key: np.asarray(indices, dtype=np.int64)
        for key, indices in df.groupby(cols, sort=False).indices.items()
    }


def sample_same_child_pairs(df, parent_cols, child_col, n_pairs, rng):
    """
    positive pair:
    - C2: 같은 c1 + 같은 c2
    - C3: 같은 c1,c2 + 같은 c3
    """
    group_cols = list(parent_cols) + [child_col]
    groups = group_indices(df, group_cols)
    eligible = [idx for idx in groups.values() if len(idx) >= 2]

    if not eligible:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.empty((n_pairs, 2), dtype=np.int64)

    for i in range(n_pairs):
        idx = eligible[rng.integers(len(eligible))]

        a_pos = rng.integers(len(idx))
        b_pos = rng.integers(len(idx) - 1)
        if b_pos >= a_pos:
            b_pos += 1

        pairs[i] = (idx[a_pos], idx[b_pos])

    return pairs


def sample_diff_child_pairs(df, parent_cols, child_col, n_pairs, rng):
    """
    negative pair:
    - C2: 같은 c1 + 다른 c2
    - C3: 같은 c1,c2 + 다른 c3
    """
    parent_key = parent_cols[0] if len(parent_cols) == 1 else list(parent_cols)
    eligible = []

    for _, parent_df in df.groupby(parent_key, sort=False):
        child_groups = [
            g.index.to_numpy(dtype=np.int64)
            for _, g in parent_df.groupby(child_col, sort=False)
        ]
        if len(child_groups) >= 2:
            eligible.append(child_groups)

    if not eligible:
        return np.empty((0, 2), dtype=np.int64)

    pairs = np.empty((n_pairs, 2), dtype=np.int64)

    for i in range(n_pairs):
        child_groups = eligible[rng.integers(len(eligible))]

        a_child = rng.integers(len(child_groups))
        b_child = rng.integers(len(child_groups) - 1)
        if b_child >= a_child:
            b_child += 1

        ga = child_groups[a_child]
        gb = child_groups[b_child]

        a = ga[rng.integers(len(ga))]
        b = gb[rng.integers(len(gb))]

        pairs[i] = (a, b)

    return pairs


def mean_cosine_for_pairs(df, pairs, embeddings_path, batch_size=4096):
    if len(pairs) == 0:
        return np.nan

    embeddings = np.load(embeddings_path, mmap_mode="r")
    embedding_rows = df["embedding_row"].astype(np.int64).to_numpy()

    total = 0.0
    count = 0

    for start in range(0, len(pairs), batch_size):
        p = pairs[start:start + batch_size]

        rows_a = embedding_rows[p[:, 0]]
        rows_b = embedding_rows[p[:, 1]]

        a = np.asarray(embeddings[rows_a], dtype=np.float32)
        b = np.asarray(embeddings[rows_b], dtype=np.float32)

        a /= np.clip(np.linalg.norm(a, axis=1, keepdims=True), 1e-12, None)
        b /= np.clip(np.linalg.norm(b, axis=1, keepdims=True), 1e-12, None)

        cos = np.sum(a * b, axis=1)

        total += float(cos.sum())
        count += len(cos)

    return total / count


def compute_delta(df, embeddings_path, parent_cols, child_col, n_pairs, seed, batch_size):
    work = df.reset_index(drop=True).copy()
    rng = np.random.default_rng(seed)

    same_pairs = sample_same_child_pairs(
        work, parent_cols, child_col, n_pairs, rng
    )
    diff_pairs = sample_diff_child_pairs(
        work, parent_cols, child_col, n_pairs, rng
    )

    same_cos = mean_cosine_for_pairs(
        work, same_pairs, embeddings_path, batch_size
    )
    diff_cos = mean_cosine_for_pairs(
        work, diff_pairs, embeddings_path, batch_size
    )

    if np.isfinite(same_cos) and np.isfinite(diff_cos):
        delta = same_cos - diff_cos
    else:
        delta = np.nan

    return same_cos, diff_cos, delta, len(same_pairs), len(diff_pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sid", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split", default="all", choices=["all", "train", "validation"])
    parser.add_argument("--pairs", type=int, default=50000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    args = parser.parse_args()

    df, embeddings_path = load_sid_eval_df(
        args.sid,
        args.data_dir,
        split=args.split,
    )

    if "embedding_row" not in df.columns:
        raise KeyError("Semantic similarity 계산에 embedding_row가 필요합니다.")

    c2_same, c2_diff, delta_c2, c2_pos_n, c2_neg_n = compute_delta(
        df=df,
        embeddings_path=embeddings_path,
        parent_cols=["c1"],
        child_col="c2",
        n_pairs=args.pairs,
        seed=args.seed,
        batch_size=args.batch_size,
    )

    c3_same, c3_diff, delta_c3, c3_pos_n, c3_neg_n = compute_delta(
        df=df,
        embeddings_path=embeddings_path,
        parent_cols=["c1", "c2"],
        child_col="c3",
        n_pairs=args.pairs,
        seed=args.seed + 1,
        batch_size=args.batch_size,
    )

    print("[C2]")
    print(f"Same c1+c2 cosine      : {c2_same:.6f}")
    print(f"Same c1, diff c2 cosine: {c2_diff:.6f}")
    print(f"ΔC2 Semantic Similarity: {delta_c2:.6f}")
    print()
    print("[C3]")
    print(f"Same c1+c2+c3 cosine   : {c3_same:.6f}")
    print(f"Same c1+c2, diff c3    : {c3_diff:.6f}")
    print(f"ΔC3 Semantic Similarity: {delta_c3:.6f}")

    # ΔC3의 positive pair는 동일 c123 그룹이 2개 이상인 경우에만 존재.
    if not np.isfinite(delta_c3):
        print()
        print(
            "주의: ΔC3를 계산할 positive/negative pair가 충분하지 않습니다. "
            "c123 collision이 거의 없으면 ΔC3는 NaN이 될 수 있습니다."
        )


if __name__ == "__main__":
    main()
