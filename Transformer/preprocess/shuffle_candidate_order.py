"""1pos4neg parquet??candidate ?쒖꽌留??욌뒗??

???꾩슂?쒓?
    build_1pos4neg_sequences.py??STEP 15-10??positive瑜???긽 留??욎뿉 遺숈씠怨?    label??[1, 0, 0, 0, 0] 由ы꽣?대줈 留뚮뱺?? 洹몃옒??positive媛 ?덉쇅 ?놁씠
    index 0???⑤떎.

    ?됯???Top-1? np.argmax / torch.argmax瑜??곕뒗???숈젏?대㈃ 媛?????
    index瑜?怨좊Ⅸ?? positive媛 ??긽 0踰덉씠硫??숈젏?????먮━媛 ?꾨? ?뺣떟?쇰줈
    泥섎━?섏뼱 Top-1???ㅼ젣 蹂蹂꾨젰蹂대떎 ?믨쾶 ?섏삩??

臾댁뾿??諛붽씀?붽?
    candidate? 蹂묐젹??而щ읆??"?먯냼 ?쒖꽌"留?諛붽씔?? ?ъ깦?뚮쭅???꾨땲??

    諛붽씀??寃?        candidate_article_ids
        candidate_c1, candidate_c2, candidate_c3, candidate_c4
        candidate_labels

    嫄대뱶由ъ? ?딅뒗 寃?        positive 湲곗궗 ?좏깮, negative 4媛??섑뵆留? negative candidate set,
        history_*, target_*, SID 媛? split, impression ?? ???쒖꽌, ??媛쒖닔

    target_* ??builder媛 candidate_*.list.first() 濡?留뚮뱺 媛믪씠??    positive ?먯껜瑜??닿퀬 ?덈떎. candidate ?쒖꽌? 臾닿??섎?濡??욎? ?딅뒗??

row ?⑥쐞 deterministic seed
    sha256(seed | split_name | impression_id | positive_article_id | row_uid)

    ?꾩뿭 RNG ?곹깭???섏〈?섏? ?딅뒗?? ???쒖꽌瑜?諛붽씀嫄곕굹 ?쇰?留??ㅼ떆 留뚮뱾?대룄
    媛숈? permutation???섏삩??

?ъ슜踰?    # 癒쇱? 10?됰쭔 誘몃━蹂닿린 (?뚯씪???곗? ?딅뒗??
    python -m src.shuffle_candidate_order \
        --input-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_v1/post_rqvae \
        --preview 10

    # ?뺤씤 ???앹꽦
    python -m src.shuffle_candidate_order \
        --input-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_v1/post_rqvae \
        --output-dir data/output/experiments/normalize_v2_uni_lu005_m05_candidate_1pos4neg_shuffled_v2/post_rqvae \
        --seed 42
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl


DEFAULT_SEED = 42

# ?쒖꽌瑜??④퍡 諛붽퓭???섎뒗 而щ읆.
# build_1pos4neg_sequences.py??CANDIDATE_COLUMNS? 媛숇떎.
CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]

# ?덈? ?욎쑝硫????섎뒗 而щ읆.
NEVER_SHUFFLE_PREFIXES = ("history_", "target_")

# 湲곕낯 泥섎━ ????뚯씪怨?split ?대쫫.
DEFAULT_FILES = [
    ("train", "train_sequences_1pos4neg.parquet"),
    ("validation", "validation_sequences_1pos4neg.parquet"),
]


def section(title: str) -> None:
    print()
    print("=" * 74)
    print(title)
    print("=" * 74)


def row_permutation(
    seed: int,
    split_name: str,
    impression_id: Any,
    positive_article_id: Any,
    row_uid: int,
    length: int,
) -> list[int]:
    """row ?섎굹??permutation. 媛숈? ?낅젰?대㈃ ??긽 媛숈? 媛믪씠 ?섏삩??"""

    key = "|".join([
        str(seed),
        str(split_name),
        str(impression_id),
        str(positive_article_id),
        str(row_uid),
    ])

    digest = hashlib.sha256(key.encode("utf-8")).digest()
    row_seed = int.from_bytes(digest[:8], "big", signed=False)

    return np.random.default_rng(row_seed).permutation(length).tolist()


def positive_position(labels: list[int]) -> int:
    positions = [index for index, value in enumerate(labels) if value == 1]

    if len(positions) != 1:
        raise ValueError(f"row??positive ?섍? 1???꾨떃?덈떎: {len(positions)}媛?)

    return positions[0]


def position_distribution(df: pl.DataFrame) -> dict[str, Any]:
    label_lists = df["candidate_labels"].to_list()
    width = len(label_lists[0]) if label_lists else 0

    counts = [0] * width

    for labels in label_lists:
        counts[positive_position(list(labels))] += 1

    total = max(sum(counts), 1)
    uniform = 1.0 / width if width else 0.0

    return {
        "counts": counts,
        "ratios": [c / total for c in counts],
        "total": sum(counts),
        "max_deviation_from_uniform": max(
            (abs(c / total - uniform) for c in counts), default=0.0
        ),
    }


def print_distribution(title: str, dist: dict[str, Any]) -> None:
    print()
    print(f"  {title}")
    print(f"  {'index':<9}{'count':>12}{'%':>10}")
    print("  " + "-" * 31)

    for index, (count, ratio) in enumerate(zip(dist["counts"], dist["ratios"])):
        print(f"  index {index:<3}{count:>12,}{ratio:>10.2%}")

    print(f"  洹좊벑 ?鍮?理쒕? ?몄감 : {dist['max_deviation_from_uniform']:.2%}")


def check_columns(df: pl.DataFrame) -> tuple[list[str], list[str]]:
    """?욎쓣 而щ읆怨?洹몃?濡???而щ읆??媛瑜몃떎."""

    present = [c for c in CANDIDATE_COLUMNS if c in df.columns]
    missing = [c for c in CANDIDATE_COLUMNS if c not in df.columns]

    if "candidate_labels" not in present:
        raise ValueError("candidate_labels 而щ읆???놁뒿?덈떎.")

    width = len(df["candidate_labels"].to_list()[0])

    for column in present:
        lengths = {len(v) for v in df[column].head(2000).to_list()}

        if lengths != {width}:
            raise ValueError(
                f"{column}??湲몄씠媛 ?쇱젙?섏? ?딆뒿?덈떎: {sorted(lengths)}"
            )

    untouched = [c for c in df.columns if c not in present]

    for column in untouched:
        if column.startswith("candidate_"):
            raise ValueError(
                f"{column}??candidate_濡??쒖옉?섎뒗????곸뿉??鍮좎죱?듬땲?? "
                "CANDIDATE_COLUMNS瑜??뺤씤?섏꽭??"
            )

    if missing:
        print(f"  二쇱쓽: ?뚯씪???녿뒗 candidate 而щ읆 {missing}")

    return present, untouched


def shuffle_frame(
    df: pl.DataFrame,
    columns: list[str],
    split_name: str,
    seed: int,
) -> tuple[pl.DataFrame, list[list[int]]]:
    """candidate 而щ읆??媛숈? permutation???곸슜?쒕떎."""

    label_lists = df["candidate_labels"].to_list()
    article_lists = df["candidate_article_ids"].to_list()

    positive_positions = [positive_position(list(v)) for v in label_lists]
    positive_ids = [
        article_lists[i][positive_positions[i]] for i in range(df.height)
    ]

    impression_ids = (
        df["impression_id"].to_list()
        if "impression_id" in df.columns
        else list(range(df.height))
    )

    # 媛숈? (impression_id, positive)媛 ?щ윭 row濡??섏삱 寃쎌슦瑜??꾪븳 ?쒕쾲
    seen: dict[tuple[Any, Any], int] = {}
    row_uids: list[int] = []

    for impression_id, positive_id in zip(impression_ids, positive_ids):
        key = (impression_id, positive_id)
        row_uids.append(seen.get(key, 0))
        seen[key] = seen.get(key, 0) + 1

    permutations = [
        row_permutation(
            seed=seed,
            split_name=split_name,
            impression_id=impression_ids[i],
            positive_article_id=positive_ids[i],
            row_uid=row_uids[i],
            length=len(label_lists[i]),
        )
        for i in range(df.height)
    ]

    new_series = []

    for column in columns:
        values = df[column].to_list()

        new_series.append(
            pl.Series(
                name=column,
                values=[
                    [row[index] for index in permutation]
                    for row, permutation in zip(values, permutations)
                ],
                dtype=df[column].dtype,
            )
        )

    return df.with_columns(new_series), permutations


def print_preview(
    original: pl.DataFrame,
    shuffled: pl.DataFrame,
    permutations: list[list[int]],
    n: int,
) -> None:
    section(f"誘몃━蹂닿린 {n}??)

    for i in range(min(n, original.height)):
        before_ids = list(original["candidate_article_ids"].to_list()[i])
        after_ids = list(shuffled["candidate_article_ids"].to_list()[i])
        before_labels = list(original["candidate_labels"].to_list()[i])
        after_labels = list(shuffled["candidate_labels"].to_list()[i])

        before_sids = [
            tuple(
                int(original[f"candidate_c{level}"].to_list()[i][k])
                for level in (1, 2, 3, 4)
            )
            for k in range(len(before_ids))
        ]
        after_sids = [
            tuple(
                int(shuffled[f"candidate_c{level}"].to_list()[i][k])
                for level in (1, 2, 3, 4)
            )
            for k in range(len(after_ids))
        ]

        impression = (
            original["impression_id"].to_list()[i]
            if "impression_id" in original.columns
            else i
        )

        print()
        print(f"  --- row {i}  impression_id={impression}"
              f"  permutation={permutations[i]}")
        print(f"    before ids    : {before_ids}")
        print(f"    after  ids    : {after_ids}")
        print(f"    before sids   : {before_sids}")
        print(f"    after  sids   : {after_sids}")
        print(f"    before labels : {before_labels}"
              f"   positive index = {positive_position(before_labels)}")
        print(f"    after  labels : {after_labels}"
              f"   positive index = {positive_position(after_labels)}")
        print(f"    target        : {original['target_article_ids'].to_list()[i]}"
              f"   (?욎? ?딆쓬)")


def verify(
    original: pl.DataFrame,
    shuffled: pl.DataFrame,
    columns: list[str],
    untouched: list[str],
) -> dict[str, Any]:
    """湲곗〈 ?뚯씪怨????뚯씪??row蹂꾨줈 鍮꾧탳?쒕떎."""

    section("寃利?)

    checks: list[tuple[str, bool, str]] = []

    checks.append((
        "row count ?숈씪",
        original.height == shuffled.height,
        f"{original.height:,} -> {shuffled.height:,}",
    ))

    checks.append((
        "而щ읆 援ъ꽦 ?숈씪",
        original.columns == shuffled.columns,
        f"{len(original.columns)}媛?,
    ))

    dtype_ok = all(
        original[c].dtype == shuffled[c].dtype for c in original.columns
    )
    checks.append(("dtype ?숈씪", dtype_ok, "List(Int32/Int64) ?좎?"))

    untouched_ok = all(
        original[c].to_list() == shuffled[c].to_list() for c in untouched
    )
    checks.append((
        f"candidate ??{len(untouched)}媛?而щ읆 蹂寃??놁쓬",
        untouched_ok,
        "history_*, target_*, 硫뷀? ?ы븿",
    ))

    before_labels = original["candidate_labels"].to_list()
    after_labels = shuffled["candidate_labels"].to_list()
    before_ids = original["candidate_article_ids"].to_list()
    after_ids = shuffled["candidate_article_ids"].to_list()

    width = len(before_labels[0])

    counts_ok = all(len(v) == width for v in after_labels)
    checks.append((f"row??candidate ??= {width}", counts_ok, ""))

    positives_ok = all(sum(v) == 1 for v in after_labels)
    checks.append(("row??positive ?뺥솗??1媛?, positives_ok, ""))

    set_ok = True
    positive_ok = True
    negative_ok = True
    order_changed = 0
    dup_ok = True

    sid_before = [original[c].to_list() for c in
                  ("candidate_c1", "candidate_c2", "candidate_c3", "candidate_c4")]
    sid_after = [shuffled[c].to_list() for c in
                 ("candidate_c1", "candidate_c2", "candidate_c3", "candidate_c4")]

    for i in range(original.height):
        a_ids, b_ids = list(before_ids[i]), list(after_ids[i])
        a_lab, b_lab = list(before_labels[i]), list(after_labels[i])

        if sorted(a_ids) != sorted(b_ids):
            set_ok = False
            break

        if len(set(b_ids)) != width:
            dup_ok = False
            break

        a_pos = a_ids[positive_position(a_lab)]
        b_pos = b_ids[positive_position(b_lab)]

        if a_pos != b_pos:
            positive_ok = False
            break

        a_neg = sorted(x for x, l in zip(a_ids, a_lab) if l == 0)
        b_neg = sorted(x for x, l in zip(b_ids, b_lab) if l == 0)

        if a_neg != b_neg:
            negative_ok = False
            break

        # SID??媛숈? 湲곗궗瑜??곕씪 ?吏곸??붿?
        a_map = {
            a_ids[k]: tuple(int(sid_before[l][i][k]) for l in range(4))
            for k in range(width)
        }
        b_map = {
            b_ids[k]: tuple(int(sid_after[l][i][k]) for l in range(4))
            for k in range(width)
        }

        if a_map != b_map:
            set_ok = False
            break

        if a_ids != b_ids:
            order_changed += 1

    checks.append(("candidate set ?숈씪 (?쒖꽌 臾댁떆)", set_ok, "SID 留ㅽ븨 ?ы븿"))
    checks.append(("positive article ?숈씪", positive_ok, ""))
    checks.append(("negative 4媛?set ?숈씪", negative_ok, ""))
    checks.append(("row ??candidate 以묐났 ?놁쓬", dup_ok, ""))
    checks.append((
        "candidate order留?蹂寃?,
        order_changed > 0,
        f"{order_changed:,} / {original.height:,} row?먯꽌 ?쒖꽌媛 諛붾?,
    ))

    print()
    for name, ok, note in checks:
        print(f"  [{'OK  ' if ok else 'FAIL'}] {name}"
              + (f"   ({note})" if note else ""))

    all_ok = all(ok for _, ok, _ in checks)

    print()
    print("  => " + ("紐⑤뱺 寃利??듦낵" if all_ok else "寃利??ㅽ뙣"))

    return {
        "all_passed": all_ok,
        "order_changed_rows": order_changed,
        "checks": [
            {"name": n, "passed": bool(o), "note": note} for n, o, note in checks
        ],
    }


def process_file(
    input_path: Path,
    output_path: Path | None,
    split_name: str,
    seed: int,
    preview: int,
) -> dict[str, Any] | None:
    section(f"[{split_name}] {input_path.name}")
    print(f"  ?낅젰 : {input_path}")

    if not input_path.exists():
        print("  ?뚯씪???놁뒿?덈떎. 嫄대꼫?곷땲??")
        return None

    df = pl.read_parquet(input_path)
    print(f"  {df.height:,} rows, {len(df.columns)} columns")

    columns, untouched = check_columns(df)

    print()
    print(f"  permutation ?곸슜 ({len(columns)}媛?")
    for c in columns:
        print(f"    {c:<24} {df[c].dtype}")

    print()
    print(f"  洹몃?濡??먮뒗 而щ읆 ({len(untouched)}媛?")
    for c in untouched:
        print(f"    {c}")

    before = position_distribution(df)
    print_distribution(f"{split_name} ??shuffle ??positive_position_distribution", before)

    if before["max_deviation_from_uniform"] > 0.05:
        print("  >>> position bias ?뺤씤??)

    work = df.head(preview) if preview else df
    shuffled, permutations = shuffle_frame(work, columns, split_name, seed)

    if preview:
        print_preview(work, shuffled, permutations, preview)
        return {"preview": True}

    after = position_distribution(shuffled)
    print_distribution(f"{split_name} ??shuffle ??positive_position_distribution", after)

    if output_path is None:
        print()
        print("  異쒕젰 寃쎈줈媛 ?놁뼱 ?뚯씪???곗? ?딆뒿?덈떎.")
        return None

    if output_path.exists():
        print()
        print(f"  ?대? 議댁옱?⑸땲?? ??뼱?곗? ?딆뒿?덈떎: {output_path}")
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    shuffled.write_parquet(output_path)

    # 湲곗〈 ?뚯씪怨????뚯씪???ㅼ떆 ?쎌뼱 鍮꾧탳?쒕떎
    written = pl.read_parquet(output_path)
    verification = verify(df, written, columns, untouched)

    if not verification["all_passed"]:
        print()
        print(f"  寃利앹뿉 ?ㅽ뙣?덉뒿?덈떎. ?앹꽦???뚯씪???뺤씤?섏꽭?? {output_path}")

    print()
    print(f"  異쒕젰 : {output_path}")

    return {
        "split_name": split_name,
        "input_path": str(input_path),
        "output_path": str(output_path),
        "num_rows": int(df.height),
        "seed": seed,
        "columns_shuffled": columns,
        "columns_untouched": untouched,
        "positive_position_distribution_before": before,
        "positive_position_distribution_after": after,
        "verification": verification,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "湲곗〈 1pos4neg parquet??candidate ?쒖꽌留??욌뒗?? "
            "?ъ깦?뚮쭅?섏? ?딅뒗??"
        )
    )
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--file", action="append", default=None,
        metavar="SPLIT:PATH",
        help="媛쒕퀎 ?뚯씪 吏?? ??test:path/to/test.parquet",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--preview", type=int, default=0,
        help="N?됰쭔 誘몃━ 蹂닿퀬 ?뚯씪???곗? ?딅뒗??",
    )
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    targets: list[tuple[str, Path, Path | None]] = []

    if args.input_dir:
        for split_name, filename in DEFAULT_FILES:
            out = (args.output_dir / filename) if args.output_dir else None
            targets.append((split_name, args.input_dir / filename, out))

    # --file? --input-dir怨??④퍡 ?????덈떎. test瑜?異붽??????대떎.
    for item in args.file or []:
        split_name, _, raw_path = item.partition(":")
        path = Path(raw_path)
        out = (args.output_dir / path.name) if args.output_dir else None
        targets.append((split_name, path, out))

    if not targets:
        parser.error("--input-dir ?먮뒗 --file 以??섎굹媛 ?꾩슂?⑸땲??")

    section("candidate order shuffle")
    print(f"  seed : {args.seed}")
    print()
    print("  諛붽씀??寃?  : candidate 愿??而щ읆???먯냼 ?쒖꽌")
    print("  ??諛붽씀??寃? positive ?좏깮, negative ?섑뵆留? candidate set,")
    print("                history, target, SID 媛? split, ???? ???쒖꽌")

    results = []

    for split_name, input_path, output_path in targets:
        result = process_file(
            input_path=input_path,
            output_path=output_path,
            split_name=split_name,
            seed=args.seed,
            preview=args.preview,
        )

        if result and not result.get("preview"):
            results.append(result)

    if args.preview:
        section("誘몃━蹂닿린留??덉뒿?덈떎. ?뚯씪???곗? ?딆븯?듬땲??")
        print("  ?뺤씤 ??--preview ?놁씠 --output-dir??二쇨퀬 ?ㅼ떆 ?ㅽ뻾?섏꽭??")
        return 0

    if args.report and results:
        report = {
            "tool": "shuffle_candidate_order",
            "candidate_order_shuffled": True,
            "resampled": False,
            "seed": args.seed,
            "row_seed_formula": (
                "sha256(seed | split_name | impression_id | "
                "positive_article_id | row_uid)"
            ),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "splits": results,
        }

        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print()
        print(f"由ы룷??: {args.report}")

    return 0 if all(
        r["verification"]["all_passed"] for r in results
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
