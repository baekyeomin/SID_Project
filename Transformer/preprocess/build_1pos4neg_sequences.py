import argparse
import json

from pathlib import Path
from typing import Any

import polars as pl

from src import config

from src.analyze_candidate_sid_collision import (
    PREFIX_LEVELS,
    PRIMARY_LEVEL,
    resolve_semantic_ids_path,
    _load_sid_lookup,
    _safe_ratio,
)


# ============================================================
# STEP 15. 1 positive + N negative sequences ?앹꽦
#
# 紐⑹쟻:
# 湲곗〈 sequences parquet???쎌뼱
# "impression??positive 1媛?+ negative N媛? N+1媛쒓? 紐⑤몢 ?쒕줈 ?ㅻⅨ SID"
# ?뺥깭???숈뒿 ?곗씠?곕? 留뚮뱺??
#
# ?뺤젙???뺤콉 (2026-09-18 援먯닔???뚯떊):
#   1. positive媛 2媛??댁긽??impression? positive留덈떎 蹂꾨룄 row濡?遺꾨━?쒕떎.
#      遺꾨━??row?ㅼ씠 ?쒕줈 ?ㅻⅨ negative瑜?媛뽯룄濡?媛뺤젣?섏? ?딅뒗??
#   2. 議곌굔??紐?梨꾩슦??row???쒖쇅?쒕떎.
#   3. negative 鍮꾩쑉? 1:4濡?怨좎젙?쒕떎.
#
# 泥섎━ ?쒖꽌 (positive ?섎굹??:
#   negative ?꾩껜 (label == 0)
#     -> positive? (c1,c2,c3)媛 媛숈? negative ?쒓굅
#     -> negative?쇰━ (c1,c2,c3) 以묐났 ?쒓굅 (洹몃９??1媛쒕쭔 ?④?)
#     -> N媛?誘몃쭔?대㈃ drop
#     -> ?⑥? 寃?以?seed 湲곕컲?쇰줈 N媛?異붿텧
#     -> [P, N1, ..., Nn]
#
# 以묒슂:
# negative ?꾨낫??諛섎뱶??label == 0??寃껊쭔 ?대떎.
# 媛숈? impression???ㅻⅨ positive???ㅼ젣濡??대┃??湲곗궗?대?濡?# negative濡??ｌ쑝硫????쒕떎.
#
# 異쒕젰:
#   湲곗〈 sequences? ?숈씪??而щ읆 援ъ꽦?대ŉ
#   target_* ??洹?row??positive ?섎굹留??대뒗??
# ============================================================


DEFAULT_NEGATIVE_COUNT = 4
DEFAULT_SEED = 42
DEFAULT_CHUNK_SIZE = 20_000


HISTORY_COLUMNS = [
    "impression_id",
    "user_id",
    "impression_time",
    "history_article_ids",
    "history_c1",
    "history_c2",
    "history_c3",
    "history_c4",
]


CANDIDATE_COLUMNS = [
    "candidate_article_ids",
    "candidate_c1",
    "candidate_c2",
    "candidate_c3",
    "candidate_c4",
    "candidate_labels",
]


SID_SUFFIXES = ["c1", "c2", "c3", "c4"]


def _build_chunk(
    chunk_df: pl.DataFrame,
    sid_columns: list[str],
    negative_count: int,
    seed: int,
    offset: int,
) -> tuple[pl.DataFrame, dict[str, int]]:
    """
    chunk ?섎굹瑜?1 positive + N negative ?뺥깭濡?諛붽씔??

    諛섑솚: (寃곌낵 DataFrame, ?듦퀎 dict)
    """

    # STEP 15-1. candidate瑜?long format?쇰줈 ?쇱튇??    long_df = (
        chunk_df
        .select(CANDIDATE_COLUMNS)
        .with_row_index("row_index", offset=offset)
        .explode(CANDIDATE_COLUMNS, empty_as_null=False)
    )

    # STEP 15-2. positive / negative 遺꾨━
    # negative??諛섎뱶??label == 0??寃껊쭔 ?대떎.
    positive_df = (
        long_df
        .filter(pl.col("candidate_labels") == 1)
        .drop("candidate_labels")
        .with_row_index("sample_id")
    )

    negative_df = (
        long_df
        .filter(pl.col("candidate_labels") == 0)
        .drop("candidate_labels")
    )

    positive_total = positive_df.height

    # STEP 15-3. (positive, negative) ?띿쓣 留뚮뱺??    pair_df = positive_df.join(
        negative_df,
        on="row_index",
        how="inner",
        suffix="_neg",
    )

    # STEP 15-4. positive? SID媛 媛숈? negative ?쒓굅
    same_sid_expression = pl.lit(True)

    for column_name in sid_columns:
        same_sid_expression = same_sid_expression & (
            pl.col(column_name) == pl.col(f"{column_name}_neg")
        )

    pair_df = pair_df.filter(~same_sid_expression)

    # STEP 15-5. ?ы쁽 媛?ν븳 ?쒖닔 遺??    # (?꾩뿭 row_index, positive article_id, negative article_id)???댁떆瑜?    # ?곗꽑?쒖쐞濡??대떎. chunk ?ш린瑜?諛붽퓭??媛숈? 媛믪씠 ?섏삩??
    pair_df = pair_df.with_columns(
        pl.struct([
            "row_index",
            "candidate_article_ids",
            "candidate_article_ids_neg",
        ])
        .hash(seed=seed)
        .alias("priority")
    )

    # STEP 15-6. negative?쇰━ SID 以묐났 ?쒓굅
    # 媛숈? SID 洹몃９?먯꽌??priority媛 媛???묒? ?섎굹留??④릿??
    negative_sid_columns = [
        f"{column_name}_neg" for column_name in sid_columns
    ]

    pair_df = (
        pair_df
        .sort("priority")
        .unique(
            subset=["sample_id"] + negative_sid_columns,
            keep="first",
            maintain_order=True,
        )
    )

    # STEP 15-7. negative媛 N媛?誘몃쭔??sample ?쒖쇅
    usable_df = (
        pair_df
        .group_by("sample_id")
        .len()
        .filter(pl.col("len") >= negative_count)
        .select("sample_id")
    )

    feasible_sample_count = usable_df.height

    pair_df = pair_df.join(usable_df, on="sample_id", how="semi")

    # STEP 15-8. sample留덈떎 N媛?異붿텧
    picked_df = (
        pair_df
        .sort(["sample_id", "priority"])
        .group_by("sample_id", maintain_order=True)
        .head(negative_count)
    )

    # STEP 15-9. negative瑜?list濡?臾띕뒗??    negative_list_df = (
        picked_df
        .group_by("sample_id", maintain_order=True)
        .agg([
            pl.col("candidate_article_ids_neg").alias("neg_article_ids"),
            *[
                pl.col(f"candidate_{suffix}_neg").alias(f"neg_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
    )

    # STEP 15-10. positive瑜??욎뿉 遺숈뿬 理쒖쥌 candidate list瑜?留뚮뱺??    kept_positive_df = positive_df.join(
        usable_df,
        on="sample_id",
        how="semi",
    )

    sample_df = (
        kept_positive_df
        .join(negative_list_df, on="sample_id", how="inner")
        .with_columns([
            pl.concat_list([
                pl.col("candidate_article_ids"),
                pl.col("neg_article_ids"),
            ]).alias("candidate_article_ids"),

            *[
                pl.concat_list([
                    pl.col(f"candidate_{suffix}"),
                    pl.col(f"neg_{suffix}"),
                ]).alias(f"candidate_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
        .with_columns(
            pl.concat_list([
                pl.lit(1, dtype=pl.Int32),
                *[pl.lit(0, dtype=pl.Int32)] * negative_count,
            ]).alias("candidate_labels")
        )
    )

    # STEP 15-11. history 而щ읆???ㅼ떆 遺숈씠怨?target??洹?positive ?섎굹濡??붾떎
    history_df = (
        chunk_df
        .select(HISTORY_COLUMNS)
        .with_row_index("row_index", offset=offset)
    )

    result_df = (
        sample_df
        .join(history_df, on="row_index", how="inner")
        .with_columns([
            pl.concat_list([
                pl.col("candidate_article_ids").list.first()
            ]).alias("target_article_ids"),

            *[
                pl.concat_list([
                    pl.col(f"candidate_{suffix}").list.first()
                ]).alias(f"target_{suffix}")
                for suffix in SID_SUFFIXES
            ],
        ])
        .select(HISTORY_COLUMNS + [
            "target_article_ids",
            *[f"target_{suffix}" for suffix in SID_SUFFIXES],
            *CANDIDATE_COLUMNS,
        ])
    )

    stats = {
        "impression_count": chunk_df.height,
        "positive_count": positive_total,
        "feasible_sample_count": feasible_sample_count,
        "output_row_count": result_df.height,
    }

    return result_df, stats


def build_1pos_n_neg(
    sequences_path: Path,
    output_path: Path,
    split_name: str,
    negative_count: int = DEFAULT_NEGATIVE_COUNT,
    seed: int = DEFAULT_SEED,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    limit: int | None = None,
    semantic_ids_path: Path | None = None,
    level_name: str = PRIMARY_LEVEL,
) -> dict[str, Any]:
    """
    sequences parquet ?섎굹瑜?1 positive + N negative ?뺥깭濡?蹂?섑빐 ??ν븳??
    """

    sid_columns = dict(PREFIX_LEVELS).get(level_name)

    if sid_columns is None:
        raise ValueError(f"?????녿뒗 level: {level_name}")

    use_external_sid = semantic_ids_path is not None

    if use_external_sid:
        semantic_ids_path = resolve_semantic_ids_path(semantic_ids_path)
        sid_lookup_df = _load_sid_lookup(semantic_ids_path)

    lazy_frame = pl.scan_parquet(sequences_path)

    schema_names = lazy_frame.collect_schema().names()

    missing_columns = [
        column_name
        for column_name in HISTORY_COLUMNS + CANDIDATE_COLUMNS
        if column_name not in schema_names
    ]

    if missing_columns:
        raise ValueError(
            "?꾩슂??而щ읆???놁뒿?덈떎: " + ", ".join(missing_columns)
        )

    total_row_count = lazy_frame.select(pl.len()).collect().item()

    if limit is not None:
        total_row_count = min(total_row_count, limit)

    totals = {
        "impression_count": 0,
        "positive_count": 0,
        "feasible_sample_count": 0,
        "output_row_count": 0,
    }

    output_frames = []

    for offset in range(0, total_row_count, chunk_size):
        current_chunk_size = min(chunk_size, total_row_count - offset)

        chunk_df = (
            lazy_frame
            .slice(offset, current_chunk_size)
            .select(HISTORY_COLUMNS + CANDIDATE_COLUMNS)
            .collect()
        )

        # ?몃? SID瑜??곕뒗 寃쎌슦 candidate SID瑜?援먯껜?쒕떎.
        if use_external_sid:
            chunk_df = _replace_candidate_sid(chunk_df, sid_lookup_df)

        result_df, stats = _build_chunk(
            chunk_df,
            sid_columns,
            negative_count,
            seed,
            offset,
        )

        output_frames.append(result_df)

        for key in totals:
            totals[key] += stats[key]

    output_df = pl.concat(output_frames, how="vertical")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_df.write_parquet(output_path)

    dropped = totals["positive_count"] - totals["output_row_count"]

    return {
        "split_name": split_name,
        "sequences_path": str(sequences_path),
        "output_path": str(output_path),
        "semantic_ids_path": (
            str(semantic_ids_path) if use_external_sid else None
        ),
        "level": level_name,
        "negative_count": negative_count,
        "seed": seed,
        "impression_count": totals["impression_count"],
        "positive_count": totals["positive_count"],
        "output_row_count": totals["output_row_count"],
        "dropped_positive_count": dropped,
        "dropped_positive_ratio": _safe_ratio(
            dropped,
            totals["positive_count"],
        ),
        "candidate_position_count": (
            totals["output_row_count"] * (negative_count + 1)
        ),
    }


def _replace_candidate_sid(
    chunk_df: pl.DataFrame,
    sid_lookup_df: pl.DataFrame,
) -> pl.DataFrame:
    """
    candidate SID瑜??몃? article_semantic_ids.parquet 媛믪쑝濡?援먯껜?쒕떎.
    """

    long_df = (
        chunk_df
        .select(["candidate_article_ids", "candidate_labels"])
        .with_row_index("row_index")
        .explode(
            ["candidate_article_ids", "candidate_labels"],
            empty_as_null=False,
        )
        .with_columns(pl.col("candidate_article_ids").cast(pl.Int64))
        .join(sid_lookup_df, on="candidate_article_ids", how="left")
        .filter(pl.col("candidate_c1").is_not_null())
        .group_by("row_index", maintain_order=True)
        .agg([
            pl.col("candidate_article_ids"),
            *[pl.col(f"candidate_{suffix}") for suffix in SID_SUFFIXES],
            pl.col("candidate_labels"),
        ])
    )

    return (
        chunk_df
        .drop(CANDIDATE_COLUMNS)
        .with_row_index("row_index")
        .join(long_df, on="row_index", how="inner")
        .drop("row_index")
    )


def _print_result(result: dict[str, Any]) -> None:
    n = result["negative_count"]

    print()
    print("=" * 70)
    print(f"[{result['split_name']}] 1 positive + {n} negative ?앹꽦")
    print("=" * 70)

    print(f"?낅젰                 : {result['sequences_path']}")
    print(f"異쒕젰                 : {result['output_path']}")
    print(f"seed                 : {result['seed']}")
    print(f"impression ??       : {result['impression_count']:,}")
    print(f"positive ??         : {result['positive_count']:,}")
    print(f"?앹꽦???숈뒿 row      : {result['output_row_count']:,}")
    print(
        "?덈씫 positive        : "
        f"{result['dropped_positive_count']:,} "
        f"({result['dropped_positive_ratio']:.4%})"
    )
    print(
        "candidate position   : "
        f"{result['candidate_position_count']:,} "
        f"(= row 횞 {n + 1})"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "sequences parquet??1 positive + N negative ?뺥깭濡?蹂?섑븳?? "
            "N+1媛쒓? 紐⑤몢 ?쒕줈 ?ㅻⅨ SID瑜?媛뽯뒗??"
        )
    )

    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="異쒕젰 ?대뜑. train/validation parquet?????덉뿉 ?앹꽦?쒕떎.",
    )

    parser.add_argument("--path", type=Path, action="append", default=None)
    parser.add_argument("--semantic-ids", type=Path, default=None)
    parser.add_argument(
        "--negatives", type=int, default=DEFAULT_NEGATIVE_COUNT
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--level", type=str, default=PRIMARY_LEVEL)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--report", type=Path, default=None)

    args = parser.parse_args()

    if args.path:
        targets = [(path.stem, path) for path in args.path]
    else:
        targets = [
            ("train_sequences", config.TRAIN_SEQUENCES_PATH),
            ("validation_sequences", config.VALIDATION_SEQUENCES_PATH),
        ]

    results = []

    for split_name, sequences_path in targets:
        output_path = (
            args.out_dir
            / f"{split_name}_1pos{args.negatives}neg.parquet"
        )

        result = build_1pos_n_neg(
            sequences_path,
            output_path,
            split_name,
            negative_count=args.negatives,
            seed=args.seed,
            chunk_size=args.chunk_size,
            limit=args.limit,
            semantic_ids_path=args.semantic_ids,
            level_name=args.level,
        )

        _print_result(result)
        results.append(result)

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"由ы룷?????: {args.report}")


if __name__ == "__main__":
    main()
