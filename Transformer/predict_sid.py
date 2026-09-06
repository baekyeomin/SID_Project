from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import gin
import pandas as pd
import torch

from torch import Tensor
from torch.utils.data import DataLoader

from data.sequence import (
    NewsSequenceDataset,
    collate_news_sequences,
)

from modules.model import NewsEncoderDecoderTransformer


BASE_DIR = Path(__file__).resolve().parent

SID = Tuple[int, int, int, int]


def resolve_path(path: str) -> Path:
    path_obj = Path(path)

    if path_obj.is_absolute():
        return path_obj

    return BASE_DIR / path_obj


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def normalize_article_id(article_id):
    if article_id is None:
        return None

    try:
        if pd.isna(article_id):
            return None
    except (TypeError, ValueError):
        pass

    return str(article_id)


def load_sid_catalog(
    article_sid_path: Path,
) -> Tuple[
    Dict[SID, str],
    Dict[Tuple[int, ...], List[SID]],
    List[SID],
]:

    if not article_sid_path.exists():
        raise FileNotFoundError(
            f"Article SID file not found:\n"
            f"{article_sid_path}"
        )

    df = pd.read_parquet(
        article_sid_path
    )

    required_columns = [
        "c1",
        "c2",
        "c3",
        "c4",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing SID columns: {missing_columns}"
        )

    if "article_id" in df.columns:
        article_id_column = "article_id"

    elif "news_id" in df.columns:
        article_id_column = "news_id"

    else:
        raise ValueError(
            "article_semantic_ids.parquet에 "
            "article_id 또는 news_id column이 없습니다."
        )

    duplicate_counts = (
        df.groupby(
            [
                "c1",
                "c2",
                "c3",
                "c4",
            ]
        )
        .size()
    )

    duplicate_sid_count = int(
        (
            duplicate_counts > 1
        ).sum()
    )

    if duplicate_sid_count > 0:
        raise ValueError(
            f"c4까지 포함했는데도 동일한 full SID를 가진 기사가 "
            f"{duplicate_sid_count}개 SID 그룹에서 발견되었습니다."
        )

    df = (
        df.sort_values(
            [
                "c1",
                "c2",
                "c3",
                "c4",
            ]
        )
        .reset_index(drop=True)
    )

    sid_to_article: Dict[
        SID,
        str,
    ] = {}

    prefix_to_sids: Dict[
        Tuple[int, ...],
        List[SID],
    ] = {}

    all_sids: List[SID] = []

    for _, row in df.iterrows():

        sid = (
            int(row["c1"]),
            int(row["c2"]),
            int(row["c3"]),
            int(row["c4"]),
        )

        article_id = normalize_article_id(
            row[article_id_column]
        )

        sid_to_article[sid] = (
            article_id
        )

        all_sids.append(
            sid
        )

        for prefix_length in [
            1,
            2,
            3,
        ]:

            prefix = sid[
                :prefix_length
            ]

            if prefix not in prefix_to_sids:
                prefix_to_sids[
                    prefix
                ] = []

            prefix_to_sids[
                prefix
            ].append(
                sid
            )

    print()
    print("SID Catalog")
    print(
        "Articles:",
        f"{len(df):,}",
    )
    print(
        "Unique full SIDs:",
        f"{len(sid_to_article):,}",
    )
    print(
        "Duplicated full SIDs:",
        duplicate_sid_count,
    )

    return (
        sid_to_article,
        prefix_to_sids,
        all_sids,
    )


def validate_catalog_against_model(
    all_sids: List[SID],
    model: NewsEncoderDecoderTransformer,
) -> None:

    if len(all_sids) == 0:
        raise ValueError(
            "SID catalog is empty."
        )

    vocab_sizes = [
        model.c1_vocab_size,
        model.c2_vocab_size,
        model.c3_vocab_size,
        model.c4_vocab_size,
    ]

    for level in range(4):

        codes = [
            sid[level]
            for sid in all_sids
        ]

        min_code = min(
            codes
        )

        max_code = max(
            codes
        )

        if min_code < 0:
            raise ValueError(
                f"Negative SID found "
                f"at level c{level + 1}."
            )

        if (
            max_code
            >= vocab_sizes[level]
        ):
            raise ValueError(
                f"SID catalog contains "
                f"c{level + 1}={max_code}, "
                f"but model vocab size is "
                f"{vocab_sizes[level]}."
            )


def find_nearest_existing_sid(
    predicted_sid: SID,
    sid_to_article: Dict[SID, str],
    prefix_to_sids: Dict[
        Tuple[int, ...],
        List[SID],
    ],
    all_sids: List[SID],
) -> Tuple[
    SID,
    int,
]:

    # distance = 4 - longest common prefix length

    if predicted_sid in sid_to_article:
        return (
            predicted_sid,
            0,
        )

    prefix3 = (
        predicted_sid[:3]
    )

    if prefix3 in prefix_to_sids:
        return (
            prefix_to_sids[
                prefix3
            ][0],
            1,
        )

    prefix2 = (
        predicted_sid[:2]
    )

    if prefix2 in prefix_to_sids:
        return (
            prefix_to_sids[
                prefix2
            ][0],
            2,
        )

    prefix1 = (
        predicted_sid[:1]
    )

    if prefix1 in prefix_to_sids:
        return (
            prefix_to_sids[
                prefix1
            ][0],
            3,
        )

    return (
        all_sids[0],
        4,
    )


@torch.no_grad()
def unconstrained_beam_search_single(
    model: NewsEncoderDecoderTransformer,
    encoder_hidden_states: Tensor,
    encoder_attention_mask: Tensor,
    beam_size: int = 20,
    top_k: int = 10,
) -> List[
    Tuple[
        SID,
        float,
    ]
]:

    if beam_size <= 0:
        raise ValueError(
            "beam_size must be > 0."
        )

    if top_k <= 0:
        raise ValueError(
            "top_k must be > 0."
        )

    keep_size = max(
        beam_size,
        top_k,
    )

    beams: List[
        Tuple[
            Tuple[int, ...],
            float,
        ]
    ] = [
        (
            (),
            0.0,
        )
    ]

    device = (
        encoder_hidden_states.device
    )

    # catalog와 관계없이 c1 → c2 → c3 → c4 생성
    for level in range(4):

        num_beams = len(
            beams
        )

        repeated_encoder_hidden = (
            encoder_hidden_states.expand(
                num_beams,
                -1,
                -1,
            )
        )

        repeated_encoder_mask = (
            encoder_attention_mask.expand(
                num_beams,
                -1,
            )
        )

        if level == 0:

            prefix_tensor = None

        else:

            prefix_tensor = (
                torch.tensor(
                    [
                        list(prefix)
                        for prefix, _
                        in beams
                    ],
                    dtype=torch.long,
                    device=device,
                )
            )

        logits = (
            model.next_token_logits(
                encoder_hidden_states=
                    repeated_encoder_hidden,

                encoder_attention_mask=
                    repeated_encoder_mask,

                prefix_sids=
                    prefix_tensor,
            )
        )

        log_probs = (
            torch.log_softmax(
                logits,
                dim=-1,
            )
        )

        candidates: List[
            Tuple[
                Tuple[int, ...],
                float,
            ]
        ] = []

        vocab_size = (
            log_probs.shape[-1]
        )

        local_k = min(
            keep_size,
            vocab_size,
        )

        for beam_idx, (
            prefix,
            previous_score,
        ) in enumerate(
            beams
        ):

            top_values, top_codes = (
                torch.topk(
                    log_probs[
                        beam_idx
                    ],
                    k=local_k,
                )
            )

            for j in range(
                local_k
            ):

                next_code = int(
                    top_codes[j].item()
                )

                next_score = (
                    previous_score
                    + float(
                        top_values[
                            j
                        ].item()
                    )
                )

                next_prefix = (
                    prefix
                    + (next_code,)
                )

                candidates.append(
                    (
                        next_prefix,
                        next_score,
                    )
                )

        candidates.sort(
            key=lambda x: x[1],
            reverse=True,
        )

        beams = (
            candidates[
                :keep_size
            ]
        )

    results = []

    for sid, score in beams:

        if len(sid) != 4:
            continue

        results.append(
            (
                (
                    int(sid[0]),
                    int(sid[1]),
                    int(sid[2]),
                    int(sid[3]),
                ),
                score,
            )
        )

    return results


def load_checkpoint(
    checkpoint_path: Path,
    model: NewsEncoderDecoderTransformer,
    device: torch.device,
) -> dict:

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "model_state_dict"
        in checkpoint
    ):

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    return checkpoint


@torch.no_grad()
def predict(
    model: NewsEncoderDecoderTransformer,
    dataloader: DataLoader,

    sid_to_article: Dict[
        SID,
        str,
    ],

    prefix_to_sids: Dict[
        Tuple[int, ...],
        List[SID],
    ],

    all_sids: List[SID],

    device: torch.device,
    beam_size: int,
    top_k: int,
) -> pd.DataFrame:

    model.eval()

    prediction_rows = []

    sample_index = 0

    for batch_idx, batch in enumerate(
        dataloader
    ):

        history_sids = (
            batch[
                "history_sids"
            ].to(device)
        )

        history_mask = (
            batch[
                "history_mask"
            ].to(device)
        )

        # target은 SID 생성에는 사용하지 않고 평가용으로만 저장
        target_sids = (
            batch[
                "target_sids"
            ]
        )

        encoder_output = (
            model.encode(
                history_sids=
                    history_sids,

                history_mask=
                    history_mask,
            )
        )

        batch_size = (
            history_sids.shape[0]
        )

        for i in range(
            batch_size
        ):

            sample_encoder_hidden = (
                encoder_output
                .hidden_states[
                    i:i + 1
                ]
            )

            sample_encoder_mask = (
                encoder_output
                .attention_mask[
                    i:i + 1
                ]
            )

            generated_candidates = (
                unconstrained_beam_search_single(
                    model=model,

                    encoder_hidden_states=
                        sample_encoder_hidden,

                    encoder_attention_mask=
                        sample_encoder_mask,

                    beam_size=
                        beam_size,

                    top_k=
                        top_k,
                )
            )

            target_sid_tensor = (
                target_sids[i]
            )

            target_sid = (
                int(
                    target_sid_tensor[0]
                ),
                int(
                    target_sid_tensor[1]
                ),
                int(
                    target_sid_tensor[2]
                ),
                int(
                    target_sid_tensor[3]
                ),
            )

            target_article_id = (
                normalize_article_id(
                    batch[
                        "target_article_ids"
                    ][i]
                )
            )

            impression_id = (
                batch[
                    "impression_ids"
                ][i]
            )

            user_id = (
                batch[
                    "user_ids"
                ][i]
            )

            impression_time = (
                batch[
                    "impression_times"
                ][i]
            )

            used_article_ids = set()

            rank = 1

            for (
                generated_sid,
                log_score,
            ) in generated_candidates:

                (
                    matched_sid,
                    sid_distance,
                ) = (
                    find_nearest_existing_sid(
                        predicted_sid=
                            generated_sid,

                        sid_to_article=
                            sid_to_article,

                        prefix_to_sids=
                            prefix_to_sids,

                        all_sids=
                            all_sids,
                    )
                )

                predicted_article_id = (
                    sid_to_article[
                        matched_sid
                    ]
                )

                # 여러 generated SID가 같은 기사로 매핑되면 중복 추천 제거
                if (
                    predicted_article_id
                    in used_article_ids
                ):
                    continue

                used_article_ids.add(
                    predicted_article_id
                )

                prediction_rows.append(
                    {
                        "sample_index":
                            sample_index,

                        "batch_index":
                            batch_idx,

                        "impression_id":
                            impression_id,

                        "user_id":
                            user_id,

                        "impression_time":
                            impression_time,

                        "target_article_id":
                            target_article_id,

                        "target_c1":
                            target_sid[0],

                        "target_c2":
                            target_sid[1],

                        "target_c3":
                            target_sid[2],

                        "target_c4":
                            target_sid[3],

                        "rank":
                            rank,

                        # Transformer가 실제 생성한 SID
                        "pred_c1":
                            generated_sid[0],

                        "pred_c2":
                            generated_sid[1],

                        "pred_c3":
                            generated_sid[2],

                        "pred_c4":
                            generated_sid[3],

                        # 실제 기사 catalog에서 찾은 가장 가까운 SID
                        "matched_c1":
                            matched_sid[0],

                        "matched_c2":
                            matched_sid[1],

                        "matched_c3":
                            matched_sid[2],

                        "matched_c4":
                            matched_sid[3],

                        "pred_article_id":
                            predicted_article_id,

                        "sid_distance":
                            sid_distance,

                        "exact_sid_match":
                            sid_distance == 0,

                        "log_score":
                            log_score,
                    }
                )

                rank += 1

                if rank > top_k:
                    break

            sample_index += 1

        print(
            f"Processed batch "
            f"{batch_idx + 1}/"
            f"{len(dataloader)}"
        )

    return pd.DataFrame(
        prediction_rows
    )


def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Predict next-news Semantic IDs "
            "and map them to nearest existing article SID."
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "checkpoint_best.pt"
        ),
    )

    parser.add_argument(
        "--test_path",
        type=str,
        default=(
            "datasets/ebnerd/"
            "test_sequences.parquet"
        ),
    )

    parser.add_argument(
        "--article_sid_path",
        type=str,
        default=(
            "datasets/ebnerd/"
            "article_semantic_ids.parquet"
        ),
    )

    parser.add_argument(
        "--output_path",
        type=str,
        default=(
            "out/transformer/ebnerd/"
            "test_predictions.parquet"
        ),
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--beam_size",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    args = parser.parse_args()

    config_path = resolve_path(
        args.config
    )

    checkpoint_path = resolve_path(
        args.checkpoint
    )

    test_path = resolve_path(
        args.test_path
    )

    article_sid_path = resolve_path(
        args.article_sid_path
    )

    output_path = resolve_path(
        args.output_path
    )

    if not config_path.exists():
        raise FileNotFoundError(
            f"Gin config not found:\n"
            f"{config_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found:\n"
            f"{checkpoint_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file not found:\n"
            f"{test_path}"
        )

    if not article_sid_path.exists():
        raise FileNotFoundError(
            f"Article SID file not found:\n"
            f"{article_sid_path}"
        )

    gin.parse_config_file(
        str(config_path),
        skip_unknown=True,
    )

    device = get_device()

    print()
    print(
        "Transformer SID Prediction"
    )

    print(
        "Device:",
        device,
    )

    print(
        "Test path:",
        test_path,
    )

    print(
        "Checkpoint:",
        checkpoint_path,
    )

    print(
        "Article SID:",
        article_sid_path,
    )

    print(
        "Beam size:",
        args.beam_size,
    )

    print(
        "Top-K:",
        args.top_k,
    )

    test_dataset = (
        NewsSequenceDataset(
            parquet_path=str(
                test_path
            ),
        )
    )

    test_loader = DataLoader(
        dataset=
            test_dataset,

        batch_size=
            args.batch_size,

        shuffle=False,

        num_workers=
            args.num_workers,

        collate_fn=
            collate_news_sequences,

        pin_memory=(
            device.type
            == "cuda"
        ),
    )

    print(
        "Test samples:",
        f"{len(test_dataset):,}",
    )

    model = (
        NewsEncoderDecoderTransformer()
        .to(device)
    )

    checkpoint = load_checkpoint(
        checkpoint_path=
            checkpoint_path,

        model=
            model,

        device=
            device,
    )

    print(
        "Checkpoint loaded."
    )

    if (
        isinstance(
            checkpoint,
            dict,
        )
        and
        "epoch"
        in checkpoint
    ):

        print(
            "Best epoch:",
            checkpoint[
                "epoch"
            ],
        )

    (
        sid_to_article,
        prefix_to_sids,
        all_sids,
    ) = load_sid_catalog(
        article_sid_path
    )

    validate_catalog_against_model(
        all_sids=
            all_sids,

        model=
            model,
    )

    predictions_df = predict(
        model=
            model,

        dataloader=
            test_loader,

        sid_to_article=
            sid_to_article,

        prefix_to_sids=
            prefix_to_sids,

        all_sids=
            all_sids,

        device=
            device,

        beam_size=
            args.beam_size,

        top_k=
            args.top_k,
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    predictions_df.to_parquet(
        output_path,
        index=False,
    )

    print()
    print(
        "Predictions saved:",
        output_path,
    )


if __name__ == "__main__":
    main()