
# 수정사항 (2026-11-23)
# - H 버전의 정규화된 canonical_entity_key 및 is_high_df 제외 로직 재사용
# - Train 기준 IDF 및 unseen_entity_idf 사용을 위한 파일 경로 인자 추가
# - 기존 R1 기반 C2 결정, 코사인 Fallback, TimeUpdate 로직 유지

import argparse

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

import generate_semantic_ids_timeupdate_hver as hver


@torch.inference_mode()
def compute_event_c2_from_mean_r1(
    model,
    dataset,
    device,
    allowed_event_ids=None,
    batch_size=512,
    num_workers=0,
    source="train_mean_r1_frozen_q2",
    return_representations=False,
):
    hver.validate_dataset_against_model(dataset, model)

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    event_sums = {}
    event_counts = {}

    print(
        "\n[R1] Building Train event representations: mean(r1(a))"
    )

    for batch_index, batch in enumerate(loader):
        event_ids = [
            int(v) for v in batch["event_id"].cpu().tolist()
        ]
        positions = [
            i
            for i, event_id in enumerate(event_ids)
            if allowed_event_ids is None
            or event_id in allowed_event_ids
        ]

        if not positions:
            continue

        selection = torch.tensor(
            positions, dtype=torch.long
        )

        x = batch["x"][selection].to(
            device=device,
            dtype=next(model.encoder.parameters()).dtype,
            non_blocking=True,
        )
        category_ids = batch["category_id"][selection].to(
            device=device,
            dtype=torch.long,
            non_blocking=True,
        )

        r1 = model.get_c2_residual(
            x=x,
            category_ids=category_ids,
        )

        for local_index, position in enumerate(positions):
            event_id = event_ids[position]
            vector = r1[local_index].detach().float().cpu()

            if event_id not in event_sums:
                event_sums[event_id] = vector.clone()
                event_counts[event_id] = 1
            else:
                event_sums[event_id].add_(vector)
                event_counts[event_id] += 1

        if (batch_index + 1) % 20 == 0:
            processed = min(
                (batch_index + 1) * batch_size,
                len(dataset),
            )
            print(f"  {processed}/{len(dataset)}")

    if not event_sums:
        empty = pd.DataFrame(
            columns=[
                "event_id",
                "event_c2",
                "num_articles",
                "source",
            ]
        )
        return (
            (empty, {})
            if return_representations
            else empty
        )

    event_ids = sorted(event_sums)
    mapping_rows = []
    representations = {}
    dtype = next(model.encoder.parameters()).dtype

    for start in range(0, len(event_ids), batch_size):
        selected_ids = event_ids[
            start:start + batch_size
        ]
        z_events = torch.stack([
            event_sums[event_id]
            / float(event_counts[event_id])
            for event_id in selected_ids
        ]).to(
            device=device,
            dtype=dtype,
        )

        if return_representations:
            for event_id, vector in zip(
                selected_ids, z_events
            ):
                representations[event_id] = (
                    vector.detach()
                    .float()
                    .cpu()
                    .numpy()
                    .copy()
                )

        c2_ids = model.quantizer_2(
            x=z_events,
            temperature=1.0,
        ).ids.detach().cpu().tolist()

        for event_id, c2 in zip(
            selected_ids, c2_ids
        ):
            mapping_rows.append({
                "event_id": int(event_id),
                "event_c2": int(c2),
                "num_articles": int(
                    event_counts[event_id]
                ),
                "source": (
                    "train_mean_r1_frozen_q2"
                    if source == "train_mean_h_frozen_q2"
                    else source
                ),
            })

    result = pd.DataFrame(mapping_rows)

    if result["event_id"].duplicated().any():
        raise RuntimeError(
            "Duplicate Train event IDs in R1 C2 mapping"
        )

    return (
        (result, representations)
        if return_representations
        else result
    )


@torch.inference_mode()
def encode_selected_articles_r1(
    model,
    dataset,
    positions,
    device,
    batch_size=512,
):
    vectors = []
    dtype = next(model.encoder.parameters()).dtype

    for start in range(
        0, len(positions), batch_size
    ):
        chunk = dataset.master.iloc[
            positions[start:start + batch_size]
        ]

        rows = (
            chunk["embedding_row"]
            .astype(int)
            .to_numpy()
        )

        x = torch.from_numpy(
            np.asarray(
                dataset.embeddings[rows],
                dtype=np.float32,
            ).copy()
        ).to(
            device=device,
            dtype=dtype,
        )

        category_ids = torch.tensor(
            chunk["model_category_id"]
            .astype(int)
            .to_numpy(),
            device=device,
            dtype=torch.long,
        )

        r1 = model.get_c2_residual(
            x=x,
            category_ids=category_ids,
        )

        vectors.append(
            r1.detach().float().cpu().numpy()
        )

    return (
        np.concatenate(vectors, axis=0)
        if vectors
        else np.empty(
            (0, model.embed_dim),
            dtype=np.float32,
        )
    )


def generate_semantic_ids(
    data_dir,
    checkpoint_path="out/rqvae/ebnerd_r1/checkpoint_final.pt",
    output_dir="out/semantic_ids/ebnerd_r1",
    batch_size=512,
    num_workers=0,
    event_time_window_days=3.0,
    train_event_master_path=None,
    article_entities_path=None,
    entity_idf_path=None,
    entity_idf_meta_path=None,
):
    original_c2 = hver.compute_event_c2_from_mean_h
    original_encoder = hver.encode_selected_articles

    try:
        # Train 사건 대표와 C2 결정: mean(r1(a))
        hver.compute_event_c2_from_mean_h = (
            compute_event_c2_from_mean_r1
        )

        # Validation 코사인 Fallback: r1(article)
        hver.encode_selected_articles = (
            encode_selected_articles_r1
        )

        # H 버전의 정규화 개체, 고빈도 제외, IDF 및 TimeUpdate 로직 재사용
        return hver.generate_semantic_ids(
            data_dir=data_dir,
            checkpoint_path=checkpoint_path,
            output_dir=output_dir,
            batch_size=batch_size,
            num_workers=num_workers,
            event_time_window_days=event_time_window_days,
            train_event_master_path=train_event_master_path,
            article_entities_path=article_entities_path,
            entity_idf_path=entity_idf_path,
            entity_idf_meta_path=entity_idf_meta_path,
        )

    finally:
        hver.compute_event_c2_from_mean_h = original_c2
        hver.encode_selected_articles = original_encoder


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data_dir",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        default="out/rqvae/ebnerd_r1/checkpoint_final.pt",
    )
    parser.add_argument(
        "--output_dir",
        default="out/semantic_ids/ebnerd_r1",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--event_time_window_days",
        type=float,
        default=3.0,
    )
    parser.add_argument(
        "--train_event_master_path",
        default=None,
    )
    parser.add_argument(
        "--article_entities_path",
        default=None,
    )
    parser.add_argument(
        "--entity_idf_path",
        default=None,
    )
    parser.add_argument(
        "--entity_idf_meta_path",
        default=None,
    )

    args = parser.parse_args()

    generate_semantic_ids(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        event_time_window_days=(
            None
            if args.event_time_window_days < 0
            else args.event_time_window_days
        ),
        train_event_master_path=args.train_event_master_path,
        article_entities_path=args.article_entities_path,
        entity_idf_path=args.entity_idf_path,
        entity_idf_meta_path=args.entity_idf_meta_path,
    )
