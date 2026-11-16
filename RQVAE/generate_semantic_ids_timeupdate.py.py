'''
1. Validation 기사 발행시각 오름차순 처리
2. 72시간 이내 Train 사건 탐색
3. 해당 사건들과 TF-IDF 개체 유사도 계산
- 최고 유사도 ≥ 0.3 → 해당 Train 사건에 매칭
- 최고 유사도 < 0.3 → 임베딩 코사인 유사도로 매칭
- 72시간 이내 사건이 아예 없음 → 전체 Train 사건을 대상으로 코사인 유사도 매칭
4. 매칭된 Train 사건 정보 갱신
- 마지막 발행시각 갱신
- 개체 집합에 Validation 기사 개체 추가

- Z와 C2는 갱신하지 않음
5. 매칭된 Train 사건의 기존 C2 상속
-------------------------------------------
event_master_with_validation.parquet(Valid까지 포함한 사건 개체 집합) 및
event_master.parquet에 있는 것  :
event_id: Train 사건 ID
event_entities: 해당 사건의 개체 집합
event_last_added_time: 사건에 마지막으로 추가된 기사 시각
event_entity_count: 사건의 개체 수
--------------------------------------------
[코드 동작 순서]
1. 데이터 불러오기: event_master.parquet에서 Train 사건의 초기 개체 집합을 가져오고 event_master_with_validation.parquet를 Validation 개체 식별에 활용.
2. 기사 정렬: Validation 기사 3,122개를 발행시각 오름차순으로 처리.
3. 시간 필터링: 현재 기사 발행시각 기준 72시간 이내인 Train 사건만 탐색.
4. TF-IDF 매칭: 개체 유사도가 0.3 이상인 사건이 있다면 가장 높은 사건 선택.
5. 코사인 Fallback: 0.3 미만이면 72시간 내 사건 중 임베딩 코사인 유사도가 가장 높은 사건 선택. 시간 조건을 만족하는 사건도 없다면 전체 Train 사건을 대상으로 선택.
6. Train 사건 갱신: 선택된 사건의 마지막 발행시각과 개체 집합 갱신. 다음 Validation 기사 매칭부터 반영.
7. C2 상속: 선택된 Train 사건의 기존 C2를 상속하며 Z와 C2는 변경하지 않음.
8. 결과 저장 및 검증: 기존 SID 생성 파이프라인을 실행하고 매칭 결과와 시간 조건 위반 여부 확인.
******
event_master_with_validation.parquet는 기사별 개체 목록이 아니라 사건 전체의 개체 목록이 들어있음
따라서 해당 사건의 개체 목록 중 현재 Validation 기사의 model_text에 실제로 등장하는 개체만 추출하도록 함
(기사 텍스트와 개체명을 대조해서 근사하는 st)

예) 사건 전체 개체가 {삼성전자, 엔비디아, 애플}이더라도 현재 기사에 삼성전자만 등장한다면 {삼성전자}만 사용
-> 미래 기사의 개체 집합을 통째로 추가하는 문제를 피할 수 있지 않을까? 
   단, 개체 후보 목록 자체에는 Validation 전체 정보가 포함돼 있으므로 엄밀한 시점별 정보 제한을 보장하지는 않음
   
'''
from pathlib import Path



import numpy as np
import pandas as pd
import torch


import generate_semantic_ids as base


@torch.inference_mode()
def match_validation_articles_to_train(
    model,
    validation_dataset,
    positions,
    train_event_representations,
    train_master,
    device,
    batch_size=512,
    event_time_window_days=3.0
):
    columns = [
        "article_id", "matched_train_event_id", "match_cosine",
        "match_used_time_window", "match_reference_train_last",
        "match_delta_hours", "eligible_train_events",
        "match_processing_order"
    ]

    if not positions:
        return pd.DataFrame(columns=columns)

    if not train_event_representations:
        raise ValueError("Train 사건 대표 벡터가 없습니다.")

    valid_master = validation_dataset.master
    valid_times, valid_time_col = base.get_publication_times(valid_master)
    train_times, train_time_col = base.get_publication_times(train_master)

    if valid_time_col is None:
        raise ValueError("Validation 발행 시각이 없어 시간순 처리를 수행할 수 없습니다.")

    if train_time_col is None:
        raise ValueError("Train 발행 시각이 없어 사건 마지막 시각을 계산할 수 없습니다.")

    train_event_ids = sorted(train_event_representations)
    train_vectors = np.stack([
        train_event_representations[e] for e in train_event_ids
    ]).astype(np.float32)

    train_vectors /= np.maximum(
        np.linalg.norm(train_vectors, axis=1, keepdims=True), 1e-12
    )

    train_last_times = pd.DataFrame({
        "event_id": train_master["event_id"].to_numpy(),
        "timestamp": train_times.to_numpy()
    }).groupby("event_id")["timestamp"].max().to_dict()

    article_vectors = base.encode_selected_articles(
        model, validation_dataset, positions, device, batch_size
    )

    article_vectors = article_vectors.astype(np.float32)
    article_vectors /= np.maximum(
        np.linalg.norm(article_vectors, axis=1, keepdims=True), 1e-12
    )

    order = sorted(
        range(len(positions)),
        key=lambda i: (
            pd.isna(valid_times.iloc[positions[i]]),
            valid_times.iloc[positions[i]]
            if pd.notna(valid_times.iloc[positions[i]])
            else pd.Timestamp.max.tz_localize("UTC"),
            str(valid_master.iloc[positions[i]]["article_id"])
        )
    )

    rows = []
    window_count = 0
    fallback_count = 0
    update_count = 0

    print("\n=== Validation Time Update Matching ===")
    print(f"Validation-only articles: {len(positions):,}")
    print(f"Train events: {len(train_event_ids):,}")
    print(f"Time window: {event_time_window_days} days")
    print("Processing order: published_time ascending")

    for processing_order, i in enumerate(order):
        position = positions[i]
        article_id = str(valid_master.iloc[position]["article_id"])
        article_time = valid_times.iloc[position]

        similarity = train_vectors @ article_vectors[i]
        eligible = np.ones(len(train_event_ids), dtype=bool)
        used_window = False
        candidate_count = 0

        if event_time_window_days is not None and pd.notna(article_time):
            hours = np.array([
                (article_time - train_last_times[e]).total_seconds() / 3600
                if pd.notna(train_last_times.get(e)) else np.inf
                for e in train_event_ids
            ])

            eligible = (hours >= 0) & (hours <= event_time_window_days * 24)
            candidate_count = int(eligible.sum())
            used_window = candidate_count > 0

            if not used_window:
                eligible[:] = True

        best = int(np.argmax(np.where(eligible, similarity, -np.inf)))
        matched_id = int(train_event_ids[best])
        previous_last = train_last_times.get(matched_id, pd.NaT)

        delta_hours = (
            (article_time - previous_last).total_seconds() / 3600
            if pd.notna(article_time) and pd.notna(previous_last)
            else np.nan
        )

        rows.append({
            "article_id": article_id,
            "matched_train_event_id": matched_id,
            "match_cosine": float(similarity[best]),
            "match_used_time_window": used_window,
            "match_reference_train_last": previous_last,
            "match_delta_hours": delta_hours,
            "eligible_train_events": candidate_count,
            "match_processing_order": processing_order
        })

        if used_window:
            window_count += 1
        else:
            fallback_count += 1

        # 매칭된 Train 사건의 마지막 시각을 현재 기사 시각으로 갱신
        if pd.notna(article_time):
            if pd.isna(previous_last) or article_time > previous_last:
                train_last_times[matched_id] = article_time
                update_count += 1

        if (processing_order + 1) % 500 == 0:
            print(f"Processed: {processing_order + 1:,}/{len(positions):,}")

    result = pd.DataFrame(rows, columns=columns)

    if result["article_id"].duplicated().any():
        raise ValueError("Validation 매칭 결과에 중복된 article_id가 있습니다.")

    print("\n=== Time Update Matching Summary ===")
    print(f"Matched articles: {len(result):,}")
    print(f"72h window: {window_count:,}")
    print(f"Fallback: {fallback_count:,}")
    print(f"Train event time updates: {update_count:,}")

    return result


def generate_semantic_ids(
    data_dir,
    checkpoint_path,
    output_dir,
    batch_size=512,
    num_workers=0,
    event_time_window_days=3.0
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    original_matcher = base.match_validation_articles_to_train

    try:
        base.match_validation_articles_to_train = match_validation_articles_to_train

        result = base.generate_semantic_ids(
            data_dir=str(data_dir),
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            event_time_window_days=event_time_window_days
        )
    finally:
        base.match_validation_articles_to_train = original_matcher

    mapping_path = output_dir / "validation_article_train_event_mapping.parquet"
    mapping = pd.read_parquet(mapping_path)

    print("\n=== Time Update Verification ===")

    if "match_processing_order" not in mapping.columns:
        raise ValueError("Time Update 로그가 저장되지 않았습니다.")

    window = mapping[mapping["match_used_time_window"]]
    violations = ~window["match_delta_hours"].between(
        0, event_time_window_days * 24
    ) if event_time_window_days is not None else pd.Series(False, index=window.index)

    print(f"Window matches: {len(window):,}")
    print(f"Time window violations: {violations.sum():,}")

    if violations.any():
        raise RuntimeError("72시간 조건을 위반한 매칭이 발견되었습니다.")

    print(f"Output directory: {output_dir}")
    print("Time Update SID generation complete.")

    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="out/semantic_ids/ebnerd/timeupdate")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--event_time_window_days", type=float, default=3.0)
    args = parser.parse_args()

    generate_semantic_ids(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        event_time_window_days=(
            args.event_time_window_days
            if args.event_time_window_days >= 0 else None
        )
    )
