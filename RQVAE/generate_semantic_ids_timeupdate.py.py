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
   
'''

from collections import Counter, defaultdict
from pathlib import Path
import argparse
import ast
import json
import math
import re
import unicodedata

import numpy as np
import pandas as pd
import torch

import generate_semantic_ids as base

# Set immediately before invoking base pipeline; restored afterwards.
_EVENT_CONTEXT = None


def event_key(value):
    """Normalize event ID without silently turning invalid/missing IDs into strings."""
    if pd.isna(value):
        raise ValueError("event_id is missing")
    return str(value)


def parse_entities(value):
    """Convert list / set / JSON array / Python list repr / delimiter string to a set."""
    if value is None:
        return set()
    if isinstance(value, (list, tuple, set, np.ndarray)):
        items = value
    elif isinstance(value, str):
        raw = value.strip()
        if not raw:
            return set()
        if raw.startswith(("[", "(", "{")):
            try:
                parsed = json.loads(raw)
            except (ValueError, TypeError):
                try:
                    parsed = ast.literal_eval(raw)
                except (SyntaxError, ValueError):
                    parsed = None
            if isinstance(parsed, dict):
                items = parsed.keys()
            elif isinstance(parsed, (list, tuple, set)):
                items = parsed
            else:
                raise ValueError(f"Cannot parse event_entities: {raw[:120]}")
        else:
            # A single entity is common; delimiters only when explicitly present.
            items = re.split(r"\s*[|;,]\s*", raw)
    else:
        try:
            if pd.isna(value):
                return set()
        except (TypeError, ValueError):
            pass
        raise TypeError(f"Unsupported event_entities type: {type(value).__name__}")

    result = set()
    for entity in items:
        if entity is None:
            continue
        label = unicodedata.normalize("NFKC", str(entity)).strip().casefold()
        if label and label not in {"nan", "none", "null"}:
            result.add(label)
    return result


def current_article_entities(all_event_entities, model_text):
    """Select only entities actually present in the current article text.

    Latin/number entities get word boundaries; Hangul entities permit attached
    grammatical particles. This is approximate string matching, not NER.
    """
    text = unicodedata.normalize("NFKC", str(model_text)).casefold()
    found = set()
    for entity in all_event_entities:
        if not entity:
            continue
        if re.search(r"[가-힣]", entity):
            present = entity in text
        else:
            present = re.search(r"(?<!\w)" + re.escape(entity) + r"(?!\w)", text) is not None
        if present:
            found.add(entity)
    return found


def load_event_entity_lookup(path):
    if not path.is_file():
        raise FileNotFoundError(f"Event master not found: {path}")
    frame = pd.read_parquet(path)
    missing = {"event_id", "event_entities"} - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")
    lookup = defaultdict(set)
    for event_id, entities in zip(frame["event_id"], frame["event_entities"]):
        lookup[event_key(event_id)].update(parse_entities(entities))
    return dict(lookup)


@torch.inference_mode()
def match_validation_articles_to_train(
    model, validation_dataset, positions, train_event_representations,
    train_master, device, batch_size=512, event_time_window_days=3.0
):
    columns = [
        "article_id", "matched_train_event_id", "match_cosine",
        "match_used_time_window", "match_reference_train_last",
        "match_delta_hours", "eligible_train_events", "match_processing_order",
        "match_method", "match_entity_tfidf", "match_article_entity_count",
        "match_train_entity_count_before", "match_new_entities_added",
    ]
    if len(positions) == 0:
        return pd.DataFrame(columns=columns)
    if not train_event_representations:
        raise ValueError("Train event representative vectors missing")
    if _EVENT_CONTEXT is None:
        raise RuntimeError("Use generate_semantic_ids() wrapper to initialize event master paths")

    valid_master = validation_dataset.master
    for name in ("article_id", "event_id", "model_text"):
        if name not in valid_master.columns:
            raise ValueError(f"validation_dataset.master missing '{name}'")
    if "event_id" not in train_master.columns:
        raise ValueError("train_master missing 'event_id'")

    valid_times, valid_time_col = base.get_publication_times(valid_master)
    train_times, train_time_col = base.get_publication_times(train_master)
    if valid_time_col is None or train_time_col is None:
        raise ValueError("Both Train and Validation must have published timestamps")
    valid_times = pd.to_datetime(valid_times, utc=True, errors="coerce")
    train_times = pd.to_datetime(train_times, utc=True, errors="coerce")
    if valid_times.iloc[list(positions)].isna().any():
        raise ValueError("Some Validation articles have no published time; cannot process causally")

    ids = sorted(train_event_representations, key=lambda x: str(x))
    keys = [event_key(i) for i in ids]
    if len(set(keys)) != len(keys):
        raise ValueError("Non-unique normalized Train event IDs")
    n = len(ids)
    vectors = np.stack([train_event_representations[e] for e in ids]).astype(np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    # Train-only event entity inventory, NOT the expanded validation inventory.
    original_train_entities = load_event_entity_lookup(_EVENT_CONTEXT["train_events"])
    expanded_event_entities = load_event_entity_lookup(_EVENT_CONTEXT["expanded_events"])
    missing_train = set(keys) - set(original_train_entities)
    if missing_train:
        raise ValueError(f"Train events missing from event_master.parquet: {list(missing_train)[:5]}")

    entity_sets = [set(original_train_entities[k]) for k in keys]

    # Fixed training-only IDF; no future Validation entity distribution in vocabulary/IDF.
    df = Counter(entity for group in entity_sets for entity in group)
    def weight_sq(entity):
        idf = math.log((1 + n) / (1 + df.get(entity, 0))) + 1.0
        return idf * idf

    norms_sq = np.array([sum(weight_sq(e) for e in group) for group in entity_sets], dtype=float)
    inverted = defaultdict(set)
    for j, group in enumerate(entity_sets):
        for token in group:
            inverted[token].add(j)

    # Initial timestamp = latest Train article timestamp only.
    last_by_id = (
        pd.DataFrame({"event_id": [event_key(x) for x in train_master["event_id"]],
                      "timestamp": train_times.to_numpy()})
        .groupby("event_id")["timestamp"].max().to_dict()
    )
    last_times = pd.to_datetime([last_by_id.get(k, pd.NaT) for k in keys], utc=True)
    if last_times.isna().all():
        raise ValueError("All Train event timestamps are missing")

    article_vectors = base.encode_selected_articles(
        model, validation_dataset, positions, device, batch_size
    ).astype(np.float32)
    if len(article_vectors) != len(positions):
        raise ValueError("Article encoding count differs from positions")
    article_vectors /= np.maximum(np.linalg.norm(article_vectors, axis=1, keepdims=True), 1e-12)

    # Do not depend on incidental DatetimeIndex positions after sorting.
    order = sorted(range(len(positions)), key=lambda i: (
        valid_times.iloc[positions[i]], str(valid_master.iloc[positions[i]]["article_id"])
    ))

    rows = []
    methods = Counter()
    updates = 0
    for rank, i in enumerate(order):
        row = valid_master.iloc[positions[i]]
        article_id = str(row["article_id"])
        timestamp = valid_times.iloc[positions[i]]
        original_event_id = event_key(row["event_id"])
        if original_event_id not in expanded_event_entities:
            raise ValueError(
                f"Validation original event_id={original_event_id} not in expanded master; "
                "cannot safely infer article entities"
            )
        article_entities = current_article_entities(
            expanded_event_entities[original_event_id], row["model_text"]
        )

        cosines = vectors @ article_vectors[i]
        if event_time_window_days is not None:
            # Earlier-or-equal last timestamp, and within 72h.
            diff_h = np.asarray((timestamp - last_times).total_seconds() / 3600.0)
            eligible = np.isfinite(diff_h) & (diff_h >= 0) & (
                diff_h <= event_time_window_days * 24
            )
        else:
            eligible = np.ones(n, dtype=bool)
        candidate_count = int(eligible.sum())
        used_window = candidate_count > 0

        tfidf_score = np.nan
        if not used_window:
            # Time-window fallback: all train events, cosine similarity.
            best = int(np.argmax(cosines))
            method = "cosine_all_no_time_candidates"
        else:
            # Binary TF-IDF entity cosine with train-only, fixed IDF weights.
            # Only events sharing entities need nonzero dot-products.
            article_norm_sq = sum(weight_sq(t) for t in article_entities)
            dots = defaultdict(float)
            if article_norm_sq > 0:
                for token in article_entities:
                    w2 = weight_sq(token)
                    for j in inverted.get(token, ()):
                        if eligible[j]:
                            dots[j] += w2
            best_tfidf = 0.0
            best_tfidf_j = None
            for j, numerator in dots.items():
                denominator = math.sqrt(article_norm_sq * norms_sq[j])
                score = numerator / denominator if denominator > 0 else 0.0
                if score > best_tfidf or (score == best_tfidf and (best_tfidf_j is None or j < best_tfidf_j)):
                    best_tfidf, best_tfidf_j = score, j
            tfidf_score = best_tfidf
            if best_tfidf_j is not None and best_tfidf >= 0.3:
                best = int(best_tfidf_j)
                method = "tfidf_72h"
            else:
                best = int(np.argmax(np.where(eligible, cosines, -np.inf)))
                method = "cosine_72h_below_threshold"

        prev_last = last_times[best]
        delta = (timestamp - prev_last).total_seconds() / 3600.0 if pd.notna(prev_last) else np.nan
        before = len(entity_sets[best])
        new_tokens = article_entities - entity_sets[best]
        if new_tokens:
            entity_sets[best].update(new_tokens)
            for token in new_tokens:
                inverted[token].add(best)
                norms_sq[best] += weight_sq(token)
        # Update only latest event timestamp and set of entities; never mutate vector/C2.
        if pd.isna(prev_last) or timestamp > prev_last:
            last_times = last_times.where(np.arange(n) != best, timestamp)
            updates += 1

        rows.append({
            "article_id": article_id,
            "matched_train_event_id": ids[best],
            "match_cosine": float(cosines[best]),
            "match_used_time_window": bool(used_window),
            "match_reference_train_last": prev_last,
            "match_delta_hours": float(delta),
            "eligible_train_events": candidate_count,
            "match_processing_order": rank,
            "match_method": method,
            "match_entity_tfidf": tfidf_score,
            "match_article_entity_count": len(article_entities),
            "match_train_entity_count_before": before,
            "match_new_entities_added": len(new_tokens),
        })
        methods[method] += 1
        if (rank + 1) % 500 == 0:
            print(f"Processed: {rank + 1:,}/{len(positions):,}")

    result = pd.DataFrame(rows, columns=columns)
    if result["article_id"].duplicated().any():
        raise ValueError("Duplicate Validation article_id in matching output")
    print("\n=== Entity TF-IDF + Time Update Matching Summary ===")
    print(f"Matched articles: {len(result):,}")
    for method, count in methods.items():
        print(f"  {method}: {count:,}")
    print(f"Event latest-time updates: {updates:,}")
    print(f"New entity additions: {result['match_new_entities_added'].sum():,}")
    return result


def generate_semantic_ids(
    data_dir, checkpoint_path, output_dir, batch_size=512,
    num_workers=0, event_time_window_days=3.0,
    train_event_master_path=None, expanded_event_master_path=None,
):
    global _EVENT_CONTEXT
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = Path(train_event_master_path) if train_event_master_path else data_dir / "event_master.parquet"
    expanded_path = Path(expanded_event_master_path) if expanded_event_master_path else data_dir / "event_master_with_validation.parquet"
    old_matcher = base.match_validation_articles_to_train
    old_context = _EVENT_CONTEXT
    try:
        _EVENT_CONTEXT = {"train_events": train_path, "expanded_events": expanded_path}
        base.match_validation_articles_to_train = match_validation_articles_to_train
        result = base.generate_semantic_ids(
            data_dir=str(data_dir),
            checkpoint_path=str(checkpoint_path),
            output_dir=str(output_dir),
            batch_size=batch_size,
            num_workers=num_workers,
            event_time_window_days=event_time_window_days,
        )
    finally:
        base.match_validation_articles_to_train = old_matcher
        _EVENT_CONTEXT = old_context

    mapping_path = output_dir / "validation_article_train_event_mapping.parquet"
    if not mapping_path.is_file():
        raise FileNotFoundError(f"Base pipeline did not write mapping to {mapping_path}")
    mapping = pd.read_parquet(mapping_path)
    required = {"match_processing_order", "match_used_time_window", "match_delta_hours"}
    if not required <= set(mapping.columns):
        raise ValueError(f"Mapping missing verification columns: {sorted(required - set(mapping.columns))}")
    window = mapping[mapping["match_used_time_window"].astype(bool)]
    if event_time_window_days is not None:
        violations = ~window["match_delta_hours"].between(0, event_time_window_days * 24)
        print(f"72h window violations: {int(violations.sum()):,}")
        if violations.any():
            raise RuntimeError("Found a matching result violating the configured time window")
    print(f"Output directory: {output_dir}")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output_dir", default="out/semantic_ids/ebnerd/tfidf_timeupdate")
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--event_time_window_days", type=float, default=3.0)
    parser.add_argument("--train_event_master_path", default=None)
    parser.add_argument("--expanded_event_master_path", default=None)
    args = parser.parse_args()
    generate_semantic_ids(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        event_time_window_days=args.event_time_window_days if args.event_time_window_days >= 0 else None,
        train_event_master_path=args.train_event_master_path,
        expanded_event_master_path=args.expanded_event_master_path,
    )
