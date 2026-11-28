"""test 寃곌낵 吏꾨떒: Top-1 ????27% 洹쇱쿂?먯꽌 留됲엳?붿? 蹂몃떎. ?숈뒿 ?놁쓬, ?대? ?덈뒗 ?덉륫 ?뚯씪留??쎈뒗??

    python experiments/v2_yeomin_rerun/analyze_test.py --runs v1 h r

蹂대뒗 寃?(run 留덈떎, seed 42)
  1. ?뺣떟 湲곗궗媛 Train 湲곗궗?몄? ??湲곗궗(validation-only)?몄? ??媛곴컖??Top-1
     ??湲곗궗???ш굔??遺숇뒗 諛⑹떇(Jaccard / 72h 肄붿궗??/ ?꾩껜 肄붿궗??怨?肄붿궗???ш린蹂?Top-1 (h, r 留?
  2. ?꾨낫 5媛?以???湲곗궗 鍮꾩쑉
  3. 移댄뀒怨좊━(c1) 援ъ“: ?뺣떟怨?c1 ??媛숈? ?ㅻ떟???덈뒗 ??/ ?녿뒗 ?됱쓽 Top-1
  4. ?숈뒿 ?녿뒗 湲곗???(媛숈? test ?됱뿉??
       ?멸린: ?몄텧 吏곸쟾 N?쒓컙(湲곕낯 24h) ?숈븞 ?꾩껜 ?ъ슜???대┃ ?섍? 媛??留롮? ?꾨낫
       理쒖떊: 諛쒗뻾 ?쒓컖??媛??理쒓렐???꾨낫
     ?숈젏?대㈃ ?뺣떟???숈젏 k媛??덉뿉 ?덉쓣 ??1/k 濡??쇰떎 (臾댁옉?꾨줈 怨좊Ⅸ 寃껉낵 媛숈? 湲곕?媛?
     articles.parquet ??total_pageviews 媛숈? ?꾩껜 湲곌컙 吏묎퀎??誘몃옒 ?뺣낫媛 ?욎뿬???곗? ?딅뒗??

寃곌낵: ?붾㈃ 異쒕젰 + <V2>/analysis/test_breakdown_seed<SEED>.json, .csv
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]


def load_clicks(raw: Path) -> dict:
    """article_id ???뺣젹???대┃ ?쒓컖 諛곗뿴 (train + validation behaviors)."""
    parts = []
    for split in ["train", "validation"]:
        b = pd.read_parquet(raw / split / "behaviors.parquet", columns=["impression_time", "article_ids_clicked"])
        b = b.explode("article_ids_clicked").dropna(subset=["article_ids_clicked"])
        parts.append(pd.DataFrame({"a": b.article_ids_clicked.astype(np.int64).to_numpy(),
                                   "t": b.impression_time.to_numpy().astype("datetime64[s]").astype(np.int64)}))
    c = pd.concat(parts, ignore_index=True).sort_values(["a", "t"])
    return {a: g.t.to_numpy() for a, g in c.groupby("a", sort=False)}


def frac_top1(df: pd.DataFrame, col: str) -> float:
    """?꾨낫 ?먯닔 col ??媛?????꾨낫瑜?怨좊? ?뚯쓽 Top-1 (?숈젏? 1/k)."""
    mx = df.groupby("sample_index")[col].transform("max")
    top = df[col] == mx
    k = top.groupby(df.sample_index).transform("sum")
    hit = (top & (df.label == 1)) / k
    return float(hit.groupby(df.sample_index).sum().mean())


def analyze(run: str, pred_path: Path, pre: Path, v2: Path, raw: Path, clicks: dict, pub: dict, window_h: float) -> dict:
    p = pd.read_parquet(pred_path, columns=["sample_index", "impression_time", "article_id", "c1", "label", "is_top1"])
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    inp = pre / "exports" / "rqvae_inputs"
    train_ids = set(pd.read_parquet(inp / "article_master.parquet", columns=["article_id"]).article_id)
    valonly_ids = set(pd.read_parquet(inp / "validation_article_master.parquet", columns=["article_id"]).article_id)
    p["kind"] = np.where(p.article_id.isin(train_ids), "train", np.where(p.article_id.isin(valonly_ids), "new", "other"))

    pos = p[p.label == 1].set_index("sample_index")
    correct = (p.is_top1 & (p.label == 1)).groupby(p.sample_index).any()
    rows = pd.DataFrame({"correct": correct, "pos_kind": pos.kind, "pos_article": pos.article_id})

    out = {"run": run, "test_rows": int(len(rows)), "model_top1": float(rows.correct.mean())}

    # 1. ?뺣떟 醫낅쪟蹂?    by_kind = rows.groupby("pos_kind").correct.agg(["mean", "size"])
    out["by_positive_kind"] = {k: {"top1": float(v["mean"]), "rows": int(v["size"]), "share": float(v["size"] / len(rows))}
                               for k, v in by_kind.iterrows()}

    # 1-1. ??湲곗궗: ?ш굔 遺숇뒗 諛⑹떇, 肄붿궗???ш린
    mp = v2 / "sid" / run / "validation_article_train_event_mapping.parquet"
    if mp.exists():
        m = pd.read_parquet(mp, columns=["article_id", "match_method", "match_cosine"])
        m["article_id"] = m.article_id.astype(np.int64)
        new = rows[rows.pos_kind == "new"].merge(m, left_on="pos_article", right_on="article_id", how="left")
        out["new_by_match_method"] = {k: {"top1": float(v["mean"]), "rows": int(v["size"])}
                                      for k, v in new.groupby("match_method").correct.agg(["mean", "size"]).iterrows()}
        new["cos_bin"] = pd.cut(new.match_cosine, [-1, 0.25, 0.35, 0.45, 1.01], labels=["<0.25", "0.25~0.35", "0.35~0.45", ">=0.45"])
        out["new_by_cosine"] = {str(k): {"top1": float(v["mean"]), "rows": int(v["size"])}
                                for k, v in new.groupby("cos_bin", observed=True).correct.agg(["mean", "size"]).iterrows()}

    # 2. ?꾨낫 援ъ꽦
    out["candidate_kind_share"] = {k: float(v) for k, v in p.kind.value_counts(normalize=True).items()}

    # 3. c1 援ъ“
    pos_c1 = p[p.label == 1].set_index("sample_index").c1
    p["pos_c1"] = p.sample_index.map(pos_c1)
    same = ((p.label == 0) & (p.c1 == p.pos_c1)).groupby(p.sample_index).any()
    rows["neg_same_c1"] = same
    out["by_negative_same_c1"] = {("?뺣떟怨?c1 媛숈? ?ㅻ떟 ?덉쓬" if k else "?놁쓬"): {"top1": float(v["mean"]), "rows": int(v["size"])}
                                  for k, v in rows.groupby("neg_same_c1").correct.agg(["mean", "size"]).iterrows()}

    # 4. 湲곗???    t = p.impression_time.to_numpy().astype("datetime64[s]").astype(np.int64)
    w = int(window_h * 3600)
    pop = np.zeros(len(p), dtype=np.int64)
    for i, (a, ti) in enumerate(zip(p.article_id.to_numpy(), t)):
        arr = clicks.get(a)
        if arr is not None:
            pop[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - w, side="left")
    p["pop"] = pop
    p["recency"] = p.article_id.map(pub).fillna(-1)
    out["baseline_popularity_top1"] = frac_top1(p, "pop")
    out["baseline_recency_top1"] = frac_top1(p, "recency")
    out["baseline_random_top1"] = 0.2
    out["popularity_window_hours"] = window_h
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="湲곕낯 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--window-hours", type=float, default=24.0)
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()

    print("?대┃ 湲곕줉 ?쎈뒗 以?...", flush=True)
    clicks = load_clicks(raw)
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), arts.published_time.to_numpy().astype("datetime64[s]").astype(np.int64)))

    results = []
    for run in a.runs:
        f = v2 / "transformer" / f"{run}_seed{a.seed}" / "test_candidate_scores.parquet"
        if not f.exists():
            print(f"[{run}] ?덉륫 ?뚯씪 ?놁쓬: {f}")
            continue
        r = analyze(run, f, pre, v2, raw, clicks, pub, a.window_hours)
        results.append(r)
        print(f"\n===== {run.upper()} (seed {a.seed}, test {r['test_rows']:,}?? =====")
        print(f"紐⑤뜽 Top-1 {r['model_top1']:.2%} 쨌 湲곗??? ?멸린({a.window_hours:.0f}h) {r['baseline_popularity_top1']:.2%} 쨌 "
              f"理쒖떊 {r['baseline_recency_top1']:.2%} 쨌 臾댁옉??20.00%")
        print("?뺣떟 醫낅쪟蹂?  " + " 쨌 ".join(f"{k} {v['top1']:.2%} ({v['rows']:,}?? {v['share']:.0%})" for k, v in r["by_positive_kind"].items()))
        if "new_by_match_method" in r:
            print("??湲곗궗 ?뺣떟, ?ш굔 遺숇뒗 諛⑹떇蹂?  " + " 쨌 ".join(f"{k} {v['top1']:.2%} ({v['rows']:,})" for k, v in r["new_by_match_method"].items()))
            print("??湲곗궗 ?뺣떟, 肄붿궗???ш린蹂?  " + " 쨌 ".join(f"{k} {v['top1']:.2%} ({v['rows']:,})" for k, v in r["new_by_cosine"].items()))
        print("?꾨낫 援ъ꽦:  " + " 쨌 ".join(f"{k} {v:.0%}" for k, v in r["candidate_kind_share"].items()))
        print("c1 援ъ“:  " + " 쨌 ".join(f"{k} {v['top1']:.2%} ({v['rows']:,})" for k, v in r["by_negative_same_c1"].items()))

    if results:
        out = v2 / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"test_breakdown_seed{a.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        flat = [{"run": r["run"], "model_top1": r["model_top1"], "popularity": r["baseline_popularity_top1"],
                 "recency": r["baseline_recency_top1"],
                 **{f"top1_pos_{k}": v["top1"] for k, v in r["by_positive_kind"].items()},
                 **{f"share_pos_{k}": v["share"] for k, v in r["by_positive_kind"].items()}} for r in results]
        pd.DataFrame(flat).to_csv(out / f"test_breakdown_seed{a.seed}.csv", index=False)
        print(f"\n??? {out}/test_breakdown_seed{a.seed}.json, .csv")


if __name__ == "__main__":
    main()
