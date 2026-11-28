"""?멸린 寃고빀 (?ы븰???놁쓬): 理쒖쥌 ?먯닔 = Transformer ?먯닔 + 貫 쨌 log(1 + 吏곸쟾 N?쒓컙 ?대┃ ??

    python experiments/v2_yeomin_rerun/popularity_fusion.py --runs v1 h r

- ?대┃ ?? ?몄텧 ?쒓컖 '?댁쟾' N?쒓컙(湲곕낯 24h) ?숈븞 ?꾩껜 ?ъ슜?먭? 洹?湲곗궗瑜??대┃???잛닔 (train + validation behaviors). 誘몃옒 ?뺣낫 ?놁쓬.
- 貫: validation ???덈컲 ?덉륫(val_candidate_scores.parquet, 5_predict_val.sh 濡?留뚮벀)?먯꽌 Top-1 ??媛???믪? 媛믪쓣 怨좊Ⅸ??
  test ?먮뒗 洹?貫 瑜???踰덈쭔 ?곸슜?쒕떎. 貫=0 ? 紐⑤뜽留? 貫=?????멸린留?
- 吏??(?됰쭏??怨꾩궛???됯퇏): Top-1, AUC, MRR, nDCG@5. ?숈젏? 臾댁옉?꾨줈 怨좊Ⅴ??寃껉낵 媛숈? 湲곕?媛믪쑝濡??쇰떎
  (Top-1: ?뺣떟???숈젏 k媛??덉씠硫?1/k, ?쒖쐞: ?숈젏 媛?대뜲 ?됯퇏 ?쒖쐞). ?멸린留??????숈젏??留롮븘???대젃寃??쒕떎.

--with-age: 湲곗궗 ?섏씠??媛숈씠 蹂몃떎. ?섏씠 = log(1 + ?몄텧 ?쒓컖 - 諛쒗뻾 ?쒓컖 (?쒓컙)), 諛쒗뻾???몄텧蹂대떎 ??쑝硫?0.
  理쒖쥌 ?먯닔 = 紐⑤뜽 ?먯닔 + 貫 쨌 log(1+?대┃ ?? + 關 쨌 ?섏씠 (關 ???뚯닔???? ?ㅻ옒??湲곗궗?쇱닔濡????꾨쫫)
  貫, 關 瑜?validation ?먯꽌 媛숈씠 怨좊Ⅴ怨? ?섏씠留?/ ?멸린+?섏씠 寃곌낵??異쒕젰?쒕떎.

寃곌낵: ?붾㈃ + <V2>/analysis/popularity_fusion_seed<SEED>.json, .csv
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO, load_clicks  # noqa: E402

LAMBDAS = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1, 1.5, 2, 3, 5, 10]
MUS = [-5, -3, -2, -1, -0.5, -0.2, 0, 0.2, 0.5, 1]


def add_pop(p: pd.DataFrame, clicks: dict, window_s: int) -> pd.DataFrame:
    t = p.impression_time.to_numpy().astype("datetime64[s]").astype(np.int64)
    pop = np.zeros(len(p), dtype=np.int64)
    for i, (a, ti) in enumerate(zip(p.article_id.to_numpy(), t)):
        arr = clicks.get(a)
        if arr is not None:
            pop[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - window_s, side="left")
    p["logpop"] = np.log1p(pop)
    return p


def add_age(p: pd.DataFrame, pub: dict) -> pd.DataFrame:
    t = p.impression_time.to_numpy().astype("datetime64[s]").astype(np.int64)
    pt = p.article_id.map(pub).to_numpy(dtype="float64")
    hours = np.clip((t - pt) / 3600.0, 0, None)
    p["logage"] = np.log1p(np.nan_to_num(hours, nan=0.0))
    return p


def load(path: Path, clicks: dict, window_s: int) -> pd.DataFrame:
    p = pd.read_parquet(path, columns=["sample_index", "impression_time", "article_id", "label", "candidate_score"])
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    p = add_pop(p, clicks, window_s).sort_values(["sample_index"], kind="stable").reset_index(drop=True)
    sizes = p.groupby("sample_index").size()
    assert (sizes == 5).all(), "?됰쭏???꾨낫 5媛쒖뿬???⑸땲??
    return p


def metrics(p: pd.DataFrame, score: np.ndarray) -> dict:
    s = score.reshape(-1, 5)
    y = p.label.to_numpy().reshape(-1, 5)
    pos = (s * y).sum(1, keepdims=True)                      # ?뺣떟 ?먯닔
    neg = (y == 0)
    greater = ((s > pos) & neg).sum(1)
    ties = ((s == pos) & neg).sum(1)
    top1 = np.where(greater == 0, 1.0 / (1 + ties), 0.0)
    auc = (4 - greater - ties + 0.5 * ties) / 4.0
    # ?숈젏 ?덉뿉???뺣떟 ?쒖쐞??1+greater .. 1+greater+ties 以?洹좊벑 ???됯퇏 (?꾩쟻???쒕줈 ??踰덉뿉 怨꾩궛)
    r = np.arange(1, 7)
    cum_rr = np.concatenate([[0.0], np.cumsum(1.0 / r)])
    cum_dg = np.concatenate([[0.0], np.cumsum(1.0 / np.log2(r + 1))])
    lo, hi = greater, greater + ties + 1               # ?쒖쐞 lo+1 .. hi
    mrr = (cum_rr[hi] - cum_rr[lo]) / (ties + 1)
    ndcg = (cum_dg[hi] - cum_dg[lo]) / (ties + 1)
    return {"top1": float(top1.mean()), "auc": float(auc.mean()), "mrr": float(mrr.mean()), "ndcg@5": float(ndcg.mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="湲곕낯 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--window-hours", type=float, default=24.0)
    ap.add_argument("--with-age", action="store_true", help="湲곗궗 ?섏씠??媛숈씠 寃고빀")
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    window_s = int(a.window_hours * 3600)

    print("?대┃ 湲곕줉 ?쎈뒗 以?...", flush=True)
    clicks = load_clicks(Path(a.raw).expanduser())
    pub = None
    if a.with_age:
        arts = pd.read_parquet(Path(a.raw).expanduser() / "articles.parquet", columns=["article_id", "published_time"])
        pub = dict(zip(arts.article_id.astype(np.int64), arts.published_time.to_numpy().astype("datetime64[s]").astype(np.int64)))

    results, rows = [], []
    for run in a.runs:
        d = v2 / "transformer" / f"{run}_seed{a.seed}"
        vf, tf = d / "val_candidate_scores.parquet", d / "test_candidate_scores.parquet"
        if not (vf.exists() and tf.exists()):
            print(f"[{run}] ?덉륫 ?뚯씪 ?놁쓬 (val: {vf.exists()}, test: {tf.exists()}) ??5_predict_val.sh 癒쇱?")
            continue
        val, test = load(vf, clicks, window_s), load(tf, clicks, window_s)
        if pub is not None:
            val, test = add_age(val, pub), add_age(test, pub)

        grid = {}
        for lam in LAMBDAS:
            grid[lam] = metrics(val, val.candidate_score.to_numpy() + lam * val.logpop.to_numpy())["top1"]
        best = max(grid, key=lambda k: (grid[k], -k))       # 媛숈쑝硫??묒? 貫
        r = {"run": run, "seed": a.seed, "window_hours": a.window_hours, "val_top1_by_lambda": grid, "best_lambda": best,
             "test": {
                 "紐⑤뜽留?(貫=0)": metrics(test, test.candidate_score.to_numpy()),
                 "?멸린留?: metrics(test, test.logpop.to_numpy()),
                 f"寃고빀 (貫={best})": metrics(test, test.candidate_score.to_numpy() + best * test.logpop.to_numpy()),
             }}
        if pub is not None:
            vs, vp, va = val.candidate_score.to_numpy(), val.logpop.to_numpy(), val.logage.to_numpy()
            grid2 = {(lam, mu): metrics(val, vs + lam * vp + mu * va)["top1"] for lam in LAMBDAS for mu in MUS}
            (bl, bm) = max(grid2, key=lambda k: (grid2[k], -abs(k[0]), -abs(k[1])))
            age_only = {mu: metrics(val, mu * va)["top1"] for mu in [-1, 1]}   # ?섏씠留? ??湲곗궗 ?곗꽑(-1) / ?ㅻ옒??湲곗궗 ?곗꽑(+1)
            mu_sign = max(age_only, key=age_only.get)
            ts, tp, ta = test.candidate_score.to_numpy(), test.logpop.to_numpy(), test.logage.to_numpy()
            r["best_lambda_mu_with_age"] = [bl, bm]
            r["val_top1_with_age"] = grid2[(bl, bm)]
            r["test"][f"?섏씠留?({'??湲곗궗 ?곗꽑' if mu_sign < 0 else '?ㅻ옒??湲곗궗 ?곗꽑'})"] = metrics(test, mu_sign * ta)
            r["test"][f"寃고빀+?섏씠 (貫={bl}, 關={bm})"] = metrics(test, ts + bl * tp + bm * ta)
        results.append(r)
        print(f"\n===== {run.upper()} (seed {a.seed}) =====")
        print("validation Top-1 by 貫: " + " 쨌 ".join(f"{k}:{v:.2%}" for k, v in grid.items()))
        print(f"?좏깮 貫 = {best} (validation Top-1 {grid[best]:.2%})")
        if "best_lambda_mu_with_age" in r:
            bl, bm = r["best_lambda_mu_with_age"]
            print(f"?섏씠 ?ы븿 ?좏깮 貫 = {bl}, 關 = {bm} (validation Top-1 {r['val_top1_with_age']:.2%})")
        for name, m in r["test"].items():
            print(f"  test {name:<14} Top-1 {m['top1']:.2%} 쨌 AUC {m['auc']:.4f} 쨌 MRR {m['mrr']:.4f} 쨌 nDCG@5 {m['ndcg@5']:.4f}")
            rows.append({"run": run, "seed": a.seed, "諛⑹떇": name, **m})

    if results:
        out = v2 / "analysis"
        out.mkdir(parents=True, exist_ok=True)
        (out / f"popularity_fusion_seed{a.seed}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        pd.DataFrame(rows).to_csv(out / f"popularity_fusion_seed{a.seed}.csv", index=False)
        print(f"\n??? {out}/popularity_fusion_seed{a.seed}.json, .csv")


if __name__ == "__main__":
    main()
