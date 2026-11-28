"""?멸린???쒓컙 李?遺꾩꽍: 湲곗궗 ?멸린?꾨? 吏곸쟾 紐??쒓컙 ?대┃?쇰줈 ? ??媛????留욌굹 (?ы븰???놁쓬).

    python experiments/v2_yeomin_rerun/popularity_window.py                       # H 쨌 R, seed 42 쨌 123 쨌 2026
    python experiments/v2_yeomin_rerun/popularity_window.py --runs h --seeds 42

吏湲??멸린??= ln(1 + ?몄텧 吏곸쟾 24?쒓컙 ?대┃ ??. ??24?쒓컙??0.5 ~ 72?쒓컙?쇰줈 諛붽퓭 媛硫?鍮꾧탳?쒕떎.
?뺤쓽??surge_fusion.py 쨌 popularity_fusion.py ? 媛숇떎.
  ?대┃ 湲곕줉 = train + validation behaviors ??紐⑤뱺 ?대┃ (?대┃ ?쒓컖 = 洹??몄텧??impression_time), 李?= [?몄텧 ?쒓컖 ??w, ?몄텧 ?쒓컖)
  ?섏씠 = ln(1 + max(0, ?몄텧 ?쒓컖 ??諛쒗뻾 ?쒓컖) ?쒓컙)
  媛以묒튂??李쎈쭏??validation ???덈컲 Top-1 濡?怨좊Ⅴ怨?(媛숈쑝硫??묒? 履?, test ????踰덈쭔. 吏??= popularity_fusion.metrics (?숈젏 湲곕?媛?.

異쒕젰 (?ㅽ뿕留덈떎)
  - 湲곌컙: validation ?덈컲 쨌 test ??泥?쨌 留덉?留??몄텧 ?쒓컖
  - ?뺣떟 쨌 ?ㅻ떟 湲곗궗 ?섏씠 遺꾩쐞?? ?뺣떟 湲곗궗 ?대┃???몄텧 紐??쒓컙 ?꾩뿉 ?덉뿀??(48?쒓컙 ?? ?꾩쟻 鍮꾩쑉)
  - 李쎈퀎: ?뺣떟 쨌 ?ㅻ떟 ?대┃ 0 鍮꾩쑉, ?멸린留?Top-1, ?멸린 + ?섏씠 (Transformer ?놁쓬), Transformer + ?멸린 + ?섏씠 (seed蹂?쨌 ?됯퇏)
?낅젰 (?쎄린留?: 1pos4neg split (validation ?덈컲, test), Transformer ?덉륫 <V2>/transformer/<run>_seed<seed>/{val,test}_candidate_scores.parquet, ?먮낯 behaviors 쨌 articles
?ㅻⅨ 紐⑤뜽 (?? NRMS) ???먯닔濡?蹂대젮硫?--pred-template 怨?--out-name. ?꾨옒 ?쒖쓽 'Transformer' ??洹?紐⑤뜽???삵븳??
寃곌낵: ?붾㈃ + <V2>/analysis/popularity_window/ (?대? ?덉쑝硫?popularity_window_2/ ?? ?덉뿉
  <run>_click_timing.csv  ?몄텧 ??0~6?쒓컙 15遺??⑥쐞 쨌 6~72?쒓컙 1?쒓컙 ?⑥쐞, ?뺣떟 쨌 ?ㅻ떟 湲곗궗 ?대┃ 鍮꾩쑉怨??꾩쟻 鍮꾩쑉
  <run>_windows.csv       李쎈퀎 ?대┃ 0 鍮꾩쑉, ?멸린留? ?멸린+?섏씠, Transformer+?멸린+?섏씠 (seed蹂?쨌 ?됯퇏) validation 쨌 test 吏??  <run>_summary.csv       湲곌컙, ?좎쭨蹂??몄텧 ?? ?뺣떟 쨌 ?ㅻ떟 湲곗궗 ?섏씠, 留덉?留??대┃???몄텧 紐?遺??? 3?쒓컙/24?쒓컙 ?대┃ 鍮꾩쑉
  result.json             ???댁슜 ?꾩껜
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO  # noqa: E402
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402
from surge_fusion import load_click_log, load_split, paths, pick, to_s, top1_fast  # noqa: E402

WINDOWS = [0.5, 1, 2, 3, 6, 12, 24, 48, 72]
BIG = 10 ** 10   # article_id * BIG + 珥?(珥?< 1e10)


def counts(key: np.ndarray, art: np.ndarray, t: np.ndarray, w_hours: float, from_hours: float = 0.0) -> np.ndarray:
    """[t ??w, t ??from) ???대┃ ??(湲곕낯 [t ??w, t)). key = ?뺣젹??article_id * BIG + ?대┃ 珥?"""
    a = art * BIG + t
    return np.searchsorted(key, a - int(from_hours * 3600), "left") - np.searchsorted(key, a - int(w_hours * 3600), "left")


def last_click_minutes(key: np.ndarray, art: np.ndarray, t: np.ndarray) -> np.ndarray:
    """?몄텧 吏곸쟾 留덉?留??대┃??紐?遺??꾩씤媛 (?놁쑝硫?nan)."""
    i = np.searchsorted(key, art * BIG + t, "left") - 1
    ic = np.maximum(i, 0)
    ok = (i >= 0) & (key[ic] // BIG == art)
    return np.where(ok, (t - key[ic] % BIG) / 60.0, np.nan)


TIMING_BINS = [(i / 4, (i + 1) / 4) for i in range(24)] + [(h, h + 1) for h in range(6, 72)]   # ?쒓컙


def windows_df(rows: list) -> pd.DataFrame:
    """李쎈퀎 寃곌낵瑜????됱뵫 ?쇱튇 ??"""
    out = []
    for x in rows:
        d = {"李??쒓컙)": x["李??쒓컙)"], "?뺣떟 ?대┃ 0": x["?뺣떟 ?대┃ 0"], "?ㅻ떟 ?대┃ 0": x["?ㅻ떟 ?대┃ 0"],
             "?멸린留?val Top-1": x["?멸린留?val"], "?멸린留?test Top-1": x["?멸린留?test"],
             "?멸린+?섏씠 貫": x["?멸린+?섏씠 (貫,關)"][0], "?멸린+?섏씠 關": x["?멸린+?섏씠 (貫,關)"][1], "?멸린+?섏씠 val Top-1": x["?멸린+?섏씠 val"]}
        d.update({f"?멸린+?섏씠 test {m}": v for m, v in x["?멸린+?섏씠 test"].items()})
        for seed, y in x["Transformer+?멸린+?섏씠 seed蹂?].items():
            d[f"T+?멸린+?섏씠 seed{seed} 貫"], d[f"T+?멸린+?섏씠 seed{seed} 關"] = y["貫,關"]
            d[f"T+?멸린+?섏씠 seed{seed} val Top-1"] = y["val"]
            d.update({f"T+?멸린+?섏씠 seed{seed} test {m}": v for m, v in y["test"].items()})
        d["T+?멸린+?섏씠 val Top-1 ?됯퇏"] = x["Transformer+?멸린+?섏씠 val ?됯퇏"]
        d["T+?멸린+?섏씠 test Top-1 ?됯퇏"] = x["Transformer+?멸린+?섏씠 test ?됯퇏"]
        out.append(d)
    return pd.DataFrame(out)


def best(score_fn, grid, Yv):
    g = {k: top1_fast(score_fn(*k), Yv) for k in grid}
    k = pick(g)
    return k, g[k]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["h", "r"])
    ap.add_argument("--seeds", nargs="+", default=["42", "123", "2026"])
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="湲곕낯 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--exp-prefix", default="v2_")
    ap.add_argument("--windows", nargs="+", type=float, default=WINDOWS)
    ap.add_argument("--pred-template", default=None,
                    help="紐⑤뜽 ?먯닔 ?뚯씪 ?꾩튂 (湲곕낯 Transformer). ?? '{v2}/nrms/{run}/final/lr0.0001_do0.2_seed{seed}/{split}_candidate_scores.parquet'"
                         " ({split} = val | test). NRMS + ?멸린 + ?섏씠 瑜?蹂????")
    ap.add_argument("--out-name", default="popularity_window", help="寃곌낵 ?대뜑 ?대쫫 (<V2>/analysis/<?대쫫>/)")
    a = ap.parse_args()
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()

    clicks = load_click_log(raw)
    key = np.sort(clicks.a.to_numpy().astype(np.int64) * BIG + clicks.t.to_numpy().astype(np.int64))
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), to_s(arts.published_time)))
    grid_pa = [(l, m) for l in LAMBDAS for m in MUS if l != 0]

    res, outputs = {}, {}
    for run in a.runs:
        D = {}   # seed ??{split: p}
        for seed in a.seeds:
            P = paths(run, pre, v2, a.exp_prefix, seed)
            if a.pred_template:
                P["pred"] = {s: Path(a.pred_template.format(v2=v2, run=run, seed=seed, split=s)).expanduser() for s in ("val", "test")}
            miss = [str(x) for x in [*P["seq"].values(), *P["pred"].values()] if not x.exists()]
            if miss:
                print(f"[{run} seed {seed}] ?낅젰 ?놁쓬 ??嫄대꼫?: {miss}")
                continue
            D[seed] = {s: load_split(P["pred"][s], P["seq"][s])[0] for s in ("val", "test")}
        if not D:
            continue
        p0 = next(iter(D.values()))
        F = {}
        for s in ("val", "test"):
            p = p0[s]
            art, t = p.article_id.to_numpy().astype(np.int64), p.t.to_numpy().astype(np.int64)
            pt = np.array([pub.get(x, np.nan) for x in art], dtype="float64")
            age_h = np.nan_to_num(np.clip((t - pt) / 3600.0, 0, None), nan=0.0)
            F[s] = {"Y": p.label.to_numpy(), "la": np.log1p(age_h), "age_h": age_h, "art": art, "t": t,
                    "lp": {w: np.log1p(counts(key, art, t, w)) for w in a.windows}}
            for seed in D:   # 紐⑤뱺 seed ????쨌 ?꾨낫 ?쒖꽌媛 媛숈?吏
                q = D[seed][s]
                if not (np.array_equal(q.article_id.to_numpy(), art) and np.array_equal(q.label.to_numpy(), F[s]["Y"])):
                    raise ValueError(f"[{run}] seed {seed} {s} ???꾨낫 ?쒖꽌媛 ?ㅻ쫭?덈떎")
        Fv, Ft = F["val"], F["test"]
        pos_t = Ft["Y"] == 1
        r = {"湲곌컙": {s: [str(pd.to_datetime(F[s]["t"].min(), unit="s")), str(pd.to_datetime(F[s]["t"].max(), unit="s"))] for s in F},
             "??: {s: int(len(F[s]["Y"]) // 5) for s in F},
             "湲곗궗 ?섏씠 遺꾩쐞??(?쒓컙, 10/25/50/75/90%)": {
                 "?뺣떟": np.percentile(Ft["age_h"][pos_t], [10, 25, 50, 75, 90]).round(1).tolist(),
                 "?ㅻ떟": np.percentile(Ft["age_h"][~pos_t], [10, 25, 50, 75, 90]).round(1).tolist()}}
        # ?뺣떟 쨌 ?ㅻ떟 湲곗궗 ?대┃???몄텧 紐??쒓컙 ?꾩뿉 ?덉뿀??(72?쒓컙 ?? 15遺?/ 1?쒓컙 ?⑥쐞)
        timing = []
        for name, m in [("?뺣떟", pos_t), ("?ㅻ떟", ~pos_t)]:
            aa, tt = Ft["art"][m], Ft["t"][m]
            tot = np.array([counts(key, aa, tt, hi, lo).sum() for lo, hi in TIMING_BINS], dtype=float)
            timing.append((name, tot / max(tot.sum(), 1)))
        tdf = pd.DataFrame({"?몄텧 ???쒖옉(遺?": [lo * 60 for lo, _ in TIMING_BINS], "?몄텧 ????遺?": [hi * 60 for _, hi in TIMING_BINS]})
        for name, sh in timing:
            tdf[f"{name} ?대┃ 鍮꾩쑉"] = sh
            tdf[f"{name} ?꾩쟻 鍮꾩쑉"] = np.cumsum(sh)
        cum = tdf["?뺣떟 ?꾩쟻 鍮꾩쑉"].to_numpy()
        ends = tdf["?몄텧 ????遺?"].to_numpy()
        r["?뺣떟 ?대┃ ?꾩쟻 鍮꾩쑉 (?몄텧 ??1쨌3쨌6쨌12쨌24?쒓컙)"] = {h: float(cum[ends == h * 60][0]) for h in (1, 3, 6, 12, 24)}
        pa, ptt = Ft["art"][pos_t], Ft["t"][pos_t]
        lc = last_click_minutes(key, pa, ptt)
        c3, c24 = counts(key, pa, ptt, 3), counts(key, pa, ptt, 24)
        ratio = np.where(c24 > 0, c3 / np.maximum(c24, 1), np.nan)
        days = pd.Series(pd.to_datetime(Ft["t"][pos_t], unit="s").date).value_counts().sort_index()
        summ = [("湲곌컙", s, f"{v[0]} ~ {v[1]}") for s, v in r["湲곌컙"].items()]
        summ += [("??, s, v) for s, v in r["??].items()]
        summ += [("test ?좎쭨蹂??몄텧", str(d), int(n)) for d, n in days.items()]
        qs = [10, 25, 50, 75, 90]
        for k, v in r["湲곗궗 ?섏씠 遺꾩쐞??(?쒓컙, 10/25/50/75/90%)"].items():
            summ += [(f"{k} 湲곗궗 ?섏씠(?쒓컙)", f"{q}%", x) for q, x in zip(qs, v)]
        summ += [("?뺣떟 湲곗궗 留덉?留??대┃(遺???", f"{q}%", float(x)) for q, x in zip(qs, np.nanpercentile(lc, qs).round(2))]
        summ += [("?뺣떟 湲곗궗 3?쒓컙/24?쒓컙 ?대┃ 鍮꾩쑉", f"{q}%", float(x)) for q, x in zip(qs, np.nanpercentile(ratio, qs).round(3))]
        summ += [("?뺣떟 ?대┃ ?놁쓬(?몄텧 ???꾩껜)", "鍮꾩쑉", float(np.isnan(lc).mean()))]

        rows = []
        for w in a.windows:
            lpv, lpt = Fv["lp"][w], Ft["lp"][w]
            row = {"李??쒓컙)": w,
                   "?뺣떟 ?대┃ 0": float((np.expm1(lpt[pos_t]) == 0).mean()),
                   "?ㅻ떟 ?대┃ 0": float((np.expm1(lpt[~pos_t]) == 0).mean()),
                   "?멸린留?val": top1_fast(lpv, Fv["Y"]), "?멸린留?test": top1_fast(lpt, Ft["Y"])}
            (l, m), v = best(lambda l, m: l * lpv + m * Fv["la"], grid_pa, Fv["Y"])
            row["?멸린+?섏씠 (貫,關)"] = [l, m]
            row["?멸린+?섏씠 val"] = v
            row["?멸린+?섏씠 test"] = metrics(pd.DataFrame({"label": Ft["Y"]}), l * lpt + m * Ft["la"])
            seeds = {}
            for seed in D:
                sv, st = D[seed]["val"].candidate_score.to_numpy(), D[seed]["test"].candidate_score.to_numpy()
                (l2, m2), v2_ = best(lambda l, m: sv + l * lpv + m * Fv["la"], [(l, m) for l in LAMBDAS for m in MUS], Fv["Y"])
                seeds[seed] = {"貫,關": [l2, m2], "val": v2_,
                               "test": metrics(pd.DataFrame({"label": Ft["Y"]}), st + l2 * lpt + m2 * Ft["la"])}
            row["Transformer+?멸린+?섏씠 seed蹂?] = seeds
            row["Transformer+?멸린+?섏씠 val ?됯퇏"] = float(np.mean([x["val"] for x in seeds.values()]))
            row["Transformer+?멸린+?섏씠 test ?됯퇏"] = float(np.mean([x["test"]["top1"] for x in seeds.values()]))
            rows.append(row)
        r["李쎈퀎"] = rows
        res[run.upper()] = r
        outputs[run] = {"timing": tdf, "summary": pd.DataFrame(summ, columns=["??ぉ", "援щ텇", "媛?]), "windows": windows_df(rows)}

        print(f"\n===== {run.upper()} (seed {', '.join(D)}) =====")
        print("湲곌컙: " + " 쨌 ".join(f"{s} {v[0]} ~ {v[1]} ({r['??][s]:,}??" for s, v in r["湲곌컙"].items()))
        print("湲곗궗 ?섏씠 (?쒓컙, 10/25/50/75/90%): " + " | ".join(f"{k} {v}" for k, v in r["湲곗궗 ?섏씠 遺꾩쐞??(?쒓컙, 10/25/50/75/90%)"].items()))
        print("?뺣떟 湲곗궗 ?대┃ ?꾩쟻 (?몄텧 ??: " + " 쨌 ".join(f"{h}?쒓컙 {v:.0%}" for h, v in r["?뺣떟 ?대┃ ?꾩쟻 鍮꾩쑉 (?몄텧 ??1쨌3쨌6쨌12쨌24?쒓컙)"].items()))
        print(f"{'李?:>5} | {'?뺣떟0':>6} {'?ㅻ떟0':>6} | {'?멸린留?val':>9} {'test':>7} | {'?멸린+?섏씠 val':>11} {'test':>7} | {'+Transformer val':>15} {'test':>7}")
        for x in rows:
            print(f"{x['李??쒓컙)']:>4g}h | {x['?뺣떟 ?대┃ 0']:>6.2%} {x['?ㅻ떟 ?대┃ 0']:>6.2%} | {x['?멸린留?val']:>9.2%} {x['?멸린留?test']:>7.2%} | "
                  f"{x['?멸린+?섏씠 val']:>11.2%} {x['?멸린+?섏씠 test']['top1']:>7.2%} | "
                  f"{x['Transformer+?멸린+?섏씠 val ?됯퇏']:>15.2%} {x['Transformer+?멸린+?섏씠 test ?됯퇏']:>7.2%}")
        bw = max(rows, key=lambda x: x["Transformer+?멸린+?섏씠 val ?됯퇏"])
        print(f"validation ?쇰줈 怨좊Ⅸ 李?(Transformer + ?멸린 + ?섏씠, seed ?됯퇏): {bw['李??쒓컙)']:g}?쒓컙 ??test {bw['Transformer+?멸린+?섏씠 test ?됯퇏']:.2%}")

    if not res:
        sys.exit("?낅젰??李얠? 紐삵뻽?듬땲??)
    out, k = v2 / "analysis" / a.out_name, 2
    while out.exists():
        out, k = v2 / "analysis" / f"{a.out_name}_{k}", k + 1
    out.mkdir(parents=True)
    for run, o in outputs.items():
        for name, df in o.items():
            df.to_csv(out / f"{run}_{'click_timing' if name == 'timing' else name}.csv", index=False, encoding="utf-8-sig")
    (out / "result.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"\n??? {out}/ ({', '.join(sorted(p.name for p in out.iterdir()))})")


if __name__ == "__main__":
    main()
