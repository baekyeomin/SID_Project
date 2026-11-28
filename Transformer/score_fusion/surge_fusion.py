"""SURGE ?꾩씠?붿뼱 1쨌2 寃고빀 (?ы븰???놁쓬): Transformer ?먯닔???ш굔 ?멸린?꾩? 留ㅼ묶 ?먯닔瑜??뷀빐 Top-1 鍮꾧탳.

    python experiments/v2_yeomin_rerun/surge_fusion.py --runs h r --seed 42

popularity_fusion.py (紐⑤뜽 + 貫쨌?멸린 + 關쨌?섏씠) 瑜?洹몃?濡?湲곗??쇰줈 ?먭퀬, ??쓣 ?섎굹???뷀븳??

  ?먯닔 = 紐⑤뜽 + 貫쨌log(1+?먭린 ?대┃) + 關쨌log(1+?섏씠)                       ??吏湲?(A)
       + 貫e쨌log(1+?ш굔 ?대┃)                                            ???꾩씠?붿뼱 1 (B)
       + c쨌留ㅼ묶                                                         ???꾩씠?붿뼱 2 (C)
       (????= D)

?꾩씠?붿뼱 1 쨌 ?ш굔 ?멸린??  ?대┃ 湲곕줉 = train + validation behaviors ??紐⑤뱺 ?대┃ (?대┃ ?쒓컖 = 洹??몄텧??impression_time).
  ?대┃??湲곗궗??SID ?뚯씪??(c1, c2) 瑜?遺숈씠怨? (c1, c2) ???뺣젹???대┃ ?쒓컖. SID ?녿뒗 ?대┃? 嫄대꼫?대떎.
  ?꾨낫留덈떎 [?몄텧 ?쒓컖 ??24h, ?몄텧 ?쒓컖) ?덉뿉???꾨낫? (c1, c2) 媛 媛숈? ?대┃ ?섎? ?쇰떎.
  (湲곗궗 ?멸린?꾩? ?ш굔 ?멸린?꾩쓽 李?24h ??--window-hours 濡?諛붽씔?? popularity_window.py 寃곌낵 12h 媛 validation 理쒓퀬)

?꾩씠?붿뼱 2 쨌 留ㅼ묶 (鍮꾩쑉??R)
  ?됰쭏??洹??됱쓽 history 濡?w = exp(???몄텧 ?쒓컖 ???대┃ ?쒓컖)/?) 瑜?留ㅺ린怨?
  (c1, c2) 蹂?w ??첨 ?꾩껜 w ??= 鍮꾩쑉??R.
  留ㅼ묶 = R ???꾨낫 (c1, c2) 移몄씠 ?덉쑝硫?洹?媛? ?놁쑝硫?館 횞 (R ?먯꽌 c1 ??媛숈? 移몃뱾????, ?????놁쑝硫?0.
  history ??(c1, c2) ??1pos4neg ?뚯씪??history_c1/history_c2 瑜??대떎. 湲곕낯? Transformer ?낅젰怨?媛숆쾶
  留덉?留?50媛쒕쭔 ?대떎 (V1 ?ㅼ젙 history 50, ?щ? data/sequence.py ??history[-50:]). --max-history 0 ?대㈃ ?꾩껜.
  history ?쒓컖? 1pos4neg ???놁뼱??蹂듭썝?쒕떎: validation history.parquet ??(湲곗궗, ?쒓컖) 怨?  validation behaviors ???꾩옱 湲곗궗(洹??몄텧 ?쒓컖)쨌?대┃(洹??몄텧 ?쒓컖 吏곹썑)?쇰줈 (?ъ슜?? 湲곗궗) 蹂??쒓컖 紐⑸줉??留뚮뱾怨?
  history ??湲곗궗留덈떎 "?몄텧 ?쒓컖 ?댁쟾??媛??理쒓렐 ?쒓컖" ???대떎 (媛숈? 湲곗궗媛 ?щ윭 踰??덉쑝硫?紐⑤몢 媛??理쒓렐 ?쒓컖).
  ?쒓컖??紐?李얠? history ??ぉ? w = 0 ?쇰줈 ?먭퀬 洹?鍮꾩쑉??異쒕젰?쒕떎.

媛以묒튂 怨좊Ⅴ湲? validation ???덈컲(val_candidate_scores.parquet)?먯꽌 Top-1 ??媛???믪? 議고빀, test ?먮뒗 ??踰덈쭔 ?곸슜.
  媛숈? Top-1 ?대㈃ 媛以묒튂媛 ?묒? 履? D ??(貫, 關) 瑜?A ??理쒖쟻媛?二쇰? 3횞3 ?쇰줈留?李얜뒗??(寃⑹옄 ?ш린 ?뚮Ц).

?낅젰
  ?덉륫   <V2>/transformer/<run>_seed<seed>/{val,test}_candidate_scores.parquet   (5_predict_val.sh, 4_transformer.sh)
  1pos4neg  v1 : /home/ubuntu/shared/datasets/ebnerd/{validation_sequences_1pos4neg_half,test_sequences_1pos4neg}.parquet
            h,r: <PRE>/experiments/v2_<run>/split/ 媛숈? ?뚯씪 ?대쫫
  SID    v1 : ~/minji/uni_post_rqvae/article_semantic_ids.parquet,  h,r: <V2>/sid/<run>/article_semantic_ids.parquet
  ?먮낯   <raw>/{train,validation}/behaviors.parquet, validation/history.parquet, articles.parquet
寃곌낵: <V2>/analysis/surge_fusion_seed<seed>.json, .csv (?대? ?덉쑝硫?硫덉땄). --max-history 瑜?諛붽씀硫??대쫫 ?앹뿉 _hist<N>/_histall,
  --level-weights 硫?G0 (?덈꺼 媛以묒튂 + ?멸린 + ?섏씠), G (+ ?꾩씠?붿뼱 2) 異붽?, ?대쫫 ?앹뿉 _lvl.
  --idea1-variants 硫??꾩씠?붿뼱 1 媛쒖꽑 (???먭린 ?대┃ 類 ?ш굔 ?멸린, ???ш굔 ?멸린 횞 留ㅼ묶): B1 (A ??, F1 쨌 F4 쨌 F14 (C ??, ?대쫫 ?앹뿉 _i1v.
  --split-history 硫?E (理쒓렐 N媛?留ㅼ묶 + 洹??댁쟾 history 留ㅼ묶, 媛以묒튂 ?곕줈) 瑜??뷀븯怨??대쫫 ?앹뿉 _split.
  --wide-grid 硫?_wide (? 336h쨌720h쨌?? 館 0.5쨌0.75쨌1, c 32쨌64쨌128 異붽?)
"""
import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_test import REPO  # noqa: E402
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402

LAMBDA_E = [0, 0.05, 0.1, 0.2, 0.3, 0.5, 1, 2, 4]
C_MATCH = [0, 0.1, 0.25, 0.5, 1, 2, 4, 8, 16]
TAUS = [24.0, 48.0, 96.0, 168.0]       # ?쒓컙
NUS = [0, 0.1, 0.2, 0.3]
# --wide-grid: 吏??寃곌낵(? 168h, 館 0.3, c 16)媛 紐⑤몢 寃⑹옄 ?앹씠??諛붽묑履??꾨낫瑜??뷀븳?? ? = inf ??媛먯뇿 ?놁쓬 (w = 1).
WIDE = {"TAUS": [336.0, 720.0, float("inf")], "NUS": [0.5, 0.75, 1.0], "C_MATCH": [32, 64, 128]}
WINDOW_H = 24.0
FILES = {"val": "validation_sequences_1pos4neg_half.parquet", "test": "test_sequences_1pos4neg.parquet"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def to_s(x) -> np.ndarray:
    return np.asarray(x).astype("datetime64[s]").astype(np.int64)


def paths(run: str, pre: Path, v2: Path, exp_prefix: str, seed: str):
    if run == "v1":
        seq = Path("/home/ubuntu/shared/datasets/ebnerd")
        sid = Path("~/minji/uni_post_rqvae/article_semantic_ids.parquet").expanduser()
    else:
        seq = pre / "experiments" / f"{exp_prefix}{run}" / "split"
        sid = v2 / "sid" / run / "article_semantic_ids.parquet"
    tdir = v2 / "transformer" / f"{run}_seed{seed}"
    return {"seq": {k: seq / f for k, f in FILES.items()}, "sid": sid,
            "pred": {"val": tdir / "val_candidate_scores.parquet", "test": tdir / "test_candidate_scores.parquet"}}


# ---------------------------------------------------------------- 怨듯넻 ?щ즺 (run 怨?臾닿?)
def load_click_log(raw: Path) -> pd.DataFrame:
    parts = []
    for split in ["train", "validation"]:
        b = pd.read_parquet(raw / split / "behaviors.parquet", columns=["impression_time", "article_ids_clicked"])
        b = b.explode("article_ids_clicked").dropna(subset=["article_ids_clicked"])
        parts.append(pd.DataFrame({"a": b.article_ids_clicked.astype(np.int64).to_numpy(), "t": to_s(b.impression_time)}))
    return pd.concat(parts, ignore_index=True)


def sorted_times_by(keys: np.ndarray, t: np.ndarray) -> dict:
    o = np.lexsort((t, keys))
    k, t = keys[o], t[o]
    cut = np.flatnonzero(np.diff(k)) + 1
    return {int(kk): tt for kk, tt in zip(k[np.r_[0, cut]], np.split(t, cut))}


def count_window(d: dict, keys: np.ndarray, t: np.ndarray, window_s: int) -> np.ndarray:
    out = np.zeros(len(keys), dtype=np.int64)
    for i, (k, ti) in enumerate(zip(keys, t)):
        arr = d.get(int(k))
        if arr is not None:
            out[i] = np.searchsorted(arr, ti, side="left") - np.searchsorted(arr, ti - window_s, side="left")
    return out


class HistoryTimes:
    """(?ъ슜?? 湲곗궗) ??洹??ъ슜?먭? 洹?湲곗궗瑜?history ???ｌ? ?쒓컖?? query ???몄텧 ?쒓컖 ?댁쟾??媛??理쒓렐 ?쒓컖."""

    def __init__(self, raw: Path):
        h = pd.read_parquet(raw / "validation" / "history.parquet", columns=["user_id", "article_id_fixed", "impression_time_fixed"])
        h = h.dropna(subset=["user_id", "article_id_fixed", "impression_time_fixed"])
        h = h.explode(["article_id_fixed", "impression_time_fixed"]).dropna()
        u = [h.user_id.astype(np.int64).to_numpy()]
        a = [h.article_id_fixed.astype(np.int64).to_numpy()]
        t = [to_s(pd.to_datetime(h.impression_time_fixed))]
        b = pd.read_parquet(raw / "validation" / "behaviors.parquet", columns=["user_id", "impression_time", "article_id", "article_ids_clicked"])
        cur = b.dropna(subset=["article_id"])
        u.append(cur.user_id.astype(np.int64).to_numpy()); a.append(cur.article_id.astype(np.int64).to_numpy()); t.append(to_s(cur.impression_time))
        clk = b.explode("article_ids_clicked").dropna(subset=["article_ids_clicked"])
        # ?대┃? 洹??몄텧??sample ??留뚮뱺 ?ㅼ뿉 history ???ㅼ뼱媛誘濡? ?몄텧 ?쒓컖 + 1珥덈줈 ?붾떎 (媛숈? ?몄텧?먮뒗 ??蹂댁엫)
        u.append(clk.user_id.astype(np.int64).to_numpy()); a.append(clk.article_ids_clicked.astype(np.int64).to_numpy()); t.append(to_s(clk.impression_time) + 1)
        u, a, t = np.concatenate(u), np.concatenate(a), np.concatenate(t)
        self.t0 = int(t.min()) - 1
        pair = u * 100_000_000 + a                                  # article_id < 1e8
        self.uniq, idx = np.unique(pair, return_inverse=True)
        comp = idx.astype(np.int64) * (1 << 32) + (t - self.t0)
        self.comp = np.sort(comp)

    def query(self, user: np.ndarray, art: np.ndarray, T: np.ndarray) -> np.ndarray:
        """?쒓컖(珥? 諛곗뿴, 紐?李얠쑝硫?-1."""
        pair = user * 100_000_000 + art
        pos = np.searchsorted(self.uniq, pair)
        pos_c = np.minimum(pos, len(self.uniq) - 1)
        ok = self.uniq[pos_c] == pair
        q = pos_c.astype(np.int64) * (1 << 32) + (T - self.t0)
        j = np.searchsorted(self.comp, q, side="right") - 1
        j_c = np.maximum(j, 0)
        ok &= (j >= 0) & ((self.comp[j_c] >> 32) == pos_c)
        out = np.where(ok, (self.comp[j_c] & ((1 << 32) - 1)) + self.t0, -1)
        return out


# ---------------------------------------------------------------- run 蹂?LEVEL_COLS = ["c1_log_prob", "c2_log_prob", "c3_log_prob"]


def load_split(pred_path: Path, seq_path: Path, levels: bool = False) -> tuple:
    p = pd.read_parquet(pred_path, columns=["sample_index", "impression_id", "user_id", "impression_time", "article_id", "c1", "c2", "label", "candidate_score"]
                        + (LEVEL_COLS if levels else []))
    p = p.sort_values(["sample_index"], kind="stable").reset_index(drop=True)
    n = p.sample_index.nunique()
    if len(p) != 5 * n:
        raise ValueError(f"{pred_path.name}: ?됰쭏???꾨낫 5媛쒓? ?꾨떃?덈떎")
    seq = pq.read_table(seq_path, columns=["impression_id", "user_id", "impression_time", "candidate_article_ids",
                                           "history_article_ids", "history_c1", "history_c2"])
    if seq.num_rows != n or p.sample_index.max() != n - 1:
        raise ValueError(f"{seq_path.name} ????{seq.num_rows} ???덉륫 sample ??{n}")
    # sample_index = 1pos4neg ??踰덊샇 ?몄? ?뺤씤 (predict_sid.py ??shuffle=False)
    cand = pc.list_flatten(seq.column("candidate_article_ids").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    if not np.array_equal(cand, p.article_id.astype(np.int64).to_numpy()):
        raise ValueError(f"{pred_path.name}: ?꾨낫 湲곗궗 ?쒖꽌媛 {seq_path.name} 怨??ㅻ쫭?덈떎 (sample_index ?뺣젹 ?뺤씤)")
    imp = to_s(seq.column("impression_time").to_numpy())
    if not np.array_equal(np.repeat(imp, 5), to_s(pd.to_datetime(p.impression_time))):
        raise ValueError("?몄텧 ?쒓컖???ㅻ쫭?덈떎")
    p["article_id"] = p.article_id.astype(np.int64)
    p["label"] = p.label.astype(int)
    p["t"] = np.repeat(imp, 5)
    return p, seq


def match_features(p: pd.DataFrame, seq, ht: HistoryTimes, taus: list, max_history: int = 50, older_only: bool = False) -> dict:
    """? 留덈떎 ?꾨낫蹂?(?먭린 移?媛? ?먭린 移??덉쓬, 媛숈? c1 移???. ?쒓컖 紐?李얠? 鍮꾩쑉??諛섑솚.
    max_history > 0 ?대㈃ ?됰쭏??history ??留덉?留?max_history 媛쒕쭔 ?대떎 (Transformer ?낅젰怨?媛숆쾶).
    older_only ?대㈃ 諛섎?濡?留덉?留?max_history 媛쒕? 類 ?섎㉧吏 (洹몃낫???댁쟾 ?대┃) 留??대떎."""
    lens = pc.list_value_length(seq.column("history_c1").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    row = np.repeat(np.arange(seq.num_rows), lens)
    hc1 = pc.list_flatten(seq.column("history_c1").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    hc2 = pc.list_flatten(seq.column("history_c2").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    ha = pc.list_flatten(seq.column("history_article_ids").combine_chunks()).to_numpy(zero_copy_only=False).astype(np.int64)
    if not (len(hc1) == len(hc2) == len(ha)):
        raise ValueError("history_c1 / history_c2 / history_article_ids 湲몄씠媛 ?ㅻ쫭?덈떎")
    if max_history > 0:
        start = np.repeat(np.cumsum(lens) - lens, lens)            # ?됱쓽 泥??먯냼 ?꾩튂
        from_end = np.repeat(lens, lens) - (np.arange(len(row)) - start)   # ?ㅼ뿉??紐?踰덉㎏ (1 = 留덉?留?
        keep = (from_end > max_history) if older_only else (from_end <= max_history)
        row, hc1, hc2, ha = row[keep], hc1[keep], hc2[keep], ha[keep]
        lens = np.maximum(lens - max_history, 0) if older_only else np.minimum(lens, max_history)
    user = seq.column("user_id").to_numpy().astype(np.int64)[row]
    T = to_s(seq.column("impression_time").to_numpy())[row]
    ht_s = ht.query(user, ha, T)
    found = ht_s >= 0
    hours = np.where(found, (T - ht_s) / 3600.0, np.inf)
    ev = hc1 * 100_000 + hc2
    crow = np.repeat(np.arange(seq.num_rows), 5)
    cev = p.c1.to_numpy().astype(np.int64) * 100_000 + p.c2.to_numpy().astype(np.int64)
    cc1 = p.c1.to_numpy().astype(np.int64)
    out = {"history_items": int(len(row)), "history_time_missing": float(1 - found.mean()) if len(row) else 0.0,
           "rows_without_history": int((lens == 0).sum())}
    for tau in taus:
        with np.errstate(invalid="ignore"):
            w = np.where(found, np.exp(-hours / tau), 0.0)   # ? = inf ???쒓컖 李얠? ??ぉ? w = 1
        h = pd.DataFrame({"row": row, "ev": ev, "c1": hc1, "w": w})
        tot = h.groupby("row").w.sum()
        cell = h.groupby(["row", "ev"]).w.sum()
        c1sum = h.groupby(["row", "c1"]).w.sum()
        denom = tot.reindex(crow).to_numpy()
        own_w = cell.reindex(pd.MultiIndex.from_arrays([crow, cev])).to_numpy()
        c1_w = c1sum.reindex(pd.MultiIndex.from_arrays([crow, cc1])).to_numpy()
        has = ~np.isnan(own_w)
        with np.errstate(invalid="ignore", divide="ignore"):
            own = np.where(has & (denom > 0), own_w / denom, 0.0)
            c1s = np.where(~np.isnan(c1_w) & (denom > 0), c1_w / denom, 0.0)
        out[tau] = (own, has, c1s)
    return out


def matching(f: tuple, nu: float) -> np.ndarray:
    own, has, c1s = f
    return np.where(has, own, nu * c1s)


def top1_fast(S: np.ndarray, Y: np.ndarray) -> float:
    S = S.reshape(-1, 5); Y = Y.reshape(-1, 5)
    top = S == S.max(axis=1, keepdims=True)
    return float(((top & (Y == 1)).sum(1) / top.sum(1)).mean())


def pick(grid: dict):
    return max(grid, key=lambda k: (round(grid[k], 10), [-abs(x) for x in k]))


def edge_warn(name: str, best: tuple, lists: list) -> list:
    """怨좊Ⅸ 媛믪씠 寃⑹옄 留??앹씠硫??뚮┝ (寃⑹옄瑜??볧? ?ㅼ떆 蹂?寃?."""
    return [f"{name}: {v} 媛 寃⑹옄 ??({L[0]}..{L[-1]})" for v, L in zip(best, lists) if len(L) > 1 and v in (L[0], L[-1]) and v != 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=["v1", "h", "r"])
    ap.add_argument("--seed", default="42")
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="湲곕낯 <pre>/v2")
    ap.add_argument("--raw", default=str(REPO / "data/raw"))
    ap.add_argument("--exp-prefix", default="v2_")
    ap.add_argument("--max-history", type=int, default=50, help="留ㅼ묶????history 留덉?留?N媛?(Transformer V1 = 50). 0 = ?꾩껜")
    ap.add_argument("--window-hours", type=float, default=WINDOW_H, help="湲곗궗 쨌 ?ш굔 ?멸린?꾨? ? 吏곸쟾 ?쒓컙 (湲곕낯 24)")
    ap.add_argument("--level-weights", action="store_true",
                    help="?덈꺼 媛以묒튂: 紐⑤뜽 ?먯닔 ???L1 + a2쨌L2 + a3쨌L3 (L = c1 쨌 c2 쨌 c3 log ?뺣쪧). G0 = + ?멸린 + ?섏씠, G = + ?멸린 + ?섏씠 + ?꾩씠?붿뼱 2")
    ap.add_argument("--idea1-variants", action="store_true",
                    help="?꾩씠?붿뼱 1 媛쒖꽑: ???먭린 ?대┃ 類 ?ш굔 ?멸린?? ???ш굔 ?멸린??횞 留ㅼ묶. A ??(B1) ? C ??(F1 쨌 F4 쨌 F14) ???뷀빐 遊?)
    ap.add_argument("--split-history", action="store_true",
                    help="E 異붽?: 理쒓렐 --max-history 媛?留ㅼ묶怨?洹??댁쟾 history 留ㅼ묶???곕줈 ?뷀븿 (媛以묒튂 ?곕줈)")
    ap.add_argument("--wide-grid", action="store_true", help="?, 館, c ?꾨낫瑜?諛붽묑履쎌쑝濡??볧옒 (湲곕낯 寃⑹옄??洹몃?濡??ы븿)")
    a = ap.parse_args()
    if a.split_history and a.max_history <= 0:
        sys.exit("--split-history ??--max-history > 0 ???뚮쭔 (理쒓렐 N媛?/ 洹??댁쟾?쇰줈 ?섎닎)")
    global TAUS, NUS, C_MATCH
    if a.wide_grid:
        TAUS, NUS, C_MATCH = TAUS + WIDE["TAUS"], NUS + WIDE["NUS"], C_MATCH + WIDE["C_MATCH"]
    pre = Path(a.pre).expanduser()
    v2 = Path(a.v2).expanduser() if a.v2 else pre / "v2"
    raw = Path(a.raw).expanduser()
    out_dir = v2 / "analysis"
    tag = f"seed{a.seed}" + ("" if a.max_history == 50 else f"_hist{a.max_history or 'all'}") + ("_wide" if a.wide_grid else "") + ("" if a.window_hours == WINDOW_H else f"_w{a.window_hours:g}") + ("_split" if a.split_history else "") + ("_i1v" if a.idea1_variants else "") + ("_lvl" if a.level_weights else "")
    jpath, cpath = out_dir / f"surge_fusion_{tag}.json", out_dir / f"surge_fusion_{tag}.csv"
    if jpath.exists() or cpath.exists():
        sys.exit(f"硫덉땄: {jpath} 媛 ?대? ?덉뒿?덈떎 (??뼱?곗? ?딆쓬)")
    window_s = int(a.window_hours * 3600)

    log("?대┃ 湲곕줉 ?쎈뒗 以?)
    clicks = load_click_log(raw)
    by_art = sorted_times_by(clicks.a.to_numpy(), clicks.t.to_numpy())
    arts = pd.read_parquet(raw / "articles.parquet", columns=["article_id", "published_time"])
    pub = dict(zip(arts.article_id.astype(np.int64), to_s(arts.published_time)))
    log("history ?쒓컖 紐⑸줉 留뚮뱶??以?)
    ht = HistoryTimes(raw)

    results, rows = [], []
    for run in a.runs:
        P = paths(run, pre, v2, a.exp_prefix, a.seed)
        need = [P["sid"], *P["seq"].values(), *P["pred"].values()]
        miss = [str(x) for x in need if not x.exists()]
        if miss:
            log(f"[{run}] ?낅젰 ?놁쓬 ??嫄대꼫?: {miss}")
            continue
        log(f"[{run}] SID쨌?ш굔 ?대┃ 以鍮?)
        sid = pd.read_parquet(P["sid"], columns=["article_id", "c1", "c2"])
        a2e = dict(zip(sid.article_id.astype(np.int64), sid.c1.astype(np.int64) * 100_000 + sid.c2.astype(np.int64)))
        cev = clicks.a.map(a2e)
        skipped = float(cev.isna().mean())
        ok = cev.notna().to_numpy()
        by_ev = sorted_times_by(cev[ok].astype(np.int64).to_numpy(), clicks.t.to_numpy()[ok])

        D, OLD, info = {}, {}, {"clicks_without_sid": skipped}
        for split in ["val", "test"]:
            log(f"[{run}] {split} ?쎄린쨌?뱀쭠 怨꾩궛")
            p, seq = load_split(P["pred"][split], P["seq"][split], levels=a.level_weights)
            aid, t = p.article_id.to_numpy(), p.t.to_numpy()
            p["logpop"] = np.log1p(count_window(by_art, aid, t, window_s))
            pt = np.array([pub.get(x, np.nan) for x in aid], dtype="float64")
            p["logage"] = np.log1p(np.nan_to_num(np.clip((t - pt) / 3600.0, 0, None), nan=0.0))
            ev = p.c1.to_numpy().astype(np.int64) * 100_000 + p.c2.to_numpy().astype(np.int64)
            p["logepop"] = np.log1p(count_window(by_ev, ev, t, window_s))
            mf = match_features(p, seq, ht, TAUS, a.max_history)
            info[split] = {k: mf[k] for k in ("history_items", "history_time_missing", "rows_without_history")}
            pos = p.label.to_numpy() == 1
            f48 = mf[48.0]
            info[split]["?뺣떟 ?꾨낫 ?ш굔??R ???덉쓬"] = float(f48[1][pos].mean())
            info[split]["?ㅻ떟 ?꾨낫 ?ш굔??R ???덉쓬"] = float(f48[1][~pos].mean())
            info[split]["?뺣떟 ?먭린 ?대┃ < 10"] = float((np.expm1(p.logpop.to_numpy()[pos]) < 10).mean())
            D[split] = (p, mf)
            if a.split_history:
                OLD[split] = match_features(p, seq, ht, TAUS, a.max_history, older_only=True)
                info[split]["?댁쟾 history ??ぉ"] = OLD[split]["history_items"]
                info[split]["?댁쟾 history ?녿뒗 ??] = OLD[split]["rows_without_history"]

        (pv, mv), (pt_, mt) = D["val"], D["test"]
        Yv, Yt = pv.label.to_numpy(), pt_.label.to_numpy()
        sv, st = pv.candidate_score.to_numpy(), pt_.candidate_score.to_numpy()
        V = {k: pv[k].to_numpy() for k in ("logpop", "logage", "logepop")}
        Tt = {k: pt_[k].to_numpy() for k in ("logpop", "logage", "logepop")}

        log(f"[{run}] 媛以묒튂 怨좊Ⅴ湲?(validation)")
        gA = {(l, m): top1_fast(sv + l * V["logpop"] + m * V["logage"], Yv) for l in LAMBDAS for m in MUS}
        bA = pick(gA)
        gB = {(l, m, le): top1_fast(sv + l * V["logpop"] + m * V["logage"] + le * V["logepop"], Yv)
              for l in LAMBDAS for m in MUS for le in LAMBDA_E}
        bB = pick(gB)
        Mv = {(tau, nu): matching(mv[tau], nu) for tau in TAUS for nu in NUS}
        Mt = {(tau, nu): matching(mt[tau], nu) for tau in TAUS for nu in NUS}
        gC = {}
        for l, m in itertools.product(LAMBDAS, MUS):
            base = sv + l * V["logpop"] + m * V["logage"]
            for (tau, nu), M in Mv.items():
                for c in C_MATCH:
                    gC[(l, m, tau, nu, c)] = top1_fast(base + c * M, Yv)
        bC = pick(gC)
        near = lambda L, x: L[max(0, L.index(x) - 1): L.index(x) + 2]
        gD = {}
        for l, m in itertools.product(near(LAMBDAS, bA[0]), near(MUS, bA[1])):
            base = sv + l * V["logpop"] + m * V["logage"]
            for le in LAMBDA_E:
                b2 = base + le * V["logepop"]
                for (tau, nu), M in Mv.items():
                    for c in C_MATCH:
                        gD[(l, m, le, tau, nu, c)] = top1_fast(b2 + c * M, Yv)
        bD = pick(gD)
        if a.level_weights:   # level_weight_fusion.py ? 媛숈? a2 쨌 a3 寃⑹옄
            A2 = [0, 0.1, 0.25, 0.5, 0.75, 1, 1.5]
            A3 = [0, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
            Lv = [pv[c].to_numpy() for c in LEVEL_COLS]
            Lt = [pt_[c].to_numpy() for c in LEVEL_COLS]
            gap = float(np.abs(Lv[0] + Lv[1] + Lv[2] - sv).max())
            if gap > 1e-3:
                raise ValueError(f"[{run}] c1+c2+c3 log ?뺣쪧 ?⑹씠 candidate_score ? ?ㅻ쫭?덈떎 (理쒕? 李⑥씠 {gap:.4g})")
            lvl_v = lambda a2, a3: Lv[0] + a2 * Lv[1] + a3 * Lv[2]
            lvl_t = lambda a2, a3: Lt[0] + a2 * Lt[1] + a3 * Lt[2]
            lC, mC, tC, nC, cC = bC
            gG0 = {(a2, a3, l, m): top1_fast(lvl_v(a2, a3) + l * V["logpop"] + m * V["logage"], Yv)
                   for a2 in A2 for a3 in A3 for l in near(LAMBDAS, bA[0]) for m in near(MUS, bA[1])}
            bG0 = pick(gG0)
            gG = {(a2, a3, l, m, tC, nC, c): top1_fast(lvl_v(a2, a3) + l * V["logpop"] + m * V["logage"] + c * Mv[(tC, nC)], Yv)
                  for a2 in A2 for a3 in A3 for l in near(LAMBDAS, lC) for m in near(MUS, mC) for c in near(C_MATCH, cC)}
            bG = pick(gG)
        if a.idea1_variants:
            # ??媛숈? ?ш굔 "?ㅻⅨ" 湲곗궗?ㅼ쓽 ?멸린 = ?ш굔 ?대┃ ???먭린 ?대┃ (媛숈? 李?
            def other(P):
                return np.log1p(np.clip(np.expm1(P["logepop"]) - np.expm1(P["logpop"]), 0, None))
            V["logother"], Tt["logother"] = other(V), other(Tt)
            gB1 = {(l, m, le): top1_fast(sv + l * V["logpop"] + m * V["logage"] + le * V["logother"], Yv)
                   for l in LAMBDAS for m in MUS for le in LAMBDA_E}
            bB1 = pick(gB1)
            # C ?꾩뿉: C ??貫, 關, ?, 館, c 瑜??먭퀬 ????쓽 媛以묒튂留?怨좊Ⅸ??            lC, mC, tC, nC, cC = bC
            baseC_v = sv + lC * V["logpop"] + mC * V["logage"] + cC * Mv[(tC, nC)]
            baseC_t = st + lC * Tt["logpop"] + mC * Tt["logage"] + cC * Mt[(tC, nC)]
            NEW = {"F1": ("???ㅻⅨ 湲곗궗 ?ш굔 ?멸린", V["logother"], Tt["logother"]),
                   "F4": ("???ш굔 ?멸린 횞 留ㅼ묶", V["logepop"] * Mv[(tC, nC)], Tt["logepop"] * Mt[(tC, nC)]),
                   "F14": ("?졻몿 ?ㅻⅨ 湲곗궗 ?ш굔 ?멸린 횞 留ㅼ묶", V["logother"] * Mv[(tC, nC)], Tt["logother"] * Mt[(tC, nC)])}
            GF = {}
            for k, (_, xv, _x) in NEW.items():
                g = {(w,): top1_fast(baseC_v + w * xv, Yv) for w in LAMBDA_E + [8, 16]}
                GF[k] = (g, pick(g))
        if a.split_history:   # E: C ??貫, 關 瑜??먭퀬 理쒓렐 N媛?留ㅼ묶 (c1) + 洹??댁쟾 history 留ㅼ묶 (c2) ???곕줈 怨좊Ⅸ??            Ov = {(tau, nu): matching(OLD["val"][tau], nu) for tau in TAUS for nu in NUS}
            Ot = {(tau, nu): matching(OLD["test"][tau], nu) for tau in TAUS for nu in NUS}
            l, m = bC[0], bC[1]
            base = sv + l * V["logpop"] + m * V["logage"]
            gE = {}
            for (tau, nu), M in Mv.items():
                for c1 in C_MATCH:
                    b1 = base + c1 * M
                    for c2 in C_MATCH:
                        gE[(l, m, tau, nu, c1, c2)] = top1_fast(b1 + c2 * Ov[(tau, nu)], Yv)
            bE = pick(gE)

        l, m = bA
        test = {
            "紐⑤뜽留?: metrics(pt_, st),
            f"A 吏湲?(貫={l}, 關={m})": metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"]),
        }
        l, m, le = bB
        test[f"B +?꾩씠?붿뼱1 (貫={l}, 關={m}, 貫e={le})"] = metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"] + le * Tt["logepop"])
        l, m, tau, nu, c = bC
        test[f"C +?꾩씠?붿뼱2 (貫={l}, 關={m}, ?={tau:g}h, 館={nu}, c={c})"] = metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"] + c * Mt[(tau, nu)])
        l, m, le, tau, nu, c = bD
        test[f"D +????(貫={l}, 關={m}, 貫e={le}, ?={tau:g}h, 館={nu}, c={c})"] = metrics(
            pt_, st + l * Tt["logpop"] + m * Tt["logage"] + le * Tt["logepop"] + c * Mt[(tau, nu)])
        # 李멸퀬: 媛????섎굹留?        if a.split_history:
            l, m, tau, nu, c1, c2 = bE
            test[f"E 理쒓렐{a.max_history}+?댁쟾 留ㅼ묶 (貫={l}, 關={m}, ?={tau:g}h, 館={nu}, c理쒓렐={c1}, c?댁쟾={c2})"] = metrics(
                pt_, st + l * Tt["logpop"] + m * Tt["logage"] + c1 * Mt[(tau, nu)] + c2 * Ot[(tau, nu)])
        if a.level_weights:
            a2, a3, l, m = bG0
            test[f"G0 ?덈꺼+?멸린+?섏씠 (a2={a2}, a3={a3}, 貫={l}, 關={m})"] = metrics(pt_, lvl_t(a2, a3) + l * Tt["logpop"] + m * Tt["logage"])
            a2, a3, l, m, tau, nu, c = bG
            test[f"G ?덈꺼+?멸린+?섏씠+?꾩씠?붿뼱2 (a2={a2}, a3={a3}, 貫={l}, 關={m}, ?={tau:g}h, 館={nu}, c={c})"] = metrics(
                pt_, lvl_t(a2, a3) + l * Tt["logpop"] + m * Tt["logage"] + c * Mt[(tau, nu)])
        if a.idea1_variants:
            l, m, le = bB1
            test[f"B1 A+???ㅻⅨ 湲곗궗 ?ш굔 ?멸린 (貫={l}, 關={m}, 貫o={le})"] = metrics(pt_, st + l * Tt["logpop"] + m * Tt["logage"] + le * Tt["logother"])
            for k, (name, _xv, xt) in NEW.items():
                (w,) = GF[k][1]
                test[f"{k} C+{name} (媛以묒튂={w})"] = metrics(pt_, baseC_t + w * xt)
        test["李멸퀬: ?ш굔 ?멸린?꾨쭔"] = metrics(pt_, Tt["logepop"])
        test["李멸퀬: 留ㅼ묶留?(?=48h, 館=0.2)"] = metrics(pt_, Mt[(48.0, 0.2)])

        warn = (edge_warn("A", bA, [LAMBDAS, MUS]) + edge_warn("B", bB, [LAMBDAS, MUS, LAMBDA_E])
                + edge_warn("C", bC, [LAMBDAS, MUS, TAUS, NUS, C_MATCH]) + edge_warn("D", bD[2:], [LAMBDA_E, TAUS, NUS, C_MATCH])
                + (edge_warn("E", bE[2:], [TAUS, NUS, C_MATCH, C_MATCH]) if a.split_history else []))
        r = {"run": run, "seed": a.seed, "max_history": a.max_history, "wide_grid": a.wide_grid, "window_hours": a.window_hours,
             "grid": {"TAUS": TAUS, "NUS": NUS, "C_MATCH": C_MATCH}, "info": info, "grid_edge_warnings": warn,
             "val_top1": {"A": gA[bA], "B": gB[bB], "C": gC[bC], "D": gD[bD], **({"E": gE[bE]} if a.split_history else {}),
                          **({"B1": gB1[bB1], **{k: g[b] for k, (g, b) in GF.items()}} if a.idea1_variants else {}),
                          **({"G0": gG0[bG0], "G": gG[bG]} if a.level_weights else {})},
             "best": {"A": bA, "B": bB, "C": bC, "D": bD, **({"E": bE} if a.split_history else {}),
                      **({"B1": bB1, **{k: b for k, (g, b) in GF.items()}} if a.idea1_variants else {}),
                      **({"G0": bG0, "G": bG} if a.level_weights else {})}, "split_history": a.split_history, "test": test}
        results.append(r)
        print(f"\n===== {run.upper()} (seed {a.seed}, 留ㅼ묶 history {'?꾩껜' if a.max_history == 0 else f'留덉?留?{a.max_history}媛?}, ?멸린??李?{a.window_hours:g}?쒓컙) =====")
        print(f"SID ?놁뼱 嫄대꼫???대┃ {skipped:.2%} 쨌 " + " 쨌 ".join(
            f"{s}: history ?쒓컖 紐?李얠쓬 {info[s]['history_time_missing']:.2%}, ?꾨낫 ?ш굔??R ???덉쓬 ?뺣떟 {info[s]['?뺣떟 ?꾨낫 ?ш굔??R ???덉쓬']:.1%} / ?ㅻ떟 {info[s]['?ㅻ떟 ?꾨낫 ?ш굔??R ???덉쓬']:.1%}"
            for s in ("val", "test")))
        for w_ in warn:
            print("  二쇱쓽 " + w_)
        print("validation Top-1: " + " 쨌 ".join(f"{k} {v:.2%}" for k, v in r["val_top1"].items()))
        for name, mm in test.items():
            print(f"  test {name:<58} Top-1 {mm['top1']:.2%} 쨌 AUC {mm['auc']:.4f} 쨌 MRR {mm['mrr']:.4f} 쨌 nDCG@5 {mm['ndcg@5']:.4f}")
            rows.append({"run": run, "seed": a.seed, "諛⑹떇": name, **mm})

    if results:
        out_dir.mkdir(parents=True, exist_ok=True)
        jpath.write_text(json.dumps(results, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        pd.DataFrame(rows).to_csv(cpath, index=False)
        log(f"??? {jpath}, {cpath}")


if __name__ == "__main__":
    main()
