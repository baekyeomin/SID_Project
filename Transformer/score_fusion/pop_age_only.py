"""SID(Transformer) ?놁씠 ?멸린 + ?섏씠留뚯쑝濡?梨꾩젏 (?ы븰???놁쓬). SID媛 ?쒖닔?섍쾶 蹂댄깭??紐レ쓣 ?ш린 ?꾪븳 湲곗???

    python experiments/v2_yeomin_rerun/pop_age_only.py                 # H 쨌 R 쨌 V1 以??덈뒗 寃?    python experiments/v2_yeomin_rerun/pop_age_only.py --runs h

?먯닔 = 貫 쨌 logpop + 關 쨌 logage   (Transformer ?먯닔 s ?놁쓬)
  貫, 關 ?꾨낫? 怨좊Ⅴ??洹쒖튃? popularity_fusion.py --with-age ? 媛숇떎 (貫 13媛?횞 關 10媛? 貫 = 0 ? 鍮쇱꽌 120媛?.
  validation ???덈컲 Top-1 理쒕? (媛숈쑝硫?|貫|, |關| ?묒? 履? ??test ????踰? 吏???뺤쓽??媛숇떎 (?숈젏 湲곕?媛?.
  SID ?먯닔媛 ?놁쑝誘濡?貫? 關??鍮꾩쑉留??섎?媛 ?덈떎.

?낅젰 (?쎄린留?
  <V2>/features/<run>/{validation_sequences_1pos4neg_half,test_sequences_1pos4neg}.parquet  (add_candidate_features.py 寃곌낵)
  <V2>/analysis/popularity_fusion_seed{42,123,2026}.json  (?덉쑝硫? SID + ?멸린 + ?섏씠 ? 鍮꾧탳??SID???쒖닔??紐?怨꾩궛)
寃곌낵: ?붾㈃ + <V2>/analysis/pop_age_only.json (?대? ?덉쑝硫?_2, _3 ??
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from popularity_fusion import LAMBDAS, MUS, metrics  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
FILES = {"validation": "validation_sequences_1pos4neg_half.parquet", "test": "test_sequences_1pos4neg.parquet"}


def load(path: Path):
    t = pq.read_table(path, columns=["candidate_labels", "candidate_logpop", "candidate_logage"])
    y = np.array(t.column("candidate_labels").to_pylist(), dtype=np.int64)
    lp = np.array(t.column("candidate_logpop").to_pylist(), dtype=np.float64)
    la = np.array(t.column("candidate_logage").to_pylist(), dtype=np.float64)
    if y.shape[1] != 5 or not (y.sum(1) == 1).all():
        raise ValueError(f"{path}: ?됰쭏???꾨낫 5媛?쨌 ?뺣떟 1媛쒖뿬???⑸땲??)
    return y, lp, la


def top1(s, y):
    pos = (s * y).sum(1, keepdims=True)
    neg = y == 0
    g = ((s > pos) & neg).sum(1)
    t = ((s == pos) & neg).sum(1)
    return float(np.where(g == 0, 1.0 / (1 + t), 0.0).mean())


def full(s, y):
    return metrics(pd.DataFrame({"label": y.reshape(-1)}), s.reshape(-1))


def fusion_top1(v2: Path, run: str) -> dict:
    """popularity_fusion_seed*.json ??SID + ?멸린 + ?섏씠 test 寃곌낵 (seed蹂?."""
    out = {}
    for f in sorted((v2 / "analysis").glob("popularity_fusion_seed*.json")):
        seed = f.stem.replace("popularity_fusion_seed", "")
        if not seed.isdigit():
            continue
        for r in json.loads(f.read_text(encoding="utf-8")):
            if r.get("run") == run:
                for k, m in r["test"].items():
                    if k.startswith("寃고빀+?섏씠"):
                        out[seed] = m
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pre", default=str(REPO / "data/output/fixedcluster_v1"))
    ap.add_argument("--v2", default=None, help="湲곕낯 <pre>/v2")
    ap.add_argument("--runs", nargs="+", default=["h", "r", "v1"])
    a = ap.parse_args()
    v2 = Path(a.v2).expanduser() if a.v2 else Path(a.pre).expanduser() / "v2"
    res = {}
    for run in a.runs:
        d = v2 / "features" / run
        if not all((d / f).exists() for f in FILES.values()):
            print(f"[{run}] ?뱀쭠 ?뚯씪 ?놁쓬: {d} (add_candidate_features.py 癒쇱?)")
            continue
        vy, vp, va = load(d / FILES["validation"])
        ty, tp, ta = load(d / FILES["test"])
        grid = {(l, m): top1(l * vp + m * va, vy) for l, m in itertools.product(LAMBDAS, MUS) if l != 0}
        bl, bm = max(grid, key=lambda k: (grid[k], -abs(k[0]), -abs(k[1])))
        r = {"怨좊Ⅸ 媛?: {"lambda": bl, "mu": bm, "validation_top1": grid[(bl, bm)]},
             "test": {"?멸린留?: full(tp, ty), "?섏씠留?(??湲곗궗 ?곗꽑)": full(-ta, ty),
                      "?멸린 + ?섏씠": full(bl * tp + bm * ta, ty)},
             "test ??: int(len(ty))}
        fu = fusion_top1(v2, run)
        if fu:
            r["SID + ?멸린 + ?섏씠 (seed蹂?Top-1)"] = {s: m["top1"] for s, m in fu.items()}
            r["SID???쒖닔??紐?(seed蹂? %p)"] = {s: (m["top1"] - r["test"]["?멸린 + ?섏씠"]["top1"]) * 100 for s, m in fu.items()}
        res[run.upper()] = r
        print(f"\n===== {run.upper()} (test {len(ty):,}?? =====")
        print(f"怨좊Ⅸ 媛? 貫 = {bl}, 關 = {bm} (validation Top-1 {grid[(bl, bm)]:.2%}, 貫 : 關 = 1 : {bm / bl:.3g})")
        for name, m in r["test"].items():
            print(f"  test {name:<16} Top-1 {m['top1']:.2%} 쨌 AUC {m['auc']:.4f} 쨌 MRR {m['mrr']:.4f} 쨌 nDCG@5 {m['ndcg@5']:.4f}")
        if fu:
            g = r["SID???쒖닔??紐?(seed蹂? %p)"]
            print("  SID + ?멸린 + ?섏씠: " + " 쨌 ".join(f"seed {s} {m['top1']:.2%}" for s, m in fu.items()))
            print("  SID???쒖닔??紐?(SID + ?멸린 + ?섏씠 ???멸린 + ?섏씠): " + " 쨌 ".join(f"seed {s} {v:+.2f}%p" for s, v in g.items())
                  + f" 쨌 ?됯퇏 {np.mean(list(g.values())):+.2f}%p")
    if not res:
        sys.exit("?뱀쭠 ?뚯씪??李얠? 紐삵뻽?듬땲??)
    out = v2 / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    f, k = out / "pop_age_only.json", 2
    while f.exists():
        f, k = out / f"pop_age_only_{k}.json", k + 1
    f.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n??? {f}")


if __name__ == "__main__":
    main()
