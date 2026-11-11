"""Podsumowanie backtestów względem średniej menedżerów FPL (H1)
oraz zestawienie trafności prognoz z wynikiem punktowym (H4).

Liczy wyniki dla każdego sezonu testowego osobno i łącznie (wszystkie kolejki
ze wszystkich sezonów, n = 38 x liczba sezonów).

Użycie: python src/summarize.py 2023-24 2024-25 2025-26
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
# xP wyłączone: zawiera informację z rozegranej kolejki (przeciek)
STRATS = ["full", "pts_only", "xg_only", "no_fdr", "no_minutes", "baseline", "hold_full", "oracle"]
MODELS = ["full", "pts_only", "xg_only", "no_fdr", "no_minutes", "baseline"]
rng = np.random.default_rng(42)


def load(season):
    avg = pd.read_csv(ROOT / "data/benchmark/gw_averages.csv")
    avg = avg[avg.season == season][["GW", "average_entry_score"]]
    out = []
    for s in STRATS:
        b = pd.read_csv(RES / f"backtest_{s}_{season}.csv").merge(avg, on="GW")
        b["season"] = season
        out.append(b)
    return pd.concat(out)


def table(bt, met):
    rows = []
    for s in STRATS:
        b = bt[bt.strategy == s]
        diff = (b.net - b.average_entry_score).to_numpy()
        boot = [rng.choice(diff, len(diff)).sum() for _ in range(10000)]
        m = met.loc[s.replace("hold_", "")] if s.replace("hold_", "") in met.index else None
        rows.append({
            "strategia": s, "pkt_netto": b.net.sum(), "srednia": b.average_entry_score.sum(),
            "przewaga": diff.sum(),
            "CI95": f"[{np.percentile(boot, 2.5):.0f}; {np.percentile(boot, 97.5):.0f}]",
            "p_ttest": ttest_rel(b.net, b.average_entry_score).pvalue,
            "p_wilcoxon": wilcoxon(b.net, b.average_entry_score).pvalue,
            "transfery": b.n_transfers.sum(), "hity": b.hits.sum(),
            "RMSE": m.RMSE if m is not None else np.nan,
            "Spearman": m.Spearman_sr_kolejka if m is not None else np.nan,
            "RMSE_top50": m.RMSE_top50 if m is not None else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    seasons = sys.argv[1:] or ["2025-26"]
    all_bt, all_tab = [], []
    for season in seasons:
        bt = load(season)
        met = pd.read_csv(RES / f"metrics_{season}.csv").set_index("model")
        tab = table(bt, met).assign(sezon=season)
        print(f"\n=== {season} ===")
        print(tab.drop(columns="sezon").round(3).to_string(index=False))
        all_bt.append(bt)
        all_tab.append(tab)
    if len(seasons) > 1:
        bt = pd.concat(all_bt)
        met = pd.concat([pd.read_csv(RES / f"metrics_{s}.csv") for s in seasons]).groupby("model").mean(numeric_only=True)
        tab = table(bt, met).assign(sezon="łącznie")
        print("\n=== ŁĄCZNIE (średnie metryk trafności z sezonów) ===")
        print(tab.drop(columns="sezon").round(3).to_string(index=False))
        all_tab.append(tab)
        # H4 na poziomie model x sezon (n = 6 modeli x liczba sezonów)
        t = pd.concat(all_tab[:-1])
        t = t[t.strategia.isin(MODELS)]
        for c in ["RMSE", "Spearman", "RMSE_top50"]:
            r = spearmanr(t[c], t.przewaga)
            print(f"H4: Spearman({c}, przewaga) = {r.statistic:.2f}, p = {r.pvalue:.3f}, n = {len(t)}")
    pd.concat(all_tab).to_csv(RES / "summary_all.csv", index=False)
