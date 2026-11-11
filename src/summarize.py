"""Podsumowanie backtestów sezonu 2025/26 względem średniej menedżerów FPL (H1)
oraz zestawienie trafności prognoz z wynikiem punktowym (H4)."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, ttest_rel, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
avg = pd.read_csv(ROOT / "data/benchmark/gw_averages.csv")
avg = avg[avg.season == "2025-26"][["GW", "average_entry_score"]]
met = pd.read_csv(RES / "metrics_2025-26.csv").set_index("model")
# xP wyłączone: zawiera informację z rozegranej kolejki (przeciek) i brakuje go w 28/38 kolejek
STRATS = ["full", "pts_only", "xg_only", "no_fdr", "no_minutes", "baseline", "hold_full", "oracle"]
rng = np.random.default_rng(42)
rows = []
for s in STRATS:
    b = pd.read_csv(RES / f"backtest_{s}_2025-26.csv").merge(avg, on="GW")
    diff = b.net - b.average_entry_score
    boot = [rng.choice(diff, len(diff)).sum() for _ in range(10000)]
    m = s.replace("hold_", "")
    rows.append({"strategia": s, "pkt_netto": b.net.sum(), "srednia_menedzerow": b.average_entry_score.sum(),
                 "przewaga": diff.sum(), "CI95_przewagi": f"[{np.percentile(boot, 2.5):.0f}; {np.percentile(boot, 97.5):.0f}]",
                 "p_ttest": ttest_rel(b.net, b.average_entry_score).pvalue,
                 "p_wilcoxon": wilcoxon(b.net, b.average_entry_score).pvalue,
                 "transfery": b.n_transfers.sum(), "hity": b.hits.sum(),
                 "RMSE": met.RMSE.get(m, np.nan), "Spearman": met.Spearman_sr_kolejka.get(m, np.nan),
                 "RMSE_top50": met.RMSE_top50.get(m, np.nan)})
out = pd.DataFrame(rows)
out.to_csv(RES / "summary_2025-26.csv", index=False)
print(out.round(3).to_string(index=False))
g = out[out.strategia.isin(["full", "pts_only", "xg_only", "no_fdr", "no_minutes", "baseline"])]
for c in ["RMSE", "Spearman", "RMSE_top50"]:
    print(f"H4: korelacja rang {c} vs pkt_netto: {spearmanr(g[c], g.pkt_netto).statistic:.2f} (n={len(g)})")
