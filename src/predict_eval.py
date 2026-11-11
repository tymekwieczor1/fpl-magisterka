"""Ewaluacja modeli predykcyjnych metodą walk-forward na sezonie testowym.

Dla każdej kolejki t sezonu testowego model jest uczony na wszystkich
wcześniejszych kolejkach (t' < t), a następnie prognozuje kolejkę t.
Wynik: plik z prognozami wszystkich wariantów + tabela metryk.
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_panel  # noqa: E402
from features import FEATURE_SETS, build_features, columns_for  # noqa: E402
from horizon import future_rows, team_schedule  # noqa: E402

H = 3  # długość horyzontu (kolejki)

OUT = Path(__file__).resolve().parents[1] / "results"
OUT.mkdir(exist_ok=True)
SEED = 42


def make_model():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05,
                                         max_leaf_nodes=31, min_samples_leaf=50,
                                         l2_regularization=1.0, random_state=SEED)


def walk_forward(df, groups, test_season="2025-26", refit_every=1):
    test = df[df.season == test_season]
    gws = sorted(test.GW.unique())
    preds = test[["code", "name", "position", "season", "GW", "t", "value", "y", "xP", "n_fixtures"]].copy()
    # Baseline: średnia punktów z 5 poprzednich kolejek x liczba meczów
    preds["pred_baseline"] = test["total_points_r5"].fillna(0) * test["n_fixtures"]
    models = {}
    sched = team_schedule(test_season)
    horizon_rows = []
    for i, gw in enumerate(gws):
        t_now = test.loc[test.GW == gw, "t"].iloc[0]
        mask_test = (df.t == t_now)
        if i % refit_every == 0:
            train = df[df.t < t_now]
            for name in FEATURE_SETS:
                # pomijamy cechy bez zmienności w danych uczących
                # (np. wkład defensywny przed jego wprowadzeniem w 2025/26)
                cols = [c for c in columns_for(name, groups) if train[c].nunique() > 1]
                m = make_model().fit(train[cols], train["y"])
                models[name] = (m, cols)
        rows_t = df.loc[mask_test]
        for name, (m, cols) in models.items():
            preds.loc[preds.GW == gw, f"pred_{name}"] = m.predict(rows_t[cols])
        # Prognozy na horyzont: k=0 (bieżąca kolejka) oraz k=1..H-1
        for k in range(H):
            if gw + k > 38:
                break
            fut = future_rows(rows_t, sched, gw + k)
            h = fut[["code", "team_id", "position", "value", "n_fixtures"]].copy()
            h["GW_origin"], h["k"] = gw, k
            scale = fut["n_fixtures"] / rows_t["n_fixtures"].values
            h["pred_baseline"] = fut["total_points_r5"].fillna(0) * fut["n_fixtures"]
            h["pred_xP"] = rows_t["xP"].values / rows_t["n_fixtures"].values * fut["n_fixtures"] if k > 0 \
                else rows_t["xP"].values
            for name, (m, cols) in models.items():
                h[f"pred_{name}"] = np.where(fut["n_fixtures"] > 0, m.predict(fut[cols]), 0.0)
            horizon_rows.append(h)
        print(f"GW{gw} done", flush=True)
    pd.concat(horizon_rows).to_csv(OUT / f"horizon_preds_{test_season}.csv", index=False)
    return preds


def metrics(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in [c for c in preds.columns if c.startswith("pred_")] + ["xP"]:
        e = preds[col] - preds["y"]
        rho = preds.groupby("GW").apply(lambda g: spearmanr(g[col], g["y"]).statistic).mean()
        # Trafność w górnym ogonie: 50 zawodników o najwyższej prognozie w kolejce
        top = preds.sort_values(col, ascending=False).groupby("GW").head(50)
        et = top[col] - top["y"]
        rows.append({"model": col.replace("pred_", ""), "RMSE": np.sqrt((e**2).mean()),
                     "MAE": e.abs().mean(), "Spearman_sr_kolejka": rho,
                     "RMSE_top50": np.sqrt((et**2).mean()), "n": len(e)})
    return pd.DataFrame(rows).sort_values("RMSE")


if __name__ == "__main__":
    t0 = time.time()
    df, groups = build_features(load_panel())
    preds = walk_forward(df, groups)
    preds.to_csv(OUT / "predictions_2025-26.csv", index=False)
    m = metrics(preds)
    m.to_csv(OUT / "metrics_2025-26.csv", index=False)
    print(m.to_string(index=False))
    print(f"czas: {time.time() - t0:.0f} s")
