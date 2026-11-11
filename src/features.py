"""Inżynieria cech. Każda cecha w kolejce t korzysta wyłącznie z informacji
dostępnych przed terminem tej kolejki:
- statystyki z meczów: średnie kroczące z poprzednich kolejek (przesunięcie o 1),
- informacje znane przed kolejką: FDR, mecz u siebie, liczba meczów, cena.

Cechy są pogrupowane, żeby testy hipotez H2 i H3 mogły włączać i wyłączać
całe grupy.
"""
import numpy as np
import pandas as pd

WINDOWS = (3, 5, 10)


def _roll(g: pd.core.groupby.SeriesGroupBy, w: int) -> pd.Series:
    # shift(1): tylko przeszłe kolejki; min_periods=1: działa od 2. występu
    return g.transform(lambda s: s.shift(1).rolling(w, min_periods=1).mean())


def build_features(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    df = panel.copy()
    # Statystyki na mecz (w DGW dzielimy przez liczbę meczów)
    per_match = ["total_points", "minutes", "starts", "bonus", "bps",
                 "expected_goals", "expected_assists", "expected_goal_involvements",
                 "expected_goals_conceded", "defensive_contribution", "clean_sheets"]
    for c in per_match:
        df[f"{c}_pm"] = df[c] / df["n_fixtures"]

    g = df.groupby("code")
    groups = {"form_pts": [], "xg": [], "minutes": [], "fdr": [], "base": [], "defcon": []}

    for w in WINDOWS:
        for c, grp in [("total_points", "form_pts"), ("bonus", "form_pts"), ("bps", "form_pts"),
                       ("expected_goals", "xg"), ("expected_assists", "xg"),
                       ("expected_goal_involvements", "xg"), ("expected_goals_conceded", "xg"),
                       ("clean_sheets", "form_pts")]:
            name = f"{c}_r{w}"
            df[name] = _roll(g[f"{c}_pm"], w)
            groups[grp].append(name)

    for w in (3, 5):
        df[f"minutes_r{w}"] = _roll(g["minutes_pm"], w)
        df[f"starts_r{w}"] = _roll(g["starts_pm"], w)
        groups["minutes"] += [f"minutes_r{w}", f"starts_r{w}"]
    df["minutes_last"] = g["minutes_pm"].shift(1)
    df["played60_r5"] = _roll(df.assign(p60=(df["minutes_pm"] >= 60).astype(float)).groupby("code")["p60"], 5)
    groups["minutes"] += ["minutes_last", "played60_r5"]

    df["defcon_r5"] = _roll(g["defensive_contribution_pm"], 5)
    groups["defcon"].append("defcon_r5")

    groups["fdr"] += ["fdr", "home_share"]

    for pos in ["GK", "DEF", "MID", "FWD"]:
        df[f"pos_{pos}"] = (df["position"] == pos).astype(int)
    df["n_prev_apps"] = g.cumcount()
    groups["base"] += ["pos_GK", "pos_DEF", "pos_MID", "pos_FWD", "value", "n_fixtures", "n_prev_apps"]

    df["y"] = df["total_points"]
    return df, groups


FEATURE_SETS = {
    # pełny model
    "full": ["base", "form_pts", "xg", "minutes", "fdr", "defcon"],
    # H2: punkty historyczne vs metryki oczekiwane
    "pts_only": ["base", "form_pts", "minutes", "fdr"],
    "xg_only": ["base", "xg", "minutes", "fdr"],
    # H3: oczekiwane minuty vs trudność terminarza
    "no_minutes": ["base", "form_pts", "xg", "fdr", "defcon"],
    "no_fdr": ["base", "form_pts", "xg", "minutes", "defcon"],
}


def columns_for(set_name: str, groups: dict) -> list[str]:
    return [c for grp in FEATURE_SETS[set_name] for c in groups[grp]]
