"""Wczytanie danych historycznych FPL (repozytorium vaastav) do jednego panelu
zawodnik x kolejka.

Jeden wiersz = jeden zawodnik w jednej kolejce. W podwójnych kolejkach (DGW)
statystyki z dwóch meczów są sumowane, a liczba meczów trafia do kolumny
`n_fixtures`. Zawodnik bez meczu w kolejce (blank GW) nie ma wiersza.
"""
from pathlib import Path

import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "vaastav"
SEASONS = ["2022-23", "2023-24", "2024-25", "2025-26", "2026-27"]

# Kolumny sumowane w obrębie kolejki (DGW = suma dwóch meczów)
SUM_COLS = [
    "total_points", "minutes", "starts", "goals_scored", "assists",
    "clean_sheets", "goals_conceded", "bonus", "bps", "saves",
    "expected_goals", "expected_assists", "expected_goal_involvements",
    "expected_goals_conceded", "influence", "creativity", "threat",
    "ict_index", "xP", "yellow_cards", "red_cards", "own_goals",
    "penalties_missed", "penalties_saved", "defensive_contribution",
]


def load_season(season: str) -> pd.DataFrame:
    d = RAW / season
    gw = pd.read_csv(d / "merged_gw.csv")
    raw = pd.read_csv(d / "players_raw.csv")[["id", "code", "team"]]
    raw = raw.rename(columns={"id": "element", "team": "team_id"})
    fx = pd.read_csv(d / "fixtures.csv")[
        ["id", "team_h", "team_a", "team_h_difficulty", "team_a_difficulty"]
    ].rename(columns={"id": "fixture"})

    gw = gw.merge(raw, on="element", how="left")
    gw = gw.merge(fx, on="fixture", how="left")
    # Trudność meczu z perspektywy zawodnika (FDR 1-5)
    gw["was_home"] = gw["was_home"].astype(str).str.lower().eq("true")
    gw["fdr"] = gw["team_h_difficulty"].where(gw["was_home"], gw["team_a_difficulty"])

    for c in SUM_COLS:
        if c not in gw:
            gw[c] = float("nan")  # np. defensive_contribution przed 2025/26
        gw[c] = pd.to_numeric(gw[c], errors="coerce")

    agg = {c: "sum" for c in SUM_COLS}
    # sum() zamienia same NaN na 0 – dla brakujących kolumn chcemy NaN
    agg["defensive_contribution"] = lambda s: s.sum(min_count=1)
    agg.update({
        "name": "first", "position": "first", "team": "first", "team_id": "first",
        "value": "first", "selected": "first", "transfers_balance": "first",
        "was_home": "mean", "fdr": "mean", "fixture": "count",
        "kickoff_time": "min",
    })
    out = gw.groupby(["code", "GW"], as_index=False).agg(agg)
    out = out.rename(columns={"fixture": "n_fixtures", "was_home": "home_share"})
    out["season"] = season
    out["value"] = out["value"] / 10.0  # cena w mln
    return out


def load_panel(seasons=SEASONS) -> pd.DataFrame:
    panel = pd.concat([load_season(s) for s in seasons], ignore_index=True)
    panel["season_idx"] = panel["season"].map({s: i for i, s in enumerate(SEASONS)})
    # Globalny indeks czasu: kolejka w ramach całego panelu
    panel["t"] = panel["season_idx"] * 38 + panel["GW"]
    return panel.sort_values(["code", "t"]).reset_index(drop=True)


if __name__ == "__main__":
    p = load_panel()
    print(p.shape)
    print(p.groupby("season").agg(rows=("code", "size"), players=("code", "nunique"),
                                  gws=("GW", "nunique"), dgw=("n_fixtures", lambda s: (s > 1).sum())))
