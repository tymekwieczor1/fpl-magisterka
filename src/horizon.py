"""Prognozy na horyzont H kolejek do przodu (dla optymalizatora z horyzontem
kroczącym).

W kolejce decyzyjnej t znamy tylko historię do t-1. Prognozę na kolejkę t+k
budujemy z wiersza cech z kolejki t (forma, minuty, xG z przeszłości),
podmieniając jedynie cechy terminarza (FDR, mecz u siebie, liczba meczów)
na te z kolejki t+k. Dzięki temu prognoza nie korzysta z informacji
z przyszłości.
"""
from pathlib import Path

import numpy as np
import pandas as pd

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "vaastav"


def team_schedule(season: str) -> pd.DataFrame:
    """Terminarz w układzie drużyna x kolejka: liczba meczów, średni FDR, udział meczów u siebie."""
    fx = pd.read_csv(RAW / season / "fixtures.csv")
    fx = fx.dropna(subset=["event"])
    home = fx[["event", "team_h", "team_h_difficulty"]].rename(
        columns={"team_h": "team_id", "team_h_difficulty": "fdr"}).assign(home=1.0)
    away = fx[["event", "team_a", "team_a_difficulty"]].rename(
        columns={"team_a": "team_id", "team_a_difficulty": "fdr"}).assign(home=0.0)
    s = pd.concat([home, away])
    s = s.groupby(["team_id", "event"]).agg(n_fixtures=("fdr", "size"), fdr=("fdr", "mean"),
                                           home_share=("home", "mean")).reset_index()
    s = s.rename(columns={"event": "GW"})
    s["GW"] = s["GW"].astype(int)
    return s


def future_rows(rows_t: pd.DataFrame, sched: pd.DataFrame, gw_target: int) -> pd.DataFrame:
    """Kopia wierszy z kolejki t z terminarzem kolejki gw_target. Drużyny bez meczu -> n_fixtures=0."""
    fut = rows_t.drop(columns=["n_fixtures", "fdr", "home_share"]).merge(
        sched[sched.GW == gw_target][["team_id", "n_fixtures", "fdr", "home_share"]],
        on="team_id", how="left")
    fut["n_fixtures"] = fut["n_fixtures"].fillna(0)
    fut["GW_target"] = gw_target
    return fut
