"""Test prospektywny w sezonie 2026/27: cotygodniowa decyzja modelu zapisywana
PRZED terminem kolejki oraz rozliczenie kolejek zakończonych.

Każde uruchomienie:
1. pobiera aktualny stan z oficjalnego API FPL (zawodnicy, terminarz, historia meczów),
2. rozlicza zakończone kolejki, dla których istnieje zapisana decyzja
   (rzeczywiste punkty składu vs średnia wszystkich menedżerów),
3. jeśli termin najbliższej kolejki wypada przed następnym planowym uruchomieniem
   (wtorek i piątek, 08:43 UTC), uczy model na wszystkich dostępnych danych,
   rozwiązuje MILP i zapisuje decyzję w live/decisions/gwNN.json.

Decyzje trafiają do repozytorium (commit z datą sprzed terminu kolejki), więc
nie można ich zmienić po fakcie. Wirtualny menedżer startuje w pierwszej
obsłużonej kolejce z budżetem 100,0 mln i cenami z dnia decyzji.

Użycie:  python src/live.py            # normalny przebieg
         python src/live.py --dry-run  # wymuś decyzję, niczego nie zapisuj
"""
import datetime as dt
import gzip
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent))
from backtest import POOL_PER_POS, autosub_points  # noqa: E402
from data import RAW, SUM_COLS, load_panel  # noqa: E402
from features import build_features, columns_for  # noqa: E402
from optimize import solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
LIVE = ROOT / "live"
DEC = LIVE / "decisions"
SNAP = ROOT / "data" / "live"
API = "https://fantasy.premierleague.com/api/"
SEASON = "2026-27"
HIST = ["2022-23", "2023-24", "2024-25", "2025-26"]
POS = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}
H = 3
SEED = 42
RUN_DAYS, RUN_HOUR, RUN_MIN = (1, 4), 8, 43  # wtorek, piątek 08:43 UTC


def get(path, tries=4):
    for i in range(tries):
        try:
            req = urllib.request.Request(API + path, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=40) as r:
                return json.load(r)
        except Exception:
            if i == tries - 1:
                raise
            time.sleep(3 * (i + 1))


def ensure_history():
    base = "https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data"
    for s in HIST:
        for f in ["gws/merged_gw.csv", "fixtures.csv", "teams.csv", "players_raw.csv"]:
            out = RAW / s / Path(f).name
            if not out.exists():
                out.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(f"{base}/{s}/{f}", out)


def next_run(now):
    d = now.replace(hour=RUN_HOUR, minute=RUN_MIN, second=0, microsecond=0)
    while d <= now or d.weekday() not in RUN_DAYS:
        d += dt.timedelta(days=1)
        d = d.replace(hour=RUN_HOUR, minute=RUN_MIN)
    return d


def current_season_panel(boot, fixtures):
    """Historia bieżącego sezonu z API w tym samym układzie co dane historyczne."""
    el = pd.DataFrame(boot["elements"])
    rows = []
    for i, pid in enumerate(el["id"]):
        rows += get(f"element-summary/{pid}/")["history"]
        if i % 50 == 49:
            time.sleep(1)
    h = pd.DataFrame(rows)
    fx = pd.DataFrame(fixtures)[["id", "team_h_difficulty", "team_a_difficulty"]].rename(columns={"id": "fixture"})
    h = h.merge(fx, on="fixture", how="left")
    h["fdr"] = h["team_h_difficulty"].where(h["was_home"], h["team_a_difficulty"])
    meta = el[["id", "code", "team", "element_type", "web_name"]].rename(
        columns={"id": "element", "team": "team_id", "web_name": "name"})
    h = h.merge(meta, on="element")
    h["position"] = h["element_type"].map(POS)
    h["xP"] = np.nan
    for c in SUM_COLS:
        h[c] = pd.to_numeric(h[c], errors="coerce")
    agg = {c: "sum" for c in SUM_COLS}
    agg.update({"name": "first", "position": "first", "team_id": "first", "value": "first",
                "selected": "first", "transfers_balance": "first", "was_home": "mean",
                "fdr": "mean", "fixture": "count", "kickoff_time": "min"})
    out = h.groupby(["code", "round"], as_index=False).agg(agg).rename(
        columns={"round": "GW", "fixture": "n_fixtures", "was_home": "home_share"})
    out["team"] = out["team_id"]
    out["season"] = SEASON
    out["value"] = out["value"] / 10.0
    return out, h


def schedule(fixtures):
    fx = pd.DataFrame(fixtures).dropna(subset=["event"])
    a = fx[["event", "team_h", "team_h_difficulty"]].set_axis(["GW", "team_id", "fdr"], axis=1).assign(home=1.0)
    b = fx[["event", "team_a", "team_a_difficulty"]].set_axis(["GW", "team_id", "fdr"], axis=1).assign(home=0.0)
    s = pd.concat([a, b]).groupby(["team_id", "GW"]).agg(
        n_fixtures=("fdr", "size"), fdr=("fdr", "mean"), home_share=("home", "mean")).reset_index()
    s["GW"] = s["GW"].astype(int)
    return s


def predict(boot, fixtures, cur, gw):
    """Prognozy dla kolejek gw..gw+H-1 z modelu uczonego na wszystkich wcześniejszych danych."""
    el = pd.DataFrame(boot["elements"])
    el = el[el["status"] != "u"]  # 'u' = zawodnik niedostępny w grze
    sched = schedule(fixtures)
    fut = pd.DataFrame({"code": el["code"], "name": el["web_name"], "position": el["element_type"].map(POS),
                        "team_id": el["team"], "team": el["team"], "value": el["now_cost"] / 10.0,
                        "GW": gw, "season": SEASON})
    fut = fut.merge(sched[sched.GW == gw][["team_id", "n_fixtures", "fdr", "home_share"]], on="team_id", how="left")
    fut["n_fixtures"] = fut["n_fixtures"].fillna(0)
    panel = pd.concat([load_panel(HIST), cur[cur.GW < gw], fut], ignore_index=True)
    panel["season_idx"] = panel["season"].map({s: i for i, s in enumerate(HIST + [SEASON])})
    panel["t"] = panel["season_idx"] * 38 + panel["GW"]
    panel = panel.sort_values(["code", "t"]).reset_index(drop=True)
    df, groups = build_features(panel)
    t_now = len(HIST) * 38 + gw
    train = df[(df.t < t_now) & df["y"].notna()]
    cols = [c for c in columns_for("full", groups) if train[c].nunique() > 1]
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0,
                                          random_state=SEED).fit(train[cols], train["y"])
    rows = df[df.t == t_now].copy()
    chance = el.set_index("code")["chance_of_playing_next_round"]
    out = rows[["code", "name", "position", "team_id", "value"]].copy()
    for k in range(H):
        r = rows.drop(columns=["n_fixtures", "fdr", "home_share"]).merge(
            sched[sched.GW == gw + k][["team_id", "n_fixtures", "fdr", "home_share"]], on="team_id", how="left")
        r["n_fixtures"] = r["n_fixtures"].fillna(0)
        p = np.where(r["n_fixtures"] > 0, model.predict(r[cols]), 0.0)
        if k == 0:
            # informacja o dostępności znana przed terminem (kontuzje, zawieszenia)
            ch = out["code"].map(chance).astype(float)
            p = p * np.where(ch.isna(), 1.0, ch / 100.0)
        out[f"pred_{k}"] = p
    out = out.merge(el[["code", "ep_next", "selected_by_percent", "chance_of_playing_next_round"]], on="code")
    return out, len(train)


def decide(pred, state):
    info = {r.code: {"pos": r.position, "team": r.team_id, "price": r.value} for r in pred.itertuples()}
    p = {(r.code, k): getattr(r, f"pred_{k}") for r in pred.itertuples() for k in range(H)}
    tot = pred.set_index("code")[[f"pred_{k}" for k in range(H)]].sum(axis=1).sort_values(ascending=False)
    pool = set(state.get("squad") or [])
    for pos in POS.values():
        pool |= set([c for c in tot.index if info[c]["pos"] == pos][:POOL_PER_POS])
    players = {c: info[c] for c in pool if c in info}
    pp = {key: v for key, v in p.items() if key[0] in players}
    first = not state.get("squad")
    return solve(players, pp, current=state.get("squad"), bank=state.get("bank", 100.0),
                 ft=state.get("ft", 1), free_transfers=first)


def settle(boot, hist, results_path):
    """Rozlicz zakończone kolejki z zapisaną decyzją."""
    done = pd.read_csv(results_path) if results_path.exists() else pd.DataFrame(
        columns=["GW", "points", "hits", "net", "average_entry_score", "decided_at"])
    ev = {e["id"]: e for e in boot["events"]}
    new = []
    for f in sorted(DEC.glob("gw*.json")):
        d = json.loads(f.read_text())
        gw = d["gw"]
        if gw in set(done["GW"]) or not ev[gw]["finished"] or not ev[gw]["data_checked"]:
            continue
        g = hist[hist["round"] == gw].groupby("code").agg(y=("total_points", "sum"), m=("minutes", "sum"))
        actual, minutes = g["y"].to_dict(), g["m"].to_dict()
        pred0 = {int(k): v for k, v in d["pred0"].items()}
        pos = {int(k): v for k, v in d["pos"].items()}
        pts = autosub_points(d["xi"], d["squad"], d["captain"], d["vice"], actual, minutes, pred0, pos)
        new.append({"GW": gw, "points": pts, "hits": d["hits"], "net": pts - 4 * d["hits"],
                    "average_entry_score": ev[gw]["average_entry_score"], "decided_at": d["decided_at"]})
    if new:
        done = pd.concat([done, pd.DataFrame(new)], ignore_index=True).sort_values("GW")
    return done, new


def main(dry=False):
    now = dt.datetime.now(dt.timezone.utc)
    DEC.mkdir(parents=True, exist_ok=True)
    SNAP.mkdir(parents=True, exist_ok=True)
    boot, fixtures = get("bootstrap-static/"), get("fixtures/")
    nxt = next((e for e in boot["events"] if e["is_next"]), None)
    cur, hist = current_season_panel(boot, fixtures)
    results, new = settle(boot, hist, LIVE / "results.csv")
    for r in new:
        print(f"Rozliczono GW{r['GW']}: model {r['net']} pkt, średnia menedżerów {r['average_entry_score']}")
    if nxt is None:
        print("Sezon zakończony – brak kolejnej kolejki.")
    else:
        gw = nxt["id"]
        deadline = dt.datetime.fromisoformat(nxt["deadline_time"].replace("Z", "+00:00"))
        due = deadline < next_run(now)
        exists = (DEC / f"gw{gw:02d}.json").exists()
        print(f"Najbliższa kolejka: GW{gw}, termin {deadline:%Y-%m-%d %H:%M} UTC, "
              f"następne uruchomienie {next_run(now):%Y-%m-%d %H:%M} UTC")
        if exists and not dry:
            print("Decyzja dla tej kolejki jest już zapisana.")
        elif not (due or dry):
            print("Za wcześnie na decyzję – zostanie podjęta przy ostatnim uruchomieniu przed terminem.")
        else:
            ensure_history()
            state_path = LIVE / "state.json"
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            pred, n_train = predict(boot, fixtures, cur, gw)
            dec = decide(pred, state)
            nm = pred.set_index("code")["name"].to_dict()
            p0 = pred.set_index("code")["pred_0"]
            commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                    capture_output=True, text=True).stdout.strip()
            out = {"gw": gw, "decided_at": now.isoformat(timespec="seconds"),
                   "deadline": deadline.isoformat(), "code_commit": commit, "n_train_rows": n_train,
                   "squad": dec["squad"], "xi": dec["xi"], "captain": dec["captain"], "vice": dec["vice"],
                   "bought": dec["bought"], "sold": dec["sold"], "hits": dec["hits"],
                   "bank": round(dec["bank"], 1), "ft_next": dec["ft_next"], "status": dec["status"],
                   "pred_xi_points": round(float(p0[dec["xi"]].sum() + p0[dec["captain"]]), 2),
                   "pred0": {str(c): round(float(p0[c]), 3) for c in dec["squad"]},
                   "pos": {str(c): pred.set_index("code")["position"][c] for c in dec["squad"]},
                   "names": {str(c): nm[c] for c in dec["squad"] + dec["sold"]}}
            print(f"Decyzja GW{gw}: kapitan {nm[dec['captain']]}, transfery +{[nm[c] for c in dec['bought']]} "
                  f"-{[nm[c] for c in dec['sold']]}, płatne: {dec['hits']}, prognoza XI: {out['pred_xi_points']}")
            print("XI:", ", ".join(nm[c] for c in dec["xi"]))
            if not dry:
                (DEC / f"gw{gw:02d}.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
                state_path.write_text(json.dumps({"squad": dec["squad"], "bank": dec["bank"],
                                                  "ft": dec["ft_next"], "last_gw": gw}))
                # prognozy wszystkich zawodników + oficjalna prognoza FPL (ep_next) do późniejszych porównań
                pred.to_csv(SNAP / f"predictions_gw{gw:02d}.csv", index=False)
                with gzip.open(SNAP / f"bootstrap_gw{gw:02d}.json.gz", "wt") as fh:
                    json.dump(boot, fh)
    if not dry:
        results.to_csv(LIVE / "results.csv", index=False)
        if len(results):
            d = results["net"].sum() - results["average_entry_score"].sum()
            print(f"Łącznie {len(results)} kolejek: model {results['net'].sum()} pkt, "
                  f"średnia {results['average_entry_score'].sum()} pkt, różnica {d:+.0f}")


if __name__ == "__main__":
    main(dry="--dry-run" in sys.argv)
