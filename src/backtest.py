"""Backtest sezonowy: symulacja menedżera FPL sterowanego modelem.

Dla każdej kolejki sezonu testowego:
1. bierzemy prognozy na horyzont H (z predict_eval.py, bez wiedzy o przyszłości),
2. rozwiązujemy MILP (optimize.solve) i realizujemy decyzję dla bieżącej kolejki,
3. liczymy rzeczywiste punkty (z automatycznymi zmianami i wicekapitanem),
4. aktualizujemy skład, bank i liczbę darmowych transferów.

Użycie: python src/backtest.py full baseline xP oracle hold_full
"""
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import load_panel  # noqa: E402
from optimize import XI_MIN, solve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
POOL_PER_POS = 30


def autosub_points(xi, squad, captain, vice, actual, minutes, pred0, pos):
    """Rzeczywiste punkty XI z automatycznymi zmianami (wg kolejności prognoz na ławce)."""
    xi = list(xi)
    bench = sorted([p for p in squad if p not in xi], key=lambda p: -pred0.get(p, 0))
    for p in [p for p in xi if minutes.get(p, 0) == 0]:
        for sub in bench:
            if minutes.get(sub, 0) == 0:
                continue
            if (pos[p] == "GK") != (pos[sub] == "GK"):
                continue
            new = [z for z in xi if z != p] + [sub]
            counts = {k: sum(pos[z] == k for z in new) for k in XI_MIN}
            if all(counts[k] >= XI_MIN[k] for k in XI_MIN):
                xi, bench = new, [b for b in bench if b != sub]
                break
    pts = sum(actual.get(p, 0) for p in xi)
    if minutes.get(captain, 0) > 0:
        pts += actual.get(captain, 0)
    elif minutes.get(vice, 0) > 0:
        pts += actual.get(vice, 0)
    return pts


def run(strategy, hz, panel, season="2025-26"):
    col = "pred_" + strategy.replace("hold_", "")
    act = panel[panel.season == season]
    info = {}  # ostatnio znane: pozycja, drużyna, cena
    squad, bank, ft = None, 100.0, 1
    log = []
    for gw in range(1, 39):
        h = hz[hz.GW_origin == gw]
        for r in h[h.k == 0].itertuples():
            info[r.code] = {"pos": r.position, "team": r.team_id, "price": r.value}
        if strategy == "oracle":  # górna granica: znajomość rzeczywistych punktów
            a = act[act.GW.between(gw, gw + 2)].set_index(["code", "GW"])["y"]
            pred = {(c, g - gw): v for (c, g), v in a.items()}
        else:
            pred = {(r.code, r.k): getattr(r, col) for r in h.itertuples()}
        # pula kandydatów: najlepsi na pozycję wg sumy prognoz + obecny skład
        tot = pd.Series(pred).groupby(level=0).sum()
        tot = tot[tot.index.isin(info)]
        pool = set()
        for pos in ["GK", "DEF", "MID", "FWD"]:
            idx = [c for c in tot.sort_values(ascending=False).index if info[c]["pos"] == pos]
            pool |= set(idx[:POOL_PER_POS])
        pool |= set(squad or [])
        players = {c: info[c] for c in pool if c in info}
        pred_pool = {key: v for key, v in pred.items() if key[0] in players}

        if strategy.startswith("hold_") and squad is not None:
            # skład zamrożony: tylko wybór XI i kapitana (zero transferów)
            players = {c: info[c] for c in squad}
            pred_pool = {key: v for key, v in pred_pool.items() if key[0] in players and key[1] == 0}
            dec = solve(players, pred_pool, current=squad, bank=bank, ft=0)
        else:
            dec = solve(players, pred_pool, current=squad, bank=bank, ft=ft,
                        free_transfers=(squad is None))
        g = act[act.GW == gw].set_index("code")
        actual, minutes = g["y"].to_dict(), g["minutes"].to_dict()
        pred0 = {c: pred.get((c, 0), 0) for c in dec["squad"]}
        pos = {c: info[c]["pos"] for c in dec["squad"]}
        pts = autosub_points(dec["xi"], dec["squad"], dec["captain"], dec["vice"],
                             actual, minutes, pred0, pos)
        net = pts - 4 * dec["hits"]
        log.append({"strategy": strategy, "GW": gw, "points": pts, "hits": dec["hits"], "net": net,
                    "n_transfers": len(dec["bought"]) if squad is not None else 0,
                    "pred_xi": sum(pred0.get(c, 0) for c in dec["xi"]) + pred0.get(dec["captain"], 0),
                    "bank": round(dec["bank"], 1), "ft_next": dec["ft_next"], "status": dec["status"]})
        squad, bank, ft = dec["squad"], dec["bank"], dec["ft_next"]
        print(strategy, gw, net, dec["status"], flush=True)
    return pd.DataFrame(log)


if __name__ == "__main__":
    strategies = sys.argv[1:] or ["full"]
    hz = pd.read_csv(RES / "horizon_preds_2025-26.csv")
    panel = load_panel()
    for st in strategies:
        t0 = time.time()
        out = run(st, hz, panel)
        out.to_csv(RES / f"backtest_{st}_2025-26.csv", index=False)
        print(f"{st}: suma netto {out.net.sum():.0f} pkt, czas {time.time() - t0:.0f} s")
