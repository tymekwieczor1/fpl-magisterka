"""Optymalizacja składu i transferów FPL (MILP, solver CBC przez PuLP).

Model wielookresowy na horyzont H kolejek. Realizowana jest tylko decyzja
dla pierwszej kolejki horyzontu (horyzont kroczący, rolling horizon).

Zasady FPL 2026/27 w modelu: budżet, skład 2-5-5-3, max 3 z klubu,
wyjściowa XI (1 GK, >=3 DEF, >=2 MID, >=1 FWD), kapitan x2,
1 darmowy transfer na kolejkę, bank do 5, kara 4 pkt za każdy dodatkowy.
Uproszczenia (opisane w pracy): cena zakupu = cena sprzedaży = cena z kolejki t,
brak chipów w wersji podstawowej.
"""
import pulp

SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
HIT = 4
MAX_FT = 5


def solve(players, pred, current=None, bank=100.0, ft=1, free_transfers=False,
          decay=0.9, bench_w=0.1, time_limit=60):
    """
    players: dict code -> {"pos","team","price"}
    pred:    dict (code, k) -> prognoza punktów w k-tej kolejce horyzontu (k=0..H-1)
    current: zbiór kodów w obecnym składzie (None = nowy skład, np. przed GW1)
    bank:    wolne środki (dla nowego składu = cały budżet)
    free_transfers: True = transfery bez limitu i bez kary (start sezonu / Wildcard)
    Zwraca dict z decyzją dla k=0.
    """
    P = list(players)
    H = 1 + max(k for (_, k) in pred)
    K = range(H)
    cur = set(current or [])
    prob = pulp.LpProblem("fpl", pulp.LpMaximize)
    def binvars(tag):
        return {p: {k: pulp.LpVariable(f"{tag}_{p}_{k}", cat="Binary") for k in K} for p in P}

    x = binvars("x")   # w składzie
    s = binvars("s")   # w wyjściowej XI
    c = binvars("c")   # kapitan
    b = binvars("b")   # zakup
    q = binvars("q")   # sprzedaż
    hits = {k: pulp.LpVariable(f"hits_{k}", lowBound=0, cat="Integer") for k in K}
    f = {k: pulp.LpVariable(f"ft_{k}", lowBound=0, upBound=MAX_FT, cat="Integer") for k in K}
    money = {k: pulp.LpVariable(f"bank_{k}", lowBound=0) for k in K}

    ev = lambda p, k: pred.get((p, k), 0.0)
    prob += pulp.lpSum(
        decay**k * (pulp.lpSum(ev(p, k) * (s[p][k] + c[p][k] + bench_w * (x[p][k] - s[p][k])) for p in P)
                    - HIT * hits[k])
        for k in K)

    for k in K:
        for pos, n in SQUAD.items():
            prob += pulp.lpSum(x[p][k] for p in P if players[p]["pos"] == pos) == n
            prob += pulp.lpSum(s[p][k] for p in P if players[p]["pos"] == pos) >= XI_MIN[pos]
            prob += pulp.lpSum(s[p][k] for p in P if players[p]["pos"] == pos) <= XI_MAX[pos]
        prob += pulp.lpSum(s[p][k] for p in P) == 11
        prob += pulp.lpSum(c[p][k] for p in P) == 1
        for team in {v["team"] for v in players.values()}:
            prob += pulp.lpSum(x[p][k] for p in P if players[p]["team"] == team) <= 3
        for p in P:
            prob += s[p][k] <= x[p][k]
            prob += c[p][k] <= s[p][k]
            prev = (1 if p in cur else 0) if k == 0 else x[p][k - 1]
            prob += x[p][k] == prev + b[p][k] - q[p][k]
            prob += b[p][k] + q[p][k] <= 1
        # bilans środków
        prev_bank = bank if k == 0 else money[k - 1]
        prob += money[k] == prev_bank + pulp.lpSum(players[p]["price"] * (q[p][k] - b[p][k]) for p in P)
        # transfery: wykorzystane darmowe <= dostępne, reszta to płatne
        n_tr = pulp.lpSum(b[p][k] for p in P)
        if k == 0 and free_transfers:
            prob += hits[k] == 0
            prob += f[k] == 1  # po starcie sezonu / Wildcard: 1 darmowy na kolejną kolejkę
        else:
            avail = ft if k == 0 else f[k - 1]
            prob += n_tr - hits[k] <= avail
            # darmowe transfery na następną kolejkę: (dostępne - wykorzystane) + 1, max 5
            prob += f[k] <= avail - (n_tr - hits[k]) + 1
        prob += f[k] >= 0

    prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=time_limit))
    k = 0
    squad = [p for p in P if x[p][k].value() > 0.5]
    xi = [p for p in P if s[p][k].value() > 0.5]
    cap = [p for p in P if c[p][k].value() > 0.5][0]
    vice = max((p for p in xi if p != cap), key=lambda p: ev(p, 0))
    bought = [p for p in P if b[p][k].value() > 0.5]
    sold = [p for p in P if q[p][k].value() > 0.5]
    return {"squad": squad, "xi": xi, "captain": cap, "vice": vice,
            "bought": bought, "sold": sold, "hits": int(round(hits[k].value())),
            "bank": money[k].value(), "ft_next": int(round(f[k].value())),
            "status": pulp.LpStatus[prob.status]}
