# FPL – praca magisterska (WZ UW)

Model wspomagania decyzji menedżera Fantasy Premier League:
predykcja punktów (GBM) + optymalizacja składu i transferów (MILP, horyzont kroczący).

## Uruchomienie
```bash
pip install -r requirements.txt
# dane historyczne (vaastav) – pobierane do data/raw/vaastav, nie są w repo
for s in 2022-23 2023-24 2024-25 2025-26 2026-27; do mkdir -p data/raw/vaastav/$s; for f in gws/merged_gw.csv fixtures.csv teams.csv players_raw.csv; do curl -s https://raw.githubusercontent.com/vaastav/Fantasy-Premier-League/master/data/$s/$f -o data/raw/vaastav/$s/$(basename $f); done; done
for s in 2023-24 2024-25 2025-26; do python src/predict_eval.py $s; python src/backtest.py $s full pts_only xg_only no_fdr no_minutes baseline hold_full oracle; done

python src/summarize.py 2023-24 2024-25 2025-26
```

## Struktura
- `src/data.py` – panel zawodnik × kolejka (DGW sumowane)
- `src/features.py` – cechy kroczące bez przecieku; zestawy cech pod H2/H3
- `src/predict_eval.py` – walk-forward, metryki, prognozy na horyzont 3 kolejek
- `src/optimize.py` – MILP (PuLP/CBC): skład, XI, kapitan, transfery, bank FT
- `src/backtest.py` – symulacja sezonu, automatyczne zmiany
- `src/summarize.py` – porównanie ze średnią menedżerów (H1) i trafność vs punkty (H4)
- `data/benchmark/gw_averages.csv` – średni wynik menedżerów w każdej kolejce (archiwalne snapshoty API)

## Uwaga metodologiczna
Kolumna `xP` z repozytorium vaastav zawiera informację z rozegranej kolejki
(przeciek) i w 2025/26 jest pusta w 28/38 kolejek – nie używamy jej ani jako cechy, ani jako benchmarku.
