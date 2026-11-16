#!/usr/bin/env bash
# pull_nhl.sh OWNER/REPO [DATE]
# Pulls everything the GitHub workflow produced (ingest tables, boards, ledger, results) into this machine and,
# when DATE is given, grades that night from the ingest tables. GitHub is the one data source reachable from
# Claude's workspace; api-web.nhle.com is not, so the download itself has to happen in the workflow.
set -euo pipefail
repo="${1:?usage: pull_nhl.sh OWNER/REPO [YYYY-MM-DD]}"; date="${2:-}"
dir="repo/${repo##*/}"
if [ -d "$dir/.git" ]; then git -C "$dir" pull --ff-only -q; else git clone -q --depth 1 "https://github.com/$repo" "$dir"; fi
ls "$dir"/data/nhl_*_schedule.csv >/dev/null 2>&1 || { echo "no data/nhl_*_schedule.csv in $repo: has the workflow run and committed data/?"; exit 1; }
python3 - "$dir" <<'PY'
import sys, glob, pandas as pd
d = sys.argv[1]
for t in ("schedule", "period_scores", "goals", "team_game_stats", "player_game_stats", "goalie_game_stats"):
    f = sorted(glob.glob(f"{d}/data/nhl_*_{t}.csv"))
    print(f"  {t:<20}", f"{len(pd.read_csv(f[-1], low_memory=False)):>7,} rows" if f else "MISSING")
s = pd.read_csv(sorted(glob.glob(f"{d}/data/nhl_*_schedule.csv"))[-1])
fin = s[s.ended_in.notna()]
print(f"  finished games: {len(fin):,}, last {fin.game_date.max()}; score checks failed: {int((~fin.score_checks.astype(bool)).sum())}")
PY
export NHL_DB="$dir/ledger/nhl.sqlite"
if [ -n "$date" ]; then
  mkdir -p results
  python3 nhl_system.py results --date "$date" --ingest "$dir/data" --out "results/$date.txt"
  python3 nhl_system.py grade   --date "$date" --results "results/$date.txt"
  python3 nhl_system.py report
fi
