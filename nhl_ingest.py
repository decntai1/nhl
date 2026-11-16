#!/usr/bin/env python3
"""
nhl_ingest.py — download NHL results and box scores (all 32 teams, every finished game, every skater and
goalie), verify keys, write CSVs + a constrained DuckDB, publish the CSVs to ProLyt. Same shape as nfl_ingest.py.

Source: the NHL's own public API, https://api-web.nhle.com/v1 (no key).
  club-schedule-season/<TEAM>/<season|now>   every game of a team      -> schedule
  gamecenter/<id>/boxscore                    player lines of one game  -> skater / goalie / team game stats
  gamecenter/<id>/landing                     scoring summary           -> goals, period scores, regulation score
  club-stats/<TEAM>/<season/2|now>            season totals             -> players, season stats
  standings/now                               standings                 -> standings (flattened as delivered)

Env:
  PROLYT_API_KEY   upload key (falls back to ANALYTIQ_API_KEY). Unset -> build locally, skip publish.
  PROLYT_URL       platform URL (falls back to ANALYTIQ_URL, default https://analytiq.dcentai.tech)
  NHL_SEASON       start year, e.g. "2026" for 2026-27 (default: current season, Jul-Jun)
  NHL_GAME_TYPES   "2" regular season (default), "2 3" with playoffs, "1 2 3" with pre-season
  NHL_OUT          output dir (default ./data)
  NHL_CACHE        response cache (default ./cache). Finished games older than NHL_RECHECK_DAYS (3) are read
                   from it and never downloaded again; keep the folder between runs (actions/cache) or the
                   first step of every run is ~2 requests per game played so far.
  NHL_OFFLINE      1 = never touch the network, build from the cache only (tests)
Needs: pandas, numpy, requests   (duckdb optional — only for the local .duckdb)

Tables (prefix nhl_<season>_):
  teams               one row per club
  schedule            one row per game: final score, REGULATION score, how it ended (REG / OT / SO)
  period_scores       game x team x period (1, 2, 3, OT, SO)
  goals               every goal: period, time, scorer, assists, strength, empty-net flag, running score
  team_game_stats     game x team: goals (final / regulation), shots, saves, PP goals, hits, blocks, W / L / OTL
  player_game_stats   game x skater: goals, assists, points, shots on goal, hits, blocks, PIM, TOI, first-goal flag
  goalie_game_stats   game x goalie: shots against, saves, goals against, decision, starter, TOI
  players             one row per player seen this season (id, full name, position, latest team)
  player_season_stats skater season totals per club        goalie_season_stats   goalie season totals per club
  standings           as delivered by the API, flattened

Betting conventions baked into the columns (the reason the regulation score is its own column):
  * home_score / away_score  = official final. A shootout adds ONE goal to the winner; nobody scores it.
  * home_reg / away_reg      = after 60 minutes. Three-way (1X2) and most period markets settle on this.
  * goals lists no shootout attempts; player goals / points never include them.
  * a skater with toi_seconds = 0 dressed but did not play: his bets are void, not lost.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

API = "https://api-web.nhle.com/v1"
CACHE = os.environ.get("NHL_CACHE", "./cache")
OFFLINE = os.environ.get("NHL_OFFLINE", "") == "1"
RECHECK_DAYS = int(os.environ.get("NHL_RECHECK_DAYS", "3"))     # stat corrections land within a day or two
TEAMS = ["ANA", "BOS", "BUF", "CAR", "CBJ", "CGY", "CHI", "COL", "DAL", "DET", "EDM", "FLA", "LAK", "MIN", "MTL",
         "NJD", "NSH", "NYI", "NYR", "OTT", "PHI", "PIT", "SEA", "SJS", "STL", "TBL", "TOR", "UTA", "VAN", "VGK",
         "WPG", "WSH"]
DONE = ("OFF", "FINAL")
_calls = {"net": 0, "cache": 0}


# ================================================================ download
def _cache_file(path: str) -> str:
    return os.path.join(CACHE, "nhl_" + hashlib.md5(path.encode()).hexdigest() + ".json")


def fetch(path: str, fresh: bool = False, retries: int = 4):
    """GET <API>/<path> as JSON. fresh=False serves from the disk cache when it is there. None on 404."""
    os.makedirs(CACHE, exist_ok=True)
    f = _cache_file(path)
    if os.path.exists(f) and (OFFLINE or not fresh):
        _calls["cache"] += 1
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    if OFFLINE:
        return None
    import requests
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0 (nhl_ingest)"}, timeout=40)
            if r.status_code == 404:
                return None
            if r.status_code == 429:                       # rate limit: wait and go again
                time.sleep(5 * attempt)
                continue
            r.raise_for_status()
            data = r.json()
            with open(f, "w", encoding="utf-8") as fh:
                json.dump(data, fh)
            _calls["net"] += 1
            time.sleep(0.12)
            return data
        except Exception as e:
            if attempt == retries:
                if os.path.exists(f):                      # a stale copy beats no row
                    print(f"  [!!] {path}: {e} — using the cached copy")
                    with open(f, encoding="utf-8") as fh:
                        return json.load(fh)
                raise RuntimeError(f"{path} failed: {e}") from e
            time.sleep(3 * attempt)
    return None


def _d(x, default=""):
    """the API wraps most names as {"default": "..."}; some endpoints give the plain string."""
    if isinstance(x, dict):
        return x.get("default", default)
    return default if x is None else x


def _secs(toi) -> int:
    """"11:22" -> 682"""
    try:
        m, s = str(toi).split(":")
        return int(m) * 60 + int(s)
    except Exception:
        return 0


def _frac(v):
    """"34/36" -> (34, 36)"""
    try:
        a, b = str(v).split("/")
        return int(a), int(b)
    except Exception:
        return (np.nan, np.nan)


# ================================================================ schedule
def load_schedule(season: int, types: set[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    code = f"{season}{season + 1}"
    arg = "now" if season == current_season() else code
    games, teams = {}, {}
    for t in TEAMS:
        d = fetch(f"club-schedule-season/{t}/{arg}", fresh=True)
        if not d or not d.get("games"):
            print(f"  [--] schedule {t}: nothing returned")
            continue
        for g in d["games"]:
            if str(g.get("season")) != code or g.get("gameType") not in types:
                continue
            games[g["id"]] = g
            for side in ("homeTeam", "awayTeam"):
                tm = g[side]
                teams[tm["abbrev"]] = {"team_abbr": tm["abbrev"], "team_id": tm.get("id"),
                                       "team_name": _d(tm.get("commonName")), "team_place": _d(tm.get("placeName"))}
    rows = []
    for g in games.values():
        h, a = g["homeTeam"], g["awayTeam"]
        rows.append({"game_id": g["id"], "season": code, "game_type": g["gameType"], "game_date": g.get("gameDate"),
                     "start_time_utc": g.get("startTimeUTC"), "venue": _d(g.get("venue")),
                     "home_team": h["abbrev"], "away_team": a["abbrev"], "game_state": g.get("gameState"),
                     "schedule_state": g.get("gameScheduleState"),
                     "home_score": h.get("score") if g.get("gameState") in DONE else np.nan,
                     "away_score": a.get("score") if g.get("gameState") in DONE else np.nan,
                     "ended_in": (g.get("gameOutcome") or {}).get("lastPeriodType") if g.get("gameState") in DONE else None})
    sch = pd.DataFrame(rows).sort_values(["game_date", "game_id"]).reset_index(drop=True) if rows else pd.DataFrame()
    tm = pd.DataFrame(teams.values()).sort_values("team_abbr").reset_index(drop=True) if teams else pd.DataFrame()
    print(f"  [ok] schedule {code}: {len(sch):,} games, {int(sch['game_state'].isin(DONE).sum()) if len(sch) else 0:,} finished")
    return sch, tm


# ================================================================ one game
def game_tables(gid: int, date: str) -> dict | None:
    """box score + scoring summary of one finished game as row lists. None if the API does not call it final."""
    old = False
    try:
        old = (dt.date.today() - dt.date.fromisoformat(str(date)[:10])).days > RECHECK_DAYS
    except Exception:
        pass
    box = fetch(f"gamecenter/{gid}/boxscore", fresh=not old)
    land = fetch(f"gamecenter/{gid}/landing", fresh=not old)
    if not box or not land or box.get("gameState") not in DONE or land.get("gameState") not in DONE:
        return None
    home, away = box["homeTeam"]["abbrev"], box["awayTeam"]["abbrev"]
    final = {home: int(box["homeTeam"].get("score", 0)), away: int(box["awayTeam"].get("score", 0))}
    sog = {home: box["homeTeam"].get("sog"), away: box["awayTeam"].get("sog")}
    ended = (box.get("gameOutcome") or {}).get("lastPeriodType") or (land.get("periodDescriptor") or {}).get("periodType")

    # ---- goals and period scores from the scoring summary (shootout attempts are not goals)
    goals, per, n, ppg = [], {}, 0, {home: 0, away: 0}
    played = set()
    for blk in (land.get("summary") or {}).get("scoring", []):
        pdsc = blk.get("periodDescriptor") or {}
        ptype, num = pdsc.get("periodType", "REG"), pdsc.get("number")
        if ptype == "SO":
            continue
        label = str(num) if ptype == "REG" else ("OT" if num in (None, 4) else f"OT{num - 3}")
        played.add(label)
        for g in blk.get("goals", []):
            n += 1
            tm = _d(g.get("teamAbbrev"))
            per[(label, tm)] = per.get((label, tm), 0) + 1
            if g.get("strength") == "pp" and tm in ppg:
                ppg[tm] += 1
            a = g.get("assists") or []
            goals.append({
                "game_id": gid, "goal_no": n, "period": label, "period_type": ptype,
                "time_in_period": g.get("timeInPeriod"),
                "game_seconds": ((num or 1) - 1) * 1200 + _secs(g.get("timeInPeriod")),
                "team": tm, "is_home": tm == home,
                "scorer_id": g.get("playerId"),
                "scorer": f"{_d(g.get('firstName'))} {_d(g.get('lastName'))}".strip() or _d(g.get("name")),
                "assist1_id": a[0].get("playerId") if len(a) > 0 else np.nan,
                "assist1": f"{_d(a[0].get('firstName'))} {_d(a[0].get('lastName'))}".strip() if len(a) > 0 else None,
                "assist2_id": a[1].get("playerId") if len(a) > 1 else np.nan,
                "assist2": f"{_d(a[1].get('firstName'))} {_d(a[1].get('lastName'))}".strip() if len(a) > 1 else None,
                "strength": g.get("strength"), "goal_modifier": g.get("goalModifier"),
                "empty_net": "empty" in str(g.get("goalModifier", "")), "shot_type": g.get("shotType"),
                "home_score": g.get("homeScore"), "away_score": g.get("awayScore"),
                "goals_to_date": g.get("goalsToDate")})
    reg = {t: sum(v for (p, tm), v in per.items() if tm == t and p in ("1", "2", "3")) for t in (home, away)}
    ot = {t: sum(v for (p, tm), v in per.items() if tm == t and p.startswith("OT")) for t in (home, away)}
    so_winner = None
    if ended == "SO":
        so_winner = home if final[home] > final[away] else away
    periods = []
    labels = ["1", "2", "3"] + sorted(p for p in played if p.startswith("OT"))
    if ended in ("OT", "SO") and not any(p.startswith("OT") for p in labels):
        labels.append("OT")
    for lab in labels:
        for t in (home, away):
            periods.append({"game_id": gid, "team": t, "period": lab, "goals": per.get((lab, t), 0)})
    if ended == "SO":
        for t in (home, away):
            periods.append({"game_id": gid, "team": t, "period": "SO", "goals": int(t == so_winner)})
    check = all(reg[t] + ot[t] + int(t == so_winner) == final[t] for t in (home, away))

    # ---- player lines
    first_scorer = goals[0]["scorer_id"] if goals else None
    last_scorer = goals[-1]["scorer_id"] if goals else None
    skaters, goalies, agg = [], [], {home: {}, away: {}}
    for side, tm, opp in (("homeTeam", home, away), ("awayTeam", away, home)):
        grp = (box.get("playerByGameStats") or {}).get(side, {})
        for kind in ("forwards", "defense"):
            for p in grp.get(kind, []):
                toi = _secs(p.get("toi"))
                skaters.append({
                    "game_id": gid, "player_id": p["playerId"], "team": tm, "opponent": opp, "is_home": tm == home,
                    "player_short": _d(p.get("name")), "position": p.get("position"),
                    "sweater": p.get("sweaterNumber"), "played": toi > 0,
                    "goals": p.get("goals", 0), "assists": p.get("assists", 0), "points": p.get("points", 0),
                    "shots_on_goal": p.get("sog", 0), "plus_minus": p.get("plusMinus", 0), "pim": p.get("pim", 0),
                    "hits": p.get("hits", 0), "blocked_shots": p.get("blockedShots", 0),
                    "pp_goals": p.get("powerPlayGoals", 0), "shifts": p.get("shifts", 0),
                    "giveaways": p.get("giveaways", 0), "takeaways": p.get("takeaways", 0),
                    "faceoff_win_pct": p.get("faceoffWinningPctg"), "toi": p.get("toi"), "toi_seconds": toi,
                    "first_goal": p["playerId"] == first_scorer, "last_goal": p["playerId"] == last_scorer})
                for k_out, k_in in (("hits", "hits"), ("blocked_shots", "blockedShots"), ("pim", "pim"),
                                    ("giveaways", "giveaways"), ("takeaways", "takeaways")):
                    agg[tm][k_out] = agg[tm].get(k_out, 0) + (p.get(k_in) or 0)
        for p in grp.get("goalies", []):
            toi = _secs(p.get("toi"))
            sv, sa = p.get("saves"), p.get("shotsAgainst")
            if sv is None or sa is None:
                sv, sa = _frac(p.get("saveShotsAgainst"))
            es, pp, sh = _frac(p.get("evenStrengthShotsAgainst")), _frac(p.get("powerPlayShotsAgainst")), \
                _frac(p.get("shorthandedShotsAgainst"))
            goalies.append({
                "game_id": gid, "player_id": p["playerId"], "team": tm, "opponent": opp, "is_home": tm == home,
                "player_short": _d(p.get("name")), "sweater": p.get("sweaterNumber"), "played": toi > 0,
                "starter": bool(p.get("starter", False)), "decision": p.get("decision"),
                "shots_against": sa, "saves": sv, "goals_against": p.get("goalsAgainst"),
                "save_pct": p.get("savePctg"), "es_saves": es[0], "es_shots_against": es[1],
                "pp_saves": pp[0], "pp_shots_against": pp[1], "sh_saves": sh[0], "sh_shots_against": sh[1],
                "pim": p.get("pim", 0), "toi": p.get("toi"), "toi_seconds": toi})
            if toi > 0:
                agg[tm]["saves"] = agg[tm].get("saves", 0) + (sv if sv == sv and sv is not None else 0)

    team_rows = []
    for tm, opp in ((home, away), (away, home)):
        won = final[tm] > final[opp]
        team_rows.append({
            "game_id": gid, "team": tm, "opponent": opp, "is_home": tm == home,
            "goals": final[tm], "goals_against": final[opp], "goals_reg": reg[tm], "goals_against_reg": reg[opp],
            "goals_ot": ot[tm], "shootout_win": tm == so_winner,
            "shots_on_goal": sog[tm], "shots_against": sog[opp], "saves": agg[tm].get("saves", np.nan),
            "pp_goals": ppg[tm], "pim": agg[tm].get("pim", 0), "hits": agg[tm].get("hits", 0),
            "blocked_shots": agg[tm].get("blocked_shots", 0), "giveaways": agg[tm].get("giveaways", 0),
            "takeaways": agg[tm].get("takeaways", 0),
            "win": won, "win_reg": reg[tm] > reg[opp], "tie_reg": reg[tm] == reg[opp],
            "result": "W" if won else ("L" if ended == "REG" else "OTL"),
            "scored_first": bool(goals) and goals[0]["team"] == tm})
    game = {"game_id": gid, "home_score": final[home], "away_score": final[away], "home_reg": reg[home],
            "away_reg": reg[away], "ended_in": ended, "went_ot": ended in ("OT", "SO"), "went_so": ended == "SO",
            "home_sog": sog[home], "away_sog": sog[away], "total_goals": final[home] + final[away],
            "total_goals_reg": reg[home] + reg[away],
            "winner": home if final[home] > final[away] else away,
            "first_goal_team": goals[0]["team"] if goals else None,
            "score_checks": check}
    return {"game": game, "goals": goals, "periods": periods, "skaters": skaters, "goalies": goalies, "teams": team_rows}


# ================================================================ season totals
def load_club_stats(season: int, types: set[int]):
    code = f"{season}{season + 1}"
    sk, gl = [], []
    for t in TEAMS:
        for gt in sorted(types & {2, 3}):
            try:
                d = fetch(f"club-stats/{t}/{code}/{gt}", fresh=True)
            except Exception:
                d = None
            if d is None and gt == 2 and season == current_season():
                try:
                    d = fetch(f"club-stats/{t}/now", fresh=True)
                except Exception as e:
                    print(f"  [--] club stats {t}: {e}")
            if not d:
                continue
            for p in d.get("skaters", []):
                gp = p.get("gamesPlayed") or 0
                sk.append({"season": code, "game_type": gt, "team": t, "player_id": p["playerId"],
                           "player_name": f"{_d(p.get('firstName'))} {_d(p.get('lastName'))}".strip(),
                           "position": p.get("positionCode"), "games_played": gp,
                           "goals": p.get("goals"), "assists": p.get("assists"), "points": p.get("points"),
                           "shots": p.get("shots"), "plus_minus": p.get("plusMinus"),
                           "pim": p.get("penaltyMinutes"), "pp_goals": p.get("powerPlayGoals"),
                           "sh_goals": p.get("shorthandedGoals"), "gw_goals": p.get("gameWinningGoals"),
                           "ot_goals": p.get("overtimeGoals"), "shooting_pct": p.get("shootingPctg"),
                           "avg_toi_seconds": p.get("avgTimeOnIcePerGame"), "avg_shifts": p.get("avgShiftsPerGame"),
                           "faceoff_win_pct": p.get("faceoffWinPctg"),
                           "goals_per_game": (p.get("goals") or 0) / gp if gp else np.nan,
                           "points_per_game": (p.get("points") or 0) / gp if gp else np.nan,
                           "shots_per_game": (p.get("shots") or 0) / gp if gp else np.nan})
            for p in d.get("goalies", []):
                gl.append({"season": code, "game_type": gt, "team": t, "player_id": p["playerId"],
                           "player_name": f"{_d(p.get('firstName'))} {_d(p.get('lastName'))}".strip(),
                           "position": "G", "games_played": p.get("gamesPlayed"), "games_started": p.get("gamesStarted"),
                           "wins": p.get("wins"), "losses": p.get("losses"), "ot_losses": p.get("overtimeLosses"),
                           "gaa": p.get("goalsAgainstAverage"), "save_pct": p.get("savePercentage"),
                           "shots_against": p.get("shotsAgainst"), "saves": p.get("saves"),
                           "goals_against": p.get("goalsAgainst"), "shutouts": p.get("shutouts"),
                           "toi_seconds": p.get("timeOnIce")})
    return pd.DataFrame(sk), pd.DataFrame(gl)


def load_standings() -> pd.DataFrame:
    """flattened as delivered — column names are the API's own (camelCase, '.' -> '_'), not checked field by field."""
    try:
        d = fetch("standings/now", fresh=True)
        rows = (d or {}).get("standings") or []
        if not rows:
            return pd.DataFrame()
        df = pd.json_normalize(rows)
        df.columns = [c.replace(".", "_") for c in df.columns]
        df = df[[c for c in df.columns if not any(k in c.lower() for k in ("logo", "_fr", "link"))]]
        if "teamAbbrev_default" in df.columns:
            df.insert(0, "team_abbr", df["teamAbbrev_default"])
        return df
    except Exception as e:
        print(f"  [--] standings: {e}")
        return pd.DataFrame()


# ================================================================ build
def main(season: int, out: str, types: set[int]) -> str:
    os.makedirs(out, exist_ok=True)
    prefix = f"nhl_{season}"
    print(f"=== Downloading NHL {season}-{str(season + 1)[2:]} — game types {sorted(types)} ===")
    sch, teams = load_schedule(season, types)
    if sch.empty:
        sys.exit("schedule is empty — wrong season, or the API is unreachable from here")

    print("\n=== Box scores ===")
    done = sch[sch["game_state"].isin(DONE)]
    acc = {k: [] for k in ("game", "goals", "periods", "skaters", "goalies", "teams")}
    failed = []
    for i, r in enumerate(done.itertuples(), 1):
        try:
            g = game_tables(int(r.game_id), r.game_date)
        except Exception as e:
            failed.append((r.game_id, str(e)[:80])); g = None
        if g is None:
            if not failed or failed[-1][0] != r.game_id:
                failed.append((r.game_id, "not final / not found"))
            continue
        acc["game"].append(g["game"])
        for k in ("goals", "periods", "skaters", "goalies", "teams"):
            acc[k] += g[k]
        if i % 100 == 0:
            print(f"  {i:,} / {len(done):,} games  ({_calls['net']:,} downloaded, {_calls['cache']:,} from cache)")
    print(f"  [ok] {len(acc['game']):,} of {len(done):,} finished games built  "
          f"({_calls['net']:,} requests, {_calls['cache']:,} cache reads)")
    for gid, why in failed[:10]:
        print(f"  [!!] {gid}: {why}")

    print("\n=== Building tables ===")
    T = {"teams": teams}
    gm = pd.DataFrame(acc["game"])
    if len(gm):
        sch = sch.drop(columns=["home_score", "away_score", "ended_in"]).merge(gm, on="game_id", how="left")
    for c in ("home_score", "away_score", "home_reg", "away_reg", "home_sog", "away_sog", "total_goals",
              "total_goals_reg"):
        if c in sch:
            sch[c] = sch[c].astype("Int64")
    T["schedule"] = sch
    ctx = sch[["game_id", "season", "game_type", "game_date"]]
    add = lambda rows: ctx.merge(pd.DataFrame(rows), on="game_id", how="inner") if rows else pd.DataFrame()  # noqa: E731
    T["period_scores"] = add(acc["periods"])
    T["goals"] = add(acc["goals"])
    T["team_game_stats"] = add(acc["teams"])
    pgs, ggs = add(acc["skaters"]), add(acc["goalies"])

    sk, gl = load_club_stats(season, types)
    names = {}
    for df in (sk, gl):
        if len(df):
            names.update(dict(zip(df["player_id"], df["player_name"])))
    for g in acc["goals"]:                                   # the scoring summary has full names too
        for i_, n_ in ((g["scorer_id"], g["scorer"]), (g["assist1_id"], g["assist1"]), (g["assist2_id"], g["assist2"])):
            if n_ and i_ == i_ and i_ not in names:
                names[i_] = n_
    for df in (pgs, ggs):
        if len(df):
            df.insert(df.columns.get_loc("player_short"), "player_name",
                      df["player_id"].map(names).fillna(df["player_short"]))
    for c in ("assist1_id", "assist2_id"):
        if len(T["goals"]):
            T["goals"][c] = T["goals"][c].astype("Int64")
    if len(pgs):
        T["player_game_stats"] = pgs.sort_values(["game_date", "game_id", "team", "player_name"])
    if len(ggs):
        T["goalie_game_stats"] = ggs.sort_values(["game_date", "game_id", "team", "player_name"])
    if len(sk):
        T["player_season_stats"] = sk.sort_values(["team", "player_name"])
    if len(gl):
        T["goalie_season_stats"] = gl.sort_values(["team", "player_name"])

    seen = []
    for df, pos in ((pgs, None), (ggs, "G")):
        if len(df):
            x = df.sort_values(["game_date", "game_id"]).groupby("player_id").agg(
                player_name=("player_name", "last"), team=("team", "last"), games_dressed=("game_id", "nunique"),
                games_played=("played", "sum"), last_game=("game_date", "max"),
                position=("position", "last") if pos is None else ("player_id", lambda s: pos)).reset_index()
            seen.append(x)
    roster = [df[["player_id", "player_name", "team", "position"]] for df in (sk, gl) if len(df)]
    if seen or roster:
        pl = pd.concat(seen, ignore_index=True) if seen else pd.DataFrame(columns=["player_id"])
        if roster:                                            # players on a roster who have not dressed yet
            extra = pd.concat(roster, ignore_index=True)
            extra = extra[~extra["player_id"].isin(pl["player_id"])].drop_duplicates("player_id", keep="last")
            pl = pd.concat([pl, extra], ignore_index=True)
        T["players"] = pl.drop_duplicates("player_id").sort_values(["team", "player_name"]).reset_index(drop=True)

    st = load_standings()
    if len(st):
        T["standings"] = st

    sanity(T)
    print(f"\n=== Writing CSVs to {out}/ ===")
    for name, df in T.items():
        path = f"{out}/{prefix}_{name}.csv"
        df.to_csv(path, index=False)
        print(f"  {name:<22} {len(df):>8,} rows  {df.shape[1]:>4} cols  {os.path.getsize(path) / 1e6:6.2f} MB")
    return prefix


def sanity(T: dict) -> None:
    """the cross-checks that would have caught a wrong field name: scores must add up three independent ways."""
    print("\n=== Sanity ===")
    sch = T["schedule"]
    fin = sch[sch.get("ended_in").notna()] if "ended_in" in sch else sch.iloc[0:0]
    if fin.empty:
        print("  no finished games")
        return
    bad = fin[~fin["score_checks"].astype(bool)]
    print(f"  {'OK ' if bad.empty else 'BAD'} period goals + shootout goal = final score: "
          f"{len(fin) - len(bad):,} of {len(fin):,} games" + ("" if bad.empty else f"  e.g. {bad['game_id'].head(3).tolist()}"))
    if "player_game_stats" in T:
        pg = T["player_game_stats"].groupby("game_id")["goals"].sum()
        tot = (fin.set_index("game_id")["total_goals"] - fin.set_index("game_id")["went_so"].astype(int)).astype(int)
        j = pd.concat([pg, tot], axis=1, keys=["players", "game"]).dropna()
        off = j[j["players"] != j["game"]]
        print(f"  {'OK ' if off.empty else 'BAD'} skater goals = game goals (shootout goal excluded): "
              f"{len(j) - len(off):,} of {len(j):,} games" + ("" if off.empty else f"  e.g. {off.index[:3].tolist()}"))
    if "goalie_game_stats" in T and "team_game_stats" in T:
        sv = T["goalie_game_stats"].groupby(["game_id", "team"])["saves"].sum()
        tg = T["team_game_stats"].set_index(["game_id", "team"])
        so_games = set(fin.loc[fin["went_so"].astype(bool), "game_id"])
        lost_so = pd.Series([(g in so_games) and not w for (g, _), w in zip(tg.index, tg["shootout_win"])], index=tg.index)
        j = pd.concat([sv, tg["shots_against"], tg["goals_against"] - lost_so.astype(int)], axis=1,
                      keys=["saves", "sa", "ga_play"]).dropna()
        off = j[(j["sa"] - j["saves"] - j["ga_play"]) != 0]            # an empty-net goal is a shot no goalie faced
        en = int(T["goals"]["empty_net"].sum()) if "goals" in T and len(T["goals"]) else 0
        print(f"  {'OK ' if len(off) <= en else 'CHK'} shots against - saves = goals against in play: "
              f"{len(j) - len(off):,} of {len(j):,} team-games; {en} empty-net goals in the data")
    print(f"  ended in regulation {int((fin['ended_in'] == 'REG').sum()):,} · overtime {int((fin['ended_in'] == 'OT').sum()):,} "
          f"· shootout {int((fin['ended_in'] == 'SO').sum()):,}   ({(fin['ended_in'] != 'REG').mean():.0%} tied after 60 min)")


# ================================================================== keys + DuckDB + publish
# Keys are VERIFIED against the data before they are declared; a failed check is reported and the key skipped.
PROLYT_URL = os.environ.get("PROLYT_URL") or os.environ.get("ANALYTIQ_URL") or "https://analytiq.dcentai.tech"
PROLYT_KEY = os.environ.get("PROLYT_API_KEY") or os.environ.get("ANALYTIQ_API_KEY", "")

PK_SPEC = {
    "teams":               ["team_abbr"],
    "schedule":            ["game_id"],
    "period_scores":       ["game_id", "team", "period"],
    "goals":               ["game_id", "goal_no"],
    "team_game_stats":     ["game_id", "team"],
    "player_game_stats":   ["game_id", "player_id"],
    "goalie_game_stats":   ["game_id", "player_id"],
    "players":             ["player_id"],
    "player_season_stats": ["season", "game_type", "team", "player_id"],
    "goalie_season_stats": ["season", "game_type", "team", "player_id"],
    "standings":           ["team_abbr"],
}
FK_SPEC = [  # (child, child cols, parent, parent cols)
    ("schedule", ["home_team"], "teams", ["team_abbr"]),
    ("schedule", ["away_team"], "teams", ["team_abbr"]),
    ("period_scores", ["game_id", "team"], "team_game_stats", ["game_id", "team"]),
    ("goals", ["game_id"], "schedule", ["game_id"]),
    ("goals", ["scorer_id"], "players", ["player_id"]),
    ("team_game_stats", ["game_id"], "schedule", ["game_id"]),
    ("team_game_stats", ["team"], "teams", ["team_abbr"]),
    ("player_game_stats", ["game_id", "team"], "team_game_stats", ["game_id", "team"]),
    ("player_game_stats", ["player_id"], "players", ["player_id"]),
    ("goalie_game_stats", ["game_id", "team"], "team_game_stats", ["game_id", "team"]),
    ("goalie_game_stats", ["player_id"], "players", ["player_id"]),
    ("player_season_stats", ["player_id"], "players", ["player_id"]),
    ("player_season_stats", ["team"], "teams", ["team_abbr"]),
    ("goalie_season_stats", ["player_id"], "players", ["player_id"]),
    ("standings", ["team_abbr"], "teams", ["team_abbr"]),
]
INDEX_SPEC = {
    "schedule": ["game_date", "home_team", "away_team", "ended_in"],
    "team_game_stats": ["team", "game_date"], "period_scores": ["team", "period"],
    "goals": ["scorer", "team", "period"],
    "player_game_stats": ["player_name", "team", "game_date", "position"],
    "goalie_game_stats": ["player_name", "team", "game_date"],
    "players": ["player_name", "team"], "player_season_stats": ["player_name", "team"],
}


def _verify_pk(df, cols):
    absent = [c for c in cols if c not in df.columns]
    if absent:
        return False, f"columns absent: {absent}"
    sub = df[cols]
    if sub.isna().any().any():
        return False, f"{int(sub.isna().any(axis=1).sum()):,} rows have a NULL in the key"
    dup = int(len(sub) - len(sub.drop_duplicates()))
    return (False, f"{dup:,} duplicate key rows") if dup else (True, f"{len(sub):,} rows, all distinct")


def _verify_fk(child, ccols, parent, pcols):
    absent = [c for c in ccols if c not in child.columns] + [c for c in pcols if c not in parent.columns]
    if absent:
        return False, f"columns absent: {absent}"
    ck = child[ccols].dropna()
    if ck.empty:
        return False, "no non-null child keys"
    pk = parent[pcols].drop_duplicates()
    pk.columns = ccols
    m = ck.merge(pk, on=ccols, how="left", indicator=True)
    orphans = m[m["_merge"] == "left_only"]
    if len(orphans):
        ex = orphans[ccols].drop_duplicates().head(3).to_dict("records")
        return False, f"{len(orphans):,} child rows point at no parent, e.g. {ex}"
    return True, f"{len(ck):,} child rows all resolve"


def publish(path):
    """POST one file to /upload. overwrite=1 is REQUIRED unattended (no one to answer the collision dialog)."""
    import requests
    try:
        with open(path, "rb") as fh:
            r = requests.post(f"{PROLYT_URL}/upload", headers={"X-API-Key": PROLYT_KEY},
                              files={"file": (os.path.basename(path), fh, "application/octet-stream")},
                              data={"overwrite": "1"}, timeout=600)
    except Exception as e:      # one file's network/TLS error must not abort the others
        print(f"  FAIL {os.path.basename(path):<46} {type(e).__name__}: {str(e)[:80]}")
        return False
    ok = r.status_code == 200
    print(f"  {'OK  ' if ok else 'FAIL'} {os.path.basename(path):<46} {r.status_code}  {'' if ok else r.text[:90]}")
    return ok


def ingest(out_dir: str, prefix: str) -> None:
    try:
        import duckdb
    except ImportError:
        print("\n! duckdb not installed — CSVs written, DuckDB step skipped. Publishing still runs.")
        duckdb = None
    frames, paths = {}, {}
    for f in sorted(os.listdir(out_dir)):
        if f.endswith(".csv") and f.startswith(prefix + "_"):
            short = f[len(prefix) + 1:-4]
            if short in PK_SPEC:                            # only this script's tables (the odds ledger lives here too)
                frames[short] = pd.read_csv(f"{out_dir}/{f}", low_memory=False)
                paths[short] = f"{out_dir}/{f}"

    print("\n=== Verifying keys before declaring them ===")
    good_pk, bad_pk, good_fk, bad_fk = {}, {}, [], []
    for t, cols in PK_SPEC.items():
        if t in frames:
            ok, why = _verify_pk(frames[t], cols)
            (good_pk if ok else bad_pk)[t] = (cols, why)
            print(f"  {'PK OK' if ok else 'PK NO'} {t:<20} {'+'.join(cols):<36} {why}")
    for c, cc, p, pc in FK_SPEC:
        if c not in frames or p not in frames:
            continue
        if p not in good_pk:
            bad_fk.append((c, cc, p, pc, "parent PK not verified")); print(f"  FK NO {c} -> {p}: parent PK not verified"); continue
        ok, why = _verify_fk(frames[c], cc, frames[p], pc)
        (good_fk.append((c, cc, p, pc)) if ok else bad_fk.append((c, cc, p, pc, why)))
        print(f"  {'FK OK' if ok else 'FK NO'} {c}.{'+'.join(cc)} -> {p:<16} {why}")

    db_path, rel_path = f"{out_dir}/{prefix}.duckdb", f"{out_dir}/{prefix}_relationships.json"
    if duckdb is not None:
        if os.path.exists(db_path):
            os.remove(db_path)
        con = duckdb.connect(db_path)
        q = lambda cs: ", ".join('"' + c + '"' for c in cs)  # noqa: E731
        order, seen = [], set()

        def visit(t):               # parents before children so FKs can be declared
            if t in seen or t not in frames:
                return
            seen.add(t)
            for c, _cc, p, _pc in good_fk:
                if c == t and p != t:
                    visit(p)
            order.append(t)
        for t in frames:
            visit(t)
        n_con = n_idx = 0
        for t in order:
            df = frames[t]
            con.register("_src", df)
            pk = good_pk.get(t)
            fks = [(cc, p, pc) for (c, cc, p, pc) in good_fk if c == t]
            try:
                if pk or fks:
                    defs = [f'"{n}" {ty}' for n, ty, *_ in con.execute("DESCRIBE _src").fetchall()]
                    if pk:
                        defs.append(f"PRIMARY KEY ({q(pk[0])})")
                    defs += [f'FOREIGN KEY ({q(cc)}) REFERENCES "{prefix}_{p}" ({q(pc)})'
                             for cc, p, pc in fks if good_pk.get(p, ([None],))[0] == pc]
                    con.execute(f'CREATE TABLE "{prefix}_{t}" ({", ".join(defs)})')
                    con.execute(f'INSERT INTO "{prefix}_{t}" SELECT * FROM _src')
                    n_con += 1
                else:
                    con.execute(f'CREATE TABLE "{prefix}_{t}" AS SELECT * FROM _src')
            except Exception as e:
                print(f"  ! {t}: constraints rejected ({str(e)[:90]}) — written without them")
                con.execute(f'DROP TABLE IF EXISTS "{prefix}_{t}"')
                con.execute(f'CREATE TABLE "{prefix}_{t}" AS SELECT * FROM _src')
            con.unregister("_src")
            for col in INDEX_SPEC.get(t, []):
                if col in df.columns:
                    try:
                        con.execute(f'CREATE INDEX "{f"idx_{t}_{col}"[:60]}" ON "{prefix}_{t}" ("{col}")')
                        n_idx += 1
                    except Exception:
                        pass
        con.close()
        print(f"\n=== DuckDB written === {db_path}  {os.path.getsize(db_path) / 1e6:.1f} MB · "
              f"{len(frames)} tables · {n_con} constrained · {n_idx} indexes")

    with open(rel_path, "w", encoding="utf-8") as f:
        json.dump({"_about": "NHL relationships VERIFIED against the data by this run. one_table is the ONE side: "
                             "summing one of ITS columns across a join to many_table double-counts it. "
                             "schedule.home_score/away_score = official final (a shootout adds one goal to the "
                             "winner); home_reg/away_reg = score after 60 minutes, which three-way (1X2) bets "
                             "settle on. Player goals never include shootout attempts. A skater with "
                             "toi_seconds = 0 dressed but did not play.",
                   "generated": pd.Timestamp.now(tz="UTC").isoformat(),
                   "primary_keys": {f"{prefix}_{t}": cols for t, (cols, _) in good_pk.items()},
                   "relationships": [{"one_table": f"{prefix}_{p}", "one_key": pc, "many_table": f"{prefix}_{c}",
                                      "many_key": cc, "cardinality": "1:N", "evidence": "verified-at-build"}
                                     for (c, cc, p, pc) in good_fk],
                   "skipped": ([{"table": t, "key": cols, "reason": why} for t, (cols, why) in bad_pk.items()]
                               + [{"child": c, "parent": p, "reason": w} for (c, _cc, p, _pc, w) in bad_fk])},
                  f, indent=2, default=str)

    if not PROLYT_KEY:
        print("\n=== Publish skipped === PROLYT_API_KEY / ANALYTIQ_API_KEY not set — files written locally only.")
        return
    print(f"\n=== Publishing to {PROLYT_URL} ===")
    sent = sum(publish(p) for p in paths.values())
    publish(rel_path)
    print(f"  {sent} of {len(paths)} tables published."
          + ("" if sent == len(paths) else "  Failures are usually the key (401), size (413), or the host being down."))
    if sent < len(paths):
        sys.exit(1)     # make the Actions run RED, not a silent no-op


def current_season(today: dt.date | None = None) -> int:
    today = today or dt.date.today()
    return today.year if today.month >= 7 else today.year - 1   # Jan-Jun still belongs to last season


if __name__ == "__main__":
    out = os.environ.get("NHL_OUT", "./data")
    season = int(os.environ.get("NHL_SEASON", "") or current_season())
    types = {int(x) for x in os.environ.get("NHL_GAME_TYPES", "2").split()}
    prefix = main(season, out, types)
    ingest(out, prefix)
