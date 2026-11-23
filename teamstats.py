#!/usr/bin/env python3
"""Team stat sheet and side-by-side comparison for NHL games, from nhl_ingest.py's tables.

    python teamstats.py ranks   DATA_DIR [--date 2026-11-23]                 # every team, every stat, rank of 32
    python teamstats.py compare DATA_DIR --date 2026-11-23 [--db ledger]     # tonight's games side by side + the book
    python teamstats.py lean    DATA_DIR [--db ledger]                       # did following the stats beat the book?

Everything is computed from games BEFORE the date, so a comparison never knows the result it is compared with.
Opponent-adjusted (marked adj): goals and shots are solved together with every opponent's strength, so ten games
against weak teams do not look like ten good games. Ranks: 1 = best for the team (most goals for, fewest against).
"""
from __future__ import annotations

import argparse
import glob
import math
import sqlite3

import numpy as np
import pandas as pd


def _t(d, name):
    return pd.read_csv(sorted(glob.glob(f"{d}/nhl_*_{name}.csv"))[-1], low_memory=False, dtype={"period": str})


def load(d):
    tg, ps, gl, sch = _t(d, "team_game_stats"), _t(d, "period_scores"), _t(d, "goals"), _t(d, "schedule")
    for x in (tg, ps, gl):
        x["d"] = x.game_date.astype(str).str[:10]
    sch["d"] = sch.game_date.astype(str).str[:10]
    return tg, ps, gl, sch


def _adjust(g: pd.DataFrame, f: str, a: str, rounds: int = 30, prior: int = 10):
    """Multiplicative attack / defence ratings: f_ij = lg * att_i * def_j. Shrunk toward 1 with `prior` games."""
    lg = g[f].mean()
    att = pd.Series(1.0, index=sorted(g.team.unique()))
    dfn = att.copy()
    n = g.groupby("team").size()
    for _ in range(rounds):
        exp_f = g.opponent.map(dfn) * lg
        att = (g[f].groupby(g.team).sum() + prior * lg) / (exp_f.groupby(g.team).sum() + prior * lg)
        exp_a = g.opponent.map(att) * lg
        dfn = (g[a].groupby(g.team).sum() + prior * lg) / (exp_a.groupby(g.team).sum() + prior * lg)
    return att * lg, dfn * lg, n


def sheet(d: str, before: str | None = None) -> pd.DataFrame:
    tg, ps, gl, sch = load(d)
    if before:
        tg, ps, gl = tg[tg.d < before], ps[ps.d < before], gl[gl.d < before]
    g = tg.copy()
    T = pd.DataFrame(index=sorted(g.team.unique()))
    gp = g.groupby("team").size()
    T["gp"] = gp
    T["gf_adj"], T["ga_adj"], _ = _adjust(g, "goals_reg", "goals_against_reg")
    T["sf_adj"], T["sa_adj"], _ = _adjust(g, "shots_on_goal", "shots_against")
    s = g.groupby("team")
    T["gf"], T["ga"] = s.goals_reg.mean(), s.goals_against_reg.mean()
    T["sh_pct"] = s.goals_reg.sum() / s.shots_on_goal.sum()
    T["sv_pct"] = 1 - s.goals_against_reg.sum() / s.shots_against.sum()
    opp_pp = g.set_index(["game_id", "team"]).pp_goals.rename("x").reset_index()
    pp_against = g.merge(opp_pp.rename(columns={"team": "opponent", "x": "pp_against"}), on=["game_id", "opponent"])
    T["ppg"] = s.pp_goals.mean()
    T["ppga"] = pp_against.groupby("team").pp_against.mean()
    T["pim"] = s.pim.mean()
    T["hits"], T["blocks"] = s.hits.mean(), s.blocked_shots.mean()
    T["take_give"] = s.takeaways.mean() - s.giveaways.mean()
    T["first_goal"] = s.scored_first.mean()
    T["tie60"] = s.tie_reg.mean()
    T["win"] = s.win.mean()
    T["reg_win"] = s.win_reg.mean()
    for per in ("1", "2", "3"):
        x = ps[ps.period == per].merge(ps[ps.period == per][["game_id", "team", "goals"]].rename(columns={"team": "opp", "goals": "ga"}), on="game_id")
        x = x[x.team != x.opp]
        T[f"p{per}_gf"], T[f"p{per}_ga"] = x.groupby("team").goals.mean(), x.groupby("team").ga.mean()
    en = gl[gl.empty_net.astype(bool)]
    T["en_for"] = en.groupby("team").size().reindex(T.index).fillna(0) / gp
    last = g.sort_values("d").groupby("team").tail(10).groupby("team")
    T["l10_w"] = last.win.sum()
    T["l10_gd"] = last.goals_reg.mean() - last.goals_against_reg.mean()
    home = g[g.is_home.astype(bool)].groupby("team")
    away = g[~g.is_home.astype(bool)].groupby("team")
    T["home_gd"] = home.goals_reg.mean() - home.goals_against_reg.mean()
    T["away_gd"] = away.goals_reg.mean() - away.goals_against_reg.mean()
    T["gd_adj"] = T.gf_adj - T.ga_adj
    # rank 1 = best for the team
    low_is_good = {"ga_adj", "sa_adj", "ga", "ppga", "pim", "p1_ga", "p2_ga", "p3_ga"}
    for c in T.columns:
        if c in ("gp",):
            continue
        T[f"r_{c}"] = T[c].rank(ascending=c in low_is_good, method="min").astype(int)
    return T


ROWS = [  # label, columns, format
    ("Goals for / against per game (adj, 60 min)", ("gf_adj", "ga_adj"), "{:.2f}"),
    ("Goal difference per game (adj)", ("gd_adj",), "{:+.2f}"),
    ("Shots for / against per game (adj)", ("sf_adj", "sa_adj"), "{:.1f}"),
    ("Shooting % / save %", ("sh_pct", "sv_pct"), "{:.1%}"),
    ("Power-play goals for / against per game", ("ppg", "ppga"), "{:.2f}"),
    ("1st period goals for / against", ("p1_gf", "p1_ga"), "{:.2f}"),
    ("2nd period goals for / against", ("p2_gf", "p2_ga"), "{:.2f}"),
    ("3rd period goals for / against", ("p3_gf", "p3_ga"), "{:.2f}"),
    ("Scores first", ("first_goal",), "{:.0%}"),
    ("Level after 60 min", ("tie60",), "{:.0%}"),
    ("Wins / regulation wins", ("win", "reg_win"), "{:.0%}"),
    ("Penalty minutes per game", ("pim",), "{:.1f}"),
    ("Hits / blocked shots per game", ("hits", "blocks"), "{:.1f}"),
    ("Last 10: wins, goal difference per game", ("l10_w", "l10_gd"), None),
    ("Goal difference home / away", ("home_gd", "away_gd"), "{:+.2f}"),
]


def cell(T, team, cols, fmt):
    if fmt is None:
        return f"{int(T.loc[team, cols[0]])} W, {T.loc[team, cols[1]]:+.2f}"
    parts = [fmt.format(T.loc[team, c]) + (f" ({T.loc[team, 'r_' + c]})" if 'r_' + c in T else "") for c in cols]
    return " / ".join(parts)


def expected_total(T, h, a, home_ice=1.025):
    lg = T.gf_adj.mean()
    lh = T.loc[h, "gf_adj"] * T.loc[a, "ga_adj"] / lg * home_ice
    la = T.loc[a, "gf_adj"] * T.loc[h, "ga_adj"] / lg / home_ice
    return lh, la


def compare(d: str, date: str, db: str = "ledger/nhl.sqlite"):
    T = sheet(d, date)
    con = sqlite3.connect(db)
    B = pd.read_sql("SELECT * FROM predictions WHERE night = ? AND p_method = 'no_vig'", con, params=[date])
    if B.empty:
        sch = load(d)[3]
        games = [(r.home_team, r.away_team) for r in sch[sch.d == date].itertuples()]
    else:
        B = B[B.captured == B.groupby("game").captured.transform("max")]
        games = sorted({(r.home, r.away) for r in B.itertuples()})
    out = []
    for h, a in games:
        if h not in T.index or a not in T.index:
            continue
        lines = [f"## {a} @ {h}", "", f"| | {a} (away) | {h} (home) |", "|---|---|---|"]
        for lab, cols, fmt in ROWS:
            lines.append(f"| {lab} | {cell(T, a, cols, fmt)} | {cell(T, h, cols, fmt)} |")
        lh, la = expected_total(T, h, a)
        g = B[B.game == f"{a}@{h}"]
        ml = g[(g.type == "ML") & (g.scope == "G") & (g.side == "home")]
        tot = g[(g.type == "TOTAL") & (g.scope == "G") & (g.ot == 1)]
        book_h = float(ml.p_book.iloc[0]) if len(ml) else float("nan")
        if len(tot):
            ov = tot[tot.side == "over"].assign(dist=lambda z: (z.p_book - 0.5).abs()).sort_values("dist").iloc[0]
            book_line = f"{ov.line:g} (over {ov.p_book:.0%})"
        else:
            book_line = "-"
        stats_lean = h if T.loc[h, "gd_adj"] + 0.2 > T.loc[a, "gd_adj"] else a        # +0.2 per game for home ice
        book_lean = h if book_h >= 0.5 else a
        flag = "" if stats_lean == book_lean else "  **stats and book disagree on the winner**"
        lines += ["", f"Stats: {h} {lh:.2f} - {a} {la:.2f} expected regulation goals (total {lh + la:.2f}); "
                      f"better goal difference: {stats_lean}.  Book: {h} to win {book_h:.0%}, total line {book_line}.{flag}", ""]
        out.append("\n".join(lines))
    text = f"# NHL comparison, {date} (stats from games before this date)\n\nRanks of 32 in brackets, 1 = best.\n\n" + "\n".join(out)
    print(text)
    return text


def lean(d: str, db: str = "ledger/nhl.sqlite"):
    """Every graded game line in the ledger, tagged 'stats agree' or 'stats disagree' using only games before its
    night; flat return of each group, counted in games."""
    con = sqlite3.connect(db)
    L = pd.read_sql("SELECT * FROM predictions WHERE result IN ('won', 'lost') AND p_method = 'no_vig' AND scope = 'G' "
                    "AND type IN ('ML', 'HANDICAP', 'TOTAL', 'TEAM_TOTAL')", con)
    rows = []
    for night, x in L.groupby("night"):
        T = sheet(d, night)
        for r in x.itertuples():
            if r.home not in T.index or r.away not in T.index:
                continue
            lh, la = expected_total(T, r.home, r.away)
            if r.type in ("ML", "HANDICAP"):
                gd = (T.loc[r.home, "gd_adj"] + 0.2) - T.loc[r.away, "gd_adj"]
                edge = gd if r.side == "home" else -gd
                kind = "side"
            elif r.type == "TOTAL":
                edge = (lh + la - r.line) * (1 if r.side == "over" else -1)
                kind = "total"
            else:
                x_ = lh if r.subject == r.home else la
                edge = (x_ - r.line) * (1 if r.side == "over" else -1)
                kind = "team total"
            rows.append(dict(night=night, game=r.game, kind=kind, agree=edge > 0, strong=abs(edge) >= 0.5, p=r.p_book,
                             odds=r.odds, y=float(r.result == "won")))
    D = pd.DataFrame(rows)
    D["ret"] = D.y * D.odds - 1
    D["cl"] = D.night + D.game

    def st(x):
        g = x.groupby("cl").ret.mean()
        se = g.std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float("nan")
        return pd.Series(dict(bets=len(x), games=len(g), won=x.y.mean(), book_said=x.p.mean(), ret=g.mean(), plus_minus=2 * se))
    print(f"{len(D):,} graded game lines, {D.cl.nunique()} games. 'agree' = the stat comparison points the same way as the bet.")
    print(D.groupby(["kind", "agree"]).apply(st).round(3).to_string())
    print("\nonly where the stats lean clearly (0.5+ goals a game):")
    print(D[D.strong].groupby(["kind", "agree"]).apply(st).round(3).to_string())
    return D


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["ranks", "compare", "lean"])
    ap.add_argument("data")
    ap.add_argument("--date")
    ap.add_argument("--db", default="ledger/nhl.sqlite")
    a = ap.parse_args()
    if a.cmd == "ranks":
        T = sheet(a.data, a.date)
        cols = ["gp", "gd_adj", "r_gd_adj", "gf_adj", "r_gf_adj", "ga_adj", "r_ga_adj", "sf_adj", "sa_adj", "sh_pct", "sv_pct",
                "ppg", "ppga", "first_goal", "tie60", "l10_w", "l10_gd"]
        print(T[cols].sort_values("r_gd_adj").round(3).to_string())
    elif a.cmd == "compare":
        compare(a.data, a.date, a.db)
    else:
        lean(a.data, a.db)
