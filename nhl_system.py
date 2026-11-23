#!/usr/bin/env python3
"""NHL grading + slip system for vegas.hu boards (Altenar .net.json captures from capture_board.py). One file,
stdlib + numpy + pandas + tabulate. Spec: NHL_SYSTEM.md (same folder).

    python nhl_system.py selftest                                 # must print PASS before anything else is trusted
    python nhl_system.py markets BOARDS_DIR                       # every market name on the boards and how it is typed
    python nhl_ingest.py                                          # all box scores -> data/nhl_<season>_*.csv
    python nhl_system.py teams   --date 2026-11-17 --ingest data [--out teams.csv]   # team goals, back-to-backs
    python nhl_system.py price   --date 2026-11-17 --boards DIR [--teams teams.csv] [--b2b TOR,BOS]
    python nhl_system.py slips   --date 2026-11-17 [--targets 3 5 7] [--exclude "BOS@TOR" ...] [--players] [--replace]
    python nhl_system.py results --date 2026-11-17 --ingest data --out results/2026-11-17.txt
    python nhl_system.py grade   --date 2026-11-17 --results results/2026-11-17.txt
    python nhl_system.py report  [--test-night 2026-11-30]
    python nhl_system.py arms                                     # the bandit's view of each kind of leg
    python nhl_system.py compare --date 2026-11-23 --ingest data  # both teams' stats side by side, with the book
    python nhl_system.py sync    --ingest data                    # grade every night that has new finished games

Everything is stored in nhl.sqlite next to this file (override: NHL_DB=path).
Principles: the book is the anchor (p = calibrated book price + w * lean); every number carries its n; games, not
rows, are the unit of evidence; every slip that is built is stored and graded (no cherry-picking); a slip is PAPER
until the leg gate in the report passes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

DB = os.environ.get("NHL_DB", str(Path(__file__).with_name("nhl.sqlite")))
MODEL_VERSION = "nhl-0.7"
API = "https://api-web.nhle.com/v1"
# word that means "goals" in a market name. 'gólpassz' is an assist, so it is excluded.
GOAL = re.compile(os.environ.get("NHL_GOAL_WORD", r"g[oó]l(?!passz)"), re.I)
PERIOD = re.compile(r"(\d)\.\s*harmad", re.I)                 # '1. harmad' = 1st period
START_W = {"main": 0.05, "period": 0.0, "player": 0.0, "other": 0.0}      # model weight before any refit
LEAN_CAP, MIN_GAMES, GATE_N = 0.05, 30, 300
LEG_P, LEG_MIN_ODDS = (0.60, 0.88), 1.15         # slip legs: win chance band and lowest price
ARM_K0 = 300      # bandit prior: worth this many games of "the arm pays what its price says". Simulation: with 50, arms of
#                   a book with NO edge moved +-6 points after 400 games; with 300 they move +-2 and a real +4-point arm
#                   is the top arm in 3 of 3 runs only by ~1,300 games. Nothing learns this faster.
CAL_GAMES, DECIDE_GAMES = 100, 300               # graded games before the first read / before a verdict
TIE_K = 1.45        # tie cells of the score grid x this: independent Poisson gives ~16.5% regulation draws, the NHL has
#                     had ~22-23% for years (teams play for the point late). `teams --ingest` prints this season's share.
# Checked on the first 330 games of 2026-27 (2026-11-16): 22.4% level after 60 min (TIE_K gives 22.3%); home / away
# regulation goals 3.08 / 3.03, ratio 1.016 +- 0.04, so home ice is cut from 1.04 to 1.025 per side (ratio 1.05, the
# long-run size); period shares 30.0 / 35.3 / 34.7%; back-to-back teams scored 2.68 against 3.11 (78 team-games,
# mixed up with travel and team quality): 0.96 kept.
HOME_ICE, B2B, REGRESS_GP = 1.025, 0.96, 25
PERIOD_SHARE = {"P1": 0.30, "P2": 0.35, "P3": 0.35}       # goal-rate multipliers; games of league average mixed in
# odd typeIds are the sportsbook's own selection codes (same in every sport on this site)
HOME, DRAW, AWAY, OVER, UNDER, HCP_H, HCP_A, ODD, EVEN, YES, NO = 1, 2, 3, 12, 13, 1714, 1715, 70, 72, 74, 76

SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (night TEXT, row_key TEXT, game TEXT, home TEXT, away TEXT, market TEXT,
  type TEXT, scope TEXT, ot INTEGER, subject TEXT, selection TEXT, side TEXT, line REAL, odds REAL, p_book REAL,
  p_model REAL, p REAL, w_used REAL, p_method TEXT, overround REAL, captured TEXT, model_version TEXT,
  status TEXT DEFAULT 'open', result TEXT, actual REAL, p_cal REAL, PRIMARY KEY (night, row_key));
CREATE TABLE IF NOT EXISTS weights (as_of TEXT, family TEXT, games INTEGER, w_fit REAL, w_se REAL, w_used REAL);
CREATE TABLE IF NOT EXISTS lean_legs (night TEXT, game TEXT, kind TEXT, agree INTEGER, strong INTEGER, p REAL, odds REAL,
  y REAL, PRIMARY KEY (night, game, kind, agree, p, odds));
CREATE TABLE IF NOT EXISTS synced (night TEXT, game TEXT, PRIMARY KEY (night, game));
CREATE TABLE IF NOT EXISTS calib (as_of TEXT, games INTEGER, a REAL, b REAL);
CREATE TABLE IF NOT EXISTS slips (night TEXT, slip_id TEXT, target REAL, odds REAL, p REAL, ev REAL, legs INTEGER,
  built TEXT, status TEXT DEFAULT 'open', result TEXT, paid REAL, PRIMARY KEY (night, slip_id));
CREATE TABLE IF NOT EXISTS slip_legs (night TEXT, slip_id TEXT, row_key TEXT, game TEXT, market TEXT,
  selection TEXT, odds REAL, p REAL, p_book REAL, captured TEXT, PRIMARY KEY (night, slip_id, row_key));
"""


def connect() -> sqlite3.Connection:
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    if 'p_cal' not in {r[1] for r in con.execute('PRAGMA table_info(predictions)')}:      # ledger from nhl-0.1
        con.execute('ALTER TABLE predictions ADD COLUMN p_cal REAL')
    return con


# ------------------------------------------------------------------ boards
def load_event(path) -> dict:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(d, dict) and "markets" in d:
        return d
    hits = [g["body"] for g in d if isinstance(g.get("body"), dict) and "markets" in g["body"] and "odds" in g["body"]]
    if not hits:
        raise ValueError(f"{path}: no GetEventDetails response in it")
    return max(hits, key=lambda b: len(b["odds"]))


def shown(price: float) -> float:
    """The price as the site shows and pays it: floored to 2 decimals."""
    return math.floor(price * 100 + 1e-6) / 100


def _line(o: dict):
    sv = str(o.get("sv") or "").split("|")[0].replace(",", ".")
    try:
        return float(sv)
    except ValueError:
        m = re.search(r"([+\-]?\d+(?:[.,]\d+)?)", o.get("name", ""))
        return float(m.group(1).replace(",", ".")) if m else None


BOOK_CODE = {"LA": "LAK", "NJ": "NJD", "SJ": "SJS", "TB": "TBL", "VEG": "VGK", "WAS": "WSH", "MON": "MTL",
             "NAS": "NSH", "CLB": "CBJ", "CAL": "CGY", "WIN": "WPG", "UTAH": "UTA"}     # the book's code -> the NHL's


def abbr(c: dict) -> str:
    a = (c.get("abbreviation") or "".join(w[0] for w in c["name"].split())[:3]).upper()
    return BOOK_CODE.get(a, a)


def classify(name: str, tids: set, teams: tuple[str, str], player: bool) -> tuple[str, str, int, str | None]:
    """-> (type, scope, ot, team-subject). scope: G (game) or P1/P2/P3. ot: 1 = overtime+shootout count.
    Structural: decided by WHICH selection codes the market has, the name only gives scope / stat / team."""
    low = name.lower()
    m = PERIOD.search(low)
    scope = f"P{m.group(1)}" if m else "G"
    ot = int(scope == "G" and "beleértve" in low and "nélkül" not in low)
    team = next((t for t in teams if t.lower() in low), None)
    if scope == "G" and re.match(r"\s*\d\s*\.", low):            # a numbered part of the game that is not a period
        return "UNPARSED", scope, ot, team
    if player:
        if re.search(r"powerplay|emberelőny|kapott gól", low):       # power-play points, shutout: not in the results
            return "UNPARSED", scope, ot, None
        stat = ("BLK" if "blokkolt" in low else "HITS" if "ütközés" in low else "AST" if "gólpassz" in low or "asszisz" in low else "SOG" if "lövés" in low or "kapura" in low else
                "SAVES" if "védés" in low or "mentés" in low else "PTS" if "pont" in low else "GOALS" if GOAL.search(low) else None)
        if scope != "G" or stat is None or re.search(r"első|utolsó", low):
            return "UNPARSED", scope, ot, None
        return f"PL_{stat}", scope, 1, None
    bare = re.sub(r"\(.*?\)", "", low)                        # '(hosszabbítást és büntetőket beleértve)' is not a combo
    if re.search(r" és | & |pontos|multi|tartomány|különbség|melyik|hányadik|első|utolsó|ki ér el", bare):
        return "UNPARSED", scope, ot, team                     # combos, bands, races: not graded in v0
    if tids == {HOME, AWAY}:
        if scope != "G" or "döntetlennél" in low or "nélkül" in low:        # draw no bet (regulation / one period)
            return "DNB", scope, 0, None
        return "ML", scope, 1, None                                        # NHL moneyline: overtime and shootout count
    if tids == {HOME, DRAW, AWAY}:
        return "1X2", scope, 0, None
    if tids and tids <= {HCP_H, HCP_A}:
        return "HANDICAP", scope, ot, None
    if tids and tids <= {OVER, UNDER}:
        goals = GOAL.search(bare) or (team and "pontok száma" in bare)      # '<team> pontok száma' = that team's goals
        if not goals or re.search(r"lövés|kapura|büntet|perc|emberelőny", bare):
            return "UNPARSED", scope, ot, team                 # shots, penalty minutes ...: add when results carry them
        return ("TEAM_TOTAL" if team else "TOTAL"), scope, ot, team
    if tids == {YES, NO} and "mindkét csapat" in low:
        k = re.search(r"(\d+) vagy több", low)                  # 'both teams score 2 or more'
        if k:
            return ("BTTS2" if k.group(1) == "2" else "UNPARSED"), scope, ot, None
        return "BTTS", scope, ot, None
    if tids == {YES, NO} and "hosszabbítás" in low and "lesz" in low:
        return "OT", "G", 1, None
    if tids == {ODD, EVEN}:
        return "ODD_EVEN", scope, ot, team
    if "dupla esély" in low or "kétesély" in low:
        return "DC", scope, 0, None
    return "UNPARSED", scope, ot, team


def board_rows(path) -> list[dict]:
    ev = load_event(path)
    comp = ev["competitors"][:2]
    home, away = abbr(comp[0]), abbr(comp[1])
    names = (comp[0]["name"], comp[1]["name"])
    cid = {comp[0]["id"]: home, comp[1]["id"]: away}
    odds = {o["id"]: o for o in ev["odds"]}
    child = {m["id"]: m for m in ev.get("childMarkets", [])}
    rows, seen = [], set()
    captured = pd.Timestamp(os.path.getmtime(path), unit="s").isoformat()

    def emit(m, parent_name, who):
        ids = [i for col in m.get("desktopOddIds", []) for i in col]
        live = [odds[i] for i in ids if i in odds and odds[i].get("oddStatus", 0) == 0 and shown(odds[i].get("price", 0)) > 1]
        if not live:
            return
        tids = {o.get("typeId") for o in live}
        typ, scope, ot, team = classify(parent_name, tids, names, who is not None)
        for o in live:
            t, nm, line = o.get("typeId"), o.get("name", ""), _line(o)
            side, subject = None, who or (cid.get(o.get("competitorId")) if typ != "UNPARSED" else None)
            if who is None and typ == "UNPARSED" and re.search(r"gólszerző|szerez gólt", parent_name.lower()) \
                    and "(" in nm and not re.search(r"első|utolsó", parent_name.lower()):
                typ_o, subject, side, line = "PL_GOALS", re.sub(r"\s*\(.*", "", nm), "over", 0.5
            else:
                typ_o = typ
            if typ_o.startswith("PL_") and side is None:
                if t in (OVER, UNDER) or re.search(r"felett|alatt", nm):       # player tables use their own codes
                    side = "over" if (t == OVER or "felett" in nm) else "under"
                else:                                          # ladder rung 'N+'
                    k = re.search(r"(\d+)\s*\+", nm)
                    side, line = ("over", int(k.group(1)) - 0.5) if k else (None, line)
            elif typ_o in ("ML", "DNB", "1X2"):
                side = {HOME: "home", DRAW: "draw", AWAY: "away"}.get(t)
            elif typ_o == "HANDICAP":
                side = "home" if t == HCP_H else "away"
                own = re.search(r"\(([+\-]?\d+(?:[.,]\d+)?)\)\s*$", nm)   # the selection's OWN line is in its name;
                if own:                                                     # the sv field is the home line for both
                    line = float(own.group(1).replace(",", "."))
                elif side == "away" and line is not None:
                    line = -line
            elif typ_o in ("TOTAL", "TEAM_TOTAL"):
                side = "over" if t == OVER else "under"
                subject = (home if team == names[0] else away) if team else None
            elif typ_o in ("BTTS", "BTTS2", "OT"):
                side = "yes" if t == YES else "no"
            elif typ_o == "ODD_EVEN":
                side = "odd" if t == ODD else "even"
                subject = (home if team == names[0] else away) if team else None
            elif typ_o == "DC" and t in (9, 10, 11):               # the site's own codes: 1X, 12, X2
                side = {9: "1X", 10: "12", 11: "X2"}[t]
            elif typ_o == "DC":
                h, a, x = names[0] in nm or "1" in nm.split()[0:1], names[1] in nm or "2" in nm, "öntetlen" in nm or "X" in nm
                side = "1X" if (h and x and not a) else "X2" if (a and x and not h) else "12" if (h and a) else None
            if side is None and typ_o != "UNPARSED":
                typ_o = "UNPARSED"
            # the key is the bet in words, not the book's ids: the book re-issues ids when it re-opens a market, and a
            # second capture must land on the same row (found on 2026-11-17: 3 of 19 slip legs "vanished" by id only)
            key = hashlib.md5(f"{away}@{home}|{parent_name}|{who or ''}|{nm}".encode()).hexdigest()[:16]
            id_key = hashlib.md5(f"{m['id']}|{o['id']}|{nm}".encode()).hexdigest()[:16]
            rows.append(dict(row_key=key, id_key=id_key, game=f"{away}@{home}", home=home, away=away, market=parent_name, type=typ_o,
                             scope=scope, ot=ot, subject=subject, selection=nm, side=side, line=line,
                             odds=shown(o["price"]), mid=m["id"], captured=captured))

    for m in ev["markets"]:
        if m["id"] in seen or m.get("typeId") == 31046:        # boosted-odds container
            continue
        seen.add(m["id"])
        kids = [child[c] for c in m.get("childMarketIds", []) if c in child]
        if kids:
            for c in kids:
                emit(c, m["name"].strip(), (c.get("childName") or "").strip() or None)
        else:
            emit(m, m["name"].strip(), None)
    return rows


def devig(B: pd.DataFrame) -> pd.DataFrame:
    """Book probability with the margin removed. A group = selections that cover every outcome between them:
    1X2 / ML / yes-no / odd-even by market; over-under by line; handicap home L with away -L.
    Groups whose implied probabilities add to 1.00-1.25 are normalised ('no_vig'); anything else (ladders, lone
    selections) is divided by the board's median overround ('assumed_margin') and never used as a leg."""
    B = B.copy()
    B["imp"] = 1 / B.odds
    ln = B.line.where(B.line.notna(), 0.0)
    ln = np.where((B.type == "HANDICAP") & (B.side == "away"), -ln, ln)
    two_sided = B.type.isin(["TOTAL", "TEAM_TOTAL", "HANDICAP"]) | (B.type.str.startswith("PL_") & B.side.isin(["over", "under"]))
    B["grp"] = B.mid.astype(str) + np.where(two_sided, "|" + pd.Series(ln).round(2).astype(str), "") + \
        "|" + B.subject.fillna("").where(B.type.str.startswith("PL_"), "")
    g = B.groupby("grp")
    B["overround"], n, sides = g.imp.transform("sum"), g.imp.transform("size"), g.side.transform("nunique")
    need = np.where(B.type == "1X2", 3, np.where(B.type == "DC", 99, 2))      # DC covers outcomes twice: never no_vig
    ok = (n == need) & (sides == need) & B.overround.between(1.0, 1.25) & (B.type != "UNPARSED")
    med = B.loc[ok, "overround"].median() if ok.any() else 1.08
    B["p_book"] = np.where(ok, B.imp / B.overround, np.minimum(B.imp / med, 0.999))
    B["p_method"] = np.where(ok, "no_vig", f"assumed_margin_{med:.3f}")
    B.loc[~ok, "overround"] = med
    return B


# ------------------------------------------------------------------ model: independent Poisson on regressed goal rates
def team_rates(teams_csv: str) -> tuple[dict, float]:
    """teams.csv: team,gp,gf,ga (season totals, regulation+OT goals, no shootout goals). Optional xgf,xga columns
    replace gf,ga when present. Rates are goals per game relative to league average, regressed to 1.0 with
    REGRESS_GP games of league-average play: 10 games of goal difference is mostly luck."""
    t = pd.read_csv(teams_csv)
    f, a = ("xgf", "xga") if {"xgf", "xga"} <= set(t.columns) else ("gf", "ga")
    lg = t[f].sum() / t.gp.sum()
    k = t.gp / (t.gp + REGRESS_GP)
    t["att"] = 1 + k * (t[f] / t.gp / lg - 1)
    t["dfn"] = 1 + k * (t[a] / t.gp / lg - 1)
    if "reg" not in t.columns:
        lg *= 0.97                                            # totals with overtime goals in them: ~3% are scored in OT
    return {r.team.upper(): (r.att, r.dfn) for r in t.itertuples()}, lg


def score_grid(lh: float, la: float, n: int = 13, tie_k: float | None = None) -> np.ndarray:
    k = np.arange(n)
    fact = np.array([math.factorial(int(i)) for i in k], dtype=float)
    ph, pa = np.exp(-lh) * lh ** k / fact, np.exp(-la) * la ** k / fact
    G = np.outer(ph, pa)
    G[np.arange(n), np.arange(n)] *= TIE_K if tie_k is None else tie_k
    return G / G.sum()


def model_p(r, lh: float, la: float) -> float | None:
    """Model probability of one row. lh/la = expected REGULATION goals (60 min). Periods use a third each.
    Overtime: a regulation tie is split by relative strength (lh/(lh+la)), shrunk halfway to a coin flip (3-on-3
    and shootouts are close to random); the OT winner gets exactly one extra goal."""
    if r.scope != "G":
        lh, la = lh * PERIOD_SHARE.get(r.scope, 1 / 3), la * PERIOD_SHARE.get(r.scope, 1 / 3)
    # tie inflation is a full-game effect (teams play for the point late); single periods are plain Poisson. Measured
    # on 343 games: tied periods 33.5 / 26.8 / 24.2%, plain Poisson 32.6 / 29.4 / 29.5%, with TIE_K 41 / 38 / 38%.
    G = score_grid(lh, la, tie_k=None if r.scope == "G" else 1.0)
    n = G.shape[0]
    H, A = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
    tie = (H == A)
    q = 0.5 + 0.5 * (lh / (lh + la) - 0.5)                   # P(home wins OT/SO | tie)
    ot = bool(r.ot) and r.scope == "G"

    def P(cond_reg, cond_home_ot=None, cond_away_ot=None):
        if not ot:
            return float(G[cond_reg].sum())
        return float(G[cond_reg & ~tie].sum() + q * G[cond_home_ot & tie].sum() + (1 - q) * G[cond_away_ot & tie].sum())

    t, s, L = r.type, r.side, r.line
    if t == "ML":
        ph = P(H > A, tie, ~tie)
        return ph if s == "home" else 1 - ph
    if t == "DNB":
        ph = G[H > A].sum() / max(G[H != A].sum(), 1e-9)
        return float(ph if s == "home" else 1 - ph)
    if t == "1X2":
        return float(G[(H > A) if s == "home" else (H < A) if s == "away" else tie].sum())
    if t == "DC":
        return float(G[(H >= A) if s == "1X" else (H <= A) if s == "X2" else (H != A)].sum())
    if t == "TOTAL" and L is not None and L % 1 == 0.5:
        po = P(H + A > L, H + A + 1 > L, H + A + 1 > L)
        return po if s == "over" else 1 - po
    if t == "TEAM_TOTAL" and L is not None and L % 1 == 0.5:
        X = H if r.subject == r.home else A
        hm = r.subject == r.home
        po = P(X > L, X + (1 if hm else 0) > L, X + (0 if hm else 1) > L)
        return po if s == "over" else 1 - po
    if t == "HANDICAP" and L is not None and L % 1 == 0.5:
        d = (H - A) if s == "home" else (A - H)
        up = 1 if s == "home" else -1
        return P(d + L > 0, d + up + L > 0, d - up + L > 0)
    if t == "BTTS":
        py = float(G[(H > 0) & (A > 0)].sum())
        return py if s == "yes" else 1 - py
    if t == "BTTS2":
        py = float(G[(H > 1) & (A > 1)].sum())
        return py if s == "yes" else 1 - py
    if t == "OT":
        return float(G[tie].sum()) if s == "yes" else float(1 - G[tie].sum())
    return None


# ------------------------------------------------------------------ price
def current_weights(con) -> dict:
    w = dict(START_W)
    for fam, wu in con.execute("SELECT family, w_used FROM weights WHERE as_of = (SELECT max(as_of) FROM weights)"):
        w[fam] = wu
    return w


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def current_calib(con) -> tuple[float, float]:
    r = con.execute("SELECT a, b FROM calib ORDER BY as_of DESC LIMIT 1").fetchone()
    return (float(r[0]), float(r[1])) if r else (0.0, 1.0)


def fit_calib(x: pd.DataFrame, lam: float = 40.0) -> tuple[float, float]:
    """p = sigmoid(a + b * logit(p_book)) on two-way priced rows. b > 1: favourites win more often than the evenly
    de-margined price says (the usual shape). Each game weighs 1 however many rows it has; ridge toward (0, 1)
    (a 400-game fit alone swings b by +-0.1 on a book with no bias at all); refused (identity) outside sane bounds."""
    k = (1 / x.groupby("cluster").cluster.transform("size")).values
    z, y, a, b = _logit(x.p_book.values), x.y.values, 0.0, 1.0
    for _ in range(50):
        p = 1 / (1 + np.exp(-np.clip(a + b * z, -30, 30)))
        w = k * p * (1 - p) + 1e-12
        g0, g1 = (k * (y - p)).sum() - lam * a, (k * (y - p) * z).sum() - lam * (b - 1)
        h00, h01, h11 = w.sum() + lam, (w * z).sum(), (w * z * z).sum() + lam
        det = h00 * h11 - h01 ** 2
        a += float(np.clip((h11 * g0 - h01 * g1) / det, -0.25, 0.25))
        b += float(np.clip((-h01 * g0 + h00 * g1) / det, -0.25, 0.25))
    return (round(a, 4), round(b, 4)) if (abs(a) <= 0.3 and 0.7 <= b <= 1.5) else (0.0, 1.0)


def family(t: str, scope: str) -> str:
    return "player" if t.startswith("PL_") else "period" if scope != "G" else \
        "main" if t in ("ML", "1X2", "DC", "TOTAL", "TEAM_TOTAL", "HANDICAP", "BTTS", "BTTS2", "OT") else "other"


def price(night: str, boards: str, teams_csv: str | None, b2b: set, con) -> pd.DataFrame:
    files = sorted(Path(boards).glob("*.json"))
    if not files:
        sys.exit(f"no .json boards in {boards}")
    B = pd.concat([devig(pd.DataFrame(board_rows(f))) for f in files], ignore_index=True)
    B = B.drop_duplicates(["game", "row_key"], keep="last")
    B["p_model"] = np.nan
    if teams_csv:
        rates, lg_reg = team_rates(teams_csv)                 # league regulation goals per team per game
        for g, x in B.groupby("game"):
            h, a = x.home.iloc[0], x.away.iloc[0]
            if h not in rates or a not in rates:
                print(f"  {g}: team not in {teams_csv}; model skipped")
                continue
            lh = lg_reg * rates[h][0] * rates[a][1] * HOME_ICE * (B2B if h in b2b else 1)
            la = lg_reg * rates[a][0] * rates[h][1] / HOME_ICE * (B2B if a in b2b else 1)
            m = x.type.isin(["ML", "DNB", "1X2", "DC", "TOTAL", "TEAM_TOTAL", "HANDICAP", "BTTS", "BTTS2", "OT"])
            B.loc[x.index[m], "p_model"] = [model_p(r, lh, la) for r in x[m].itertuples()]
            print(f"  {g}: model regulation goals {h} {lh:.2f} - {a} {la:.2f}")
    w = current_weights(con)
    B["w_used"] = [w[family(t, s)] for t, s in zip(B.type, B.scope)]
    lean = (B.p_model - B.p_book).clip(-LEAN_CAP, LEAN_CAP)
    ca, cb = current_calib(con)                              # (0, 1) until the report has fitted one on >= 100 games
    shift = (1 / (1 + np.exp(-(ca + cb * _logit(B.p_book)))) - B.p_book).clip(-0.03, 0.03)     # never more than 3 points
    B["p_cal"] = np.where(B.p_method == "no_vig", B.p_book + shift, B.p_book)
    B["p"] = np.clip(np.where(B.p_model.notna(), B.p_cal + B.w_used * lean, B.p_cal), 0.001, 0.999)
    bad = B[(B.p <= 0) | (B.p >= 1) | (B.odds <= 1) | ((B.p_method == "no_vig") & ~B.overround.between(1.0, 1.25))]
    print(f"price audit: {len(bad)} broken invariants on {len(B):,} rows from {len(files)} boards")
    cols = ["row_key", "game", "home", "away", "market", "type", "scope", "ot", "subject", "selection", "side", "line",
            "odds", "p_book", "p_model", "p", "w_used", "p_method", "overround", "captured", "p_cal"]
    # no delete: a row that is gone from a later capture keeps its last price, so a slip leg can always be graded
    con.executemany(f"INSERT OR REPLACE INTO predictions (night, model_version, {','.join(cols)}) VALUES "
                    f"(?, ?, {','.join('?' * len(cols))})",
                    [(night, MODEL_VERSION, *[None if isinstance(v, float) and math.isnan(v) else v for v in r])
                     for r in B[cols].astype(object).values.tolist()])
    con.commit()
    t = B.groupby("type").agg(rows=("odds", "size"), no_vig=("p_method", lambda s: (s == "no_vig").sum()))
    print(t.sort_values("rows", ascending=False).to_string())
    un = B[B.type == "UNPARSED"].market.value_counts()
    print(f"\nUNPARSED market names: {len(un)} (run `markets` to list them; add a rule in classify() only with a "
          f"settlement rule in settle())")
    return B


# ------------------------------------------------------------------ results
def read_results(path: str) -> tuple[dict, list[str]]:
    """Text format (spec 6):
        GAME BOS@TOR
        P TOR 1 0 2          regulation goals by period; a 4th number = overtime goals (0 or 1)
        P BOS 0 2 1 1
        SO BOS               only when a shootout decided it (winner)
        TOR Auston Matthews: 1 0 5 2 3   skater: goals assists shots-on-goal [blocked-shots hits]
        G TOR Joseph Woll: 27 3          goalie: saves goals-against
        COMPLETE BOS TOR     teams whose skater list is the full box (else a missing player is 'unknown', never 0)
    """
    games, errs, g = {}, [], None
    for n, ln in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        ln = ln.split("#")[0].strip()
        if not ln:
            continue
        try:
            if ln.startswith("GAME "):
                g = ln.split()[1].upper()
                a, h = g.split("@")
                games[g] = dict(home=h, away=a, per={}, so=None, sk={}, gk={}, complete=set())
            elif ln.startswith("P "):
                _, t, *v = ln.split()
                assert t in (games[g]["home"], games[g]["away"]) and len(v) in (3, 4)
                games[g]["per"][t] = [int(x) for x in v] + [0] * (4 - len(v))
            elif ln.startswith("SO "):
                games[g]["so"] = ln.split()[1].upper()
            elif ln.startswith("COMPLETE"):
                games[g]["complete"] |= {x.upper() for x in ln.split()[1:]}
            elif ln.startswith("G "):
                t, rest = ln[2:].split(" ", 1)
                nm, v = rest.rsplit(":", 1)
                games[g]["gk"][nm.strip()] = dict(team=t, saves=int(v.split()[0]), ga=int(v.split()[1]))
            else:
                t, rest = ln.split(" ", 1)
                nm, v = rest.rsplit(":", 1)
                gl, ast, sog, *more = (int(x) for x in v.split())      # optional 4th, 5th: blocked shots, hits
                assert len(more) in (0, 2)
                nm, sk = nm.strip(), games[g]["sk"]
                if nm in sk or f"{nm} #1" in sk:               # two players with one name in a game (Vancouver's two
                    if nm in sk:                               # Elias Petterssons): both kept for the goal count, and
                        sk[f"{nm} #1"] = sk.pop(nm)            # _find() sees two candidates and settles neither
                    nm = f"{nm} #{sum(k.startswith(nm + ' #') for k in sk) + 1}"
                sk[nm] = dict(team=t, goals=gl, ast=ast, pts=gl + ast, sog=sog,
                              blk=more[0] if more else None, hits=more[1] if more else None)
        except Exception as e:
            errs.append(f"line {n}: {ln!r} ({e})")
    for g, d in games.items():
        if set(d["per"]) != {d["home"], d["away"]}:
            errs.append(f"{g}: needs one P line per team"); continue
        h, a = d["per"][d["home"]], d["per"][d["away"]]
        if sum(h[:3]) == sum(a[:3]):
            if h[3] + a[3] == 0 and d["so"] not in (d["home"], d["away"]):
                errs.append(f"{g}: tied after regulation, needs an OT goal or an SO line")
            if h[3] + a[3] > 1 or (h[3] + a[3] == 1 and d["so"]):
                errs.append(f"{g}: overtime ends on the first goal; OT goals {h[3]}+{a[3]} with SO={d['so']}")
        elif h[3] + a[3] or d["so"]:
            errs.append(f"{g}: not tied after regulation but has OT/SO")
        for t in (d["home"], d["away"]):
            s = sum(p["goals"] for p in d["sk"].values() if p["team"] == t)
            if s > sum(d["per"][t]):
                errs.append(f"{g} {t}: skaters' goals {s} > team goals {sum(d['per'][t])}")
            if t in d["complete"] and s != sum(d["per"][t]):
                errs.append(f"{g} {t}: marked COMPLETE but skaters' goals {s} != {sum(d['per'][t])}")
    return games, errs


def _norm(s: str) -> str:
    s = s.lower().translate(str.maketrans("áéíóöőúüűäåčšžřñç", "aeiooouuuaacszrnc"))
    return re.sub(r"[^a-z ]", "", re.sub(r"\b(jr|sr|ii|iii)\b\.?", "", s)).strip()


def _find(pool: dict, name: str):
    """Box scores list 'A. Matthews', boards 'Auston Matthews' (or 'Matthews, Auston'): exact name, else surname +
    first initial. Two candidates (brothers, shared surnames) = no match, never a guess."""
    if "," in name:
        name = " ".join(reversed([x.strip() for x in name.split(",", 1)]))
    n = _norm(name)
    for n in (n, " ".join(n.split()[1:] + n.split()[:1])):       # the board also writes 'Yurov Danila' (surname first)
        hit = [v for k, v in pool.items() if _norm(k) == n]
        if len(hit) != 1 and n:
            hit = [v for k, v in pool.items() if _norm(k).split()[-1:] == n.split()[-1:] and _norm(k)[:1] == n[:1]]
        if len(hit) == 1:
            return hit[0]
        if len(hit) > 1:                                          # two candidates: never guess, and do not try the other order
            return None
    return None


def settle(r, d: dict) -> tuple[str | None, float | None]:
    """-> (won|lost|push|None, actual). None = cannot be decided from the result file (never guessed)."""
    h, a = d["per"][d["home"]], d["per"][d["away"]]
    if r.scope == "G":
        hg, ag = sum(h[:3]), sum(a[:3])
        if r.ot:                                              # OT goal, or one goal to the shootout winner
            hg += h[3] + (d["so"] == d["home"])
            ag += a[3] + (d["so"] == d["away"])
    else:
        i = int(r.scope[1]) - 1
        hg, ag = h[i], a[i]
    t, s, L = r.type, r.side, r.line
    yn = lambda c: "won" if c else "lost"
    if t in ("ML", "1X2"):
        return yn((hg > ag) if s == "home" else (hg < ag) if s == "away" else (hg == ag)), hg - ag
    if t == "DNB":
        return ("push" if hg == ag else yn((hg > ag) == (s == "home"))), hg - ag
    if t == "DC":
        return yn((hg >= ag) if s == "1X" else (hg <= ag) if s == "X2" else (hg != ag)), hg - ag
    if t in ("TOTAL", "TEAM_TOTAL"):
        x = hg + ag if t == "TOTAL" else (hg if r.subject == d["home"] else ag)
        return ("push" if x == L else yn((x > L) == (s == "over"))), x
    if t == "HANDICAP":
        m = (hg - ag) if s == "home" else (ag - hg)
        return ("push" if m + L == 0 else yn(m + L > 0)), m
    if t == "BTTS":
        return yn((hg > 0 and ag > 0) == (s == "yes")), min(hg, ag)
    if t == "BTTS2":
        return yn((hg > 1 and ag > 1) == (s == "yes")), min(hg, ag)
    if t == "OT":
        return yn((sum(h[:3]) == sum(a[:3])) == (s == "yes")), None
    if t == "ODD_EVEN":
        x = hg + ag if not r.subject else (hg if r.subject == d["home"] else ag)
        return yn((x % 2 == 1) == (s == "odd")), x
    if t.startswith("PL_"):
        stat = t[3:].lower()
        pool = d["gk"] if stat == "saves" else d["sk"]
        hit = _find(pool, r.subject or "")
        if hit is None:
            # not in the box: a 0 only if that team's list is complete (he dressed and did nothing is in the box
            # too; absent from a complete box = did not play = void at the book -> no result)
            return None, None
        x = hit[stat]
        if x is None:                                         # the results file does not carry this stat
            return None, None
        return ("push" if x == L else yn((x > L) == (s == "over"))), x
    return None, None


def grade(night: str, results: str, con) -> None:
    games, errs = read_results(results)
    for e in errs:
        print("ERROR", e)
    if errs:
        sys.exit("results file rejected: fix it, nothing was graded")
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ?", con, params=[night])
    if P.empty:
        sys.exit(f"no predictions stored for {night}")
    out = []
    for r in P.itertuples():
        if r.type == "UNPARSED":
            out.append(("unparsed", None, None, night, r.row_key)); continue
        d = games.get(r.game)
        if d is None:
            out.append(("no_result", None, None, night, r.row_key)); continue
        res, act = settle(r, d)
        out.append(("graded" if res else "no_result", res, act, night, r.row_key))
    con.executemany("UPDATE predictions SET status = ?, result = ?, actual = ? WHERE night = ? AND row_key = ?", out)
    con.commit()
    s = pd.DataFrame(out, columns=["status", "result", "actual", "night", "key"])
    print(f"{night}: {len(games)} games in results, {P.game.nunique()} on the boards; "
          f"missing: {sorted(set(P.game) - set(games)) or 'none'}")
    print(s.groupby(["status", "result"], dropna=False).size().to_string())
    G = pd.read_sql("SELECT * FROM predictions WHERE night = ? AND status = 'graded'", con, params=[night])
    bad = G[((G.result == "won") & (G.type.isin(["TOTAL", "TEAM_TOTAL"])) & (G.side == "over") & (G.actual < G.line))]
    print(f"grade audit: {len(bad)} broken invariants")


    grade_slips(night, con)


# ------------------------------------------------------------------ bandit over leg types
def arm_of(t: str, scope: str, p: float) -> str:
    kind = "period" if scope != "G" else {"HANDICAP": "puck_line", "ML": "winner", "1X2": "winner", "TOTAL": "total",
                                          "TEAM_TOTAL": "team_goals", "BTTS": "team_goals", "BTTS2": "team_goals"}.get(t, "other")
    return f"{kind} {'60-75' if p < 0.75 else '75-88'}"


def _legs_only(P: pd.DataFrame) -> pd.DataFrame:
    fam = pd.Series([family(t, s) for t, s in zip(P.type, P.scope)], index=P.index)
    P = P[(P.p_method == "no_vig") & P.p.between(*LEG_P) & (P.odds >= LEG_MIN_ODDS) & ~P.type.isin(["UNPARSED", "DNB"])
          & (fam != "player")].copy()
    P["arm"] = [arm_of(t, s, p) for t, s, p in zip(P.type, P.scope, P.p)]
    return P


def arms(con) -> pd.DataFrame:
    """One arm = a kind of leg x a probability band. The ledger grades EVERY candidate leg on the board, picked or
    not, so every arm is observed every night: there is nothing to explore, and the slips simply follow the
    posterior mean (a Thompson draw would only add noise). Per arm, in flat return per game:
      prior      = what the book's own price implies (minus its margin), worth ARM_K0 games
      posterior  = (ARM_K0 * prior + games * observed) / (ARM_K0 + games), sd = spread / sqrt(ARM_K0 + games)
      shift      = posterior - prior: how much better or worse than its price the arm has paid. `slips` adds it."""
    A = _legs_only(pd.read_sql("SELECT * FROM predictions", con))
    if A.empty:
        return pd.DataFrame(columns=["arm", "rows", "games", "prior", "observed", "posterior", "sd", "shift", "p_best"])
    A["implied"] = A.p_book * A.odds - 1
    T = A.groupby("arm").agg(prior=("implied", "mean")).reset_index()
    D = A[(A.status == "graded") & A.result.isin(["won", "lost"])].copy()
    D["ret"] = (D.result == "won") * D.odds - 1
    g = D.groupby(["arm", D.night + "|" + D.game]).ret.mean().reset_index()
    o = g.groupby("arm").ret.agg(games="size", observed="mean", spread="std").reset_index()
    T = T.merge(o, on="arm", how="left").merge(D.groupby("arm").size().rename("rows").reset_index(), on="arm", how="left")
    T[["games", "rows"]] = T[["games", "rows"]].fillna(0).astype(int)
    sig = T.spread.where(T.games >= 10, 0.5).fillna(0.5).clip(lower=0.3)      # ten straight wins are not a spread of zero
    T["posterior"] = (ARM_K0 * T.prior + T.games * T.observed.fillna(0)) / (ARM_K0 + T.games)
    T["sd"] = sig / np.sqrt(ARM_K0 + T.games)
    T["shift"] = (T.posterior - T.prior).clip(-0.10, 0.10)
    draws = np.random.default_rng(0).normal(T.posterior.values, T.sd.values, size=(20000, len(T)))
    T["p_best"] = np.bincount(draws.argmax(axis=1), minlength=len(T)) / 20000
    return T[["arm", "rows", "games", "prior", "observed", "posterior", "sd", "shift", "p_best"]].sort_values("posterior", ascending=False)


def stat_arms(con) -> pd.DataFrame:
    """The team-stat comparison as bandit arms: graded game lines the book priced 40-60%, split by whether the stats
    sheet (teamstats.py, games before the night only) pointed the same way. Same prior as the other arms."""
    D = pd.read_sql("SELECT * FROM lean_legs WHERE p BETWEEN 0.40 AND 0.60", con)
    if D.empty:
        return pd.DataFrame()
    D["arm"] = D.kind + np.where(D.agree == 1, " | stats agree", " | stats disagree")
    D["ret"] = D.y * D.odds - 1
    D["implied"] = D.p * D.odds - 1
    rows = []
    for arm, x in D.groupby("arm"):
        g = x.groupby(x.night + x.game).ret.mean()
        prior, n = x.implied.mean(), len(g)
        sig = max(float(g.std(ddof=1)) if n >= 10 else 0.5, 0.3)
        post = (ARM_K0 * prior + n * g.mean()) / (ARM_K0 + n)
        rows.append(dict(arm=arm, bets=len(x), games=n, prior=prior, observed=g.mean(), posterior=post,
                         sd=sig / math.sqrt(ARM_K0 + n), shift=float(np.clip(post - prior, -0.1, 0.1))))
    return pd.DataFrame(rows).sort_values("posterior", ascending=False)


# ------------------------------------------------------------------ slips: built by rule, stored, graded
def leg_pool(night: str, con, players: bool = False, exclude=()) -> pd.DataFrame:
    """Legs a slip may use: two-way priced (margin removed inside the pair), win chance 60-88%, price >= 1.15,
    one line per quantity, at most 6 per game. Ranked by p * odds - 1 = how little margin the leg costs.
    Player lines are out unless asked for: a late scratch or a goalie change hits them hardest."""
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ? AND status = 'open'", con, params=[night])
    if P.empty:
        sys.exit(f"no open predictions for {night}: run price first")
    P = P[P.captured == P.groupby("game").captured.transform("max")]       # only prices from the latest capture of each game
    P = _legs_only(P) if not players else P[(P.p_method == "no_vig") & P.p.between(*LEG_P) & (P.odds >= LEG_MIN_ODDS)
                                            & ~P.type.isin(["UNPARSED", "DNB"])].assign(arm=lambda z: [arm_of(t, s_, p) for t, s_, p in zip(z.type, z.scope, z.p)])
    text = P.game + " " + P.market.fillna("") + " " + P.selection.fillna("") + " " + P.subject.fillna("")
    for n in exclude:
        P = P[~text.str.contains(n, case=False, regex=False)]
    A = arms(con).set_index("arm")
    shift = P.arm.map(A["shift"]).fillna(0.0) if len(A) else 0.0
    P = P.copy()
    P["p"] = (P.p + shift / P.odds).clip(0.50, 0.95)          # the arm's record moves the leg's chance (0 until games are graded)
    P["edge"] = P.p * P.odds - 1
    P["subject"] = P.subject.fillna("")
    P = P.sort_values("edge", ascending=False).groupby(["game", "type", "scope", "subject"]).head(1)
    return P.sort_values("edge", ascending=False).groupby("game").head(6).reset_index(drop=True)


def build_slip(d: pd.DataFrame, target: float, tol: float = 0.20, width: int = 400, max_legs: int = 12):
    """One leg per game (legs from one game move together, so their chances cannot be multiplied). Among slips
    paying target .. target * 1.2, the one with the highest p * odds, i.e. the least margin paid in total."""
    if d.empty:
        return None
    lo, hi = math.log(target) - 1e-9, math.log(target * (1 + tol))
    d = d.reset_index(drop=True)
    lo_o, lp, gm = np.log(d.odds.values), np.log(d.p.values), d.game.values
    beam, best = [((), 0.0, 0.0)], None
    for _ in range(max_legs):
        nxt = []
        for legs, o, p in beam:
            gs = {gm[i] for i in legs}
            for j in range(legs[-1] + 1 if legs else 0, len(d)):
                if gm[j] in gs or o + lo_o[j] > hi:
                    continue
                c = (legs + (j,), o + lo_o[j], p + lp[j])
                if c[1] >= lo:
                    best = c if best is None or c[1] + c[2] > best[1] + best[2] else best
                else:
                    nxt.append(c)
        if not nxt:
            break
        a_ = sorted(nxt, key=lambda c: c[1] + c[2], reverse=True)[:width // 2]
        b_ = sorted(nxt, key=lambda c: c[2], reverse=True)[:width // 2]
        beam = list({c[0]: c for c in a_ + b_}.values())
    return None if best is None else d.loc[list(best[0])]


def slips(night: str, targets, con, players: bool = False, exclude=(), replace: bool = False) -> None:
    pool = leg_pool(night, con, players, exclude)
    done = con.execute("SELECT count(DISTINCT night || game) FROM predictions WHERE status = 'graded'").fetchone()[0]
    print(f"{night}: {len(pool)} candidate legs in {pool.game.nunique() if len(pool) else 0} games "
          f"(two-way priced, {LEG_P[0]:.0%}-{LEG_P[1]:.0%}, price >= {LEG_MIN_ODDS})")
    A = arms(con)
    if len(A):
        print("bandit arms (flat return per game; shift = posterior - prior is added to each leg of that arm):")
        print(A.round(3).to_string(index=False))
    print(f"ledger: {done} graded games. PAPER ONLY: the report decides at {DECIDE_GAMES} games whether legs like "
          f"these lose less than the margin says. Until then every slip below loses on average.")
    redo = {f"{t:g}x" for t in targets} if replace else set()      # legs of every other slip of the night stay taken
    used = {k for k, sid_ in con.execute("SELECT row_key, slip_id FROM slip_legs WHERE night = ?", [night]) if sid_ not in redo}
    for t in targets:
        sid = f"{t:g}x"
        old = con.execute("SELECT odds, p, status FROM slips WHERE night = ? AND slip_id = ?", [night, sid]).fetchone()
        if old and not replace:
            print(f"\n  {sid}: already built ({old[0]:.2f}, P {old[1]:.1%}, {old[2]}); --replace builds it again")
            used |= {r[0] for r in con.execute("SELECT row_key FROM slip_legs WHERE night = ? AND slip_id = ?", [night, sid])}
            continue
        legs, note = build_slip(pool[~pool.row_key.isin(used)], t), ""
        if legs is None:
            legs, note = build_slip(pool, t), " (shares legs with another slip tonight: they win and lose together)"
        if legs is None:
            need = math.ceil(math.log(t) / math.log(1.45))
            print(f"\n  {sid}: not reachable with one leg per game (about {need}+ games with a 60-88% leg needed)")
            continue
        used |= set(legs.row_key)
        o, p = float(np.prod(legs.odds)), float(np.prod(legs.p))
        now = pd.Timestamp.now("UTC").isoformat()
        con.execute("DELETE FROM slips WHERE night = ? AND slip_id = ?", [night, sid])
        con.execute("DELETE FROM slip_legs WHERE night = ? AND slip_id = ?", [night, sid])
        con.execute("INSERT INTO slips (night, slip_id, target, odds, p, ev, legs, built) VALUES (?,?,?,?,?,?,?,?)",
                    [night, sid, t, round(o, 4), p, p * o - 1, len(legs), now])
        con.executemany("INSERT INTO slip_legs VALUES (?,?,?,?,?,?,?,?,?,?)",
                        [(night, sid, r.row_key, r.game, r.market, r.selection, r.odds, r.p, r.p_book, r.captured)
                         for r in legs.itertuples()])
        print(f"\n  {sid} -> pays {o:.2f} | {len(legs)} legs | wins {p:.1%} (1 in {1 / p:.1f}) | needs {1 / o:.1%} to "
              f"break even | expected return {p * o - 1:+.1%}{note}")
        for r in legs.itertuples():
            sel = r.selection if not r.subject or r.subject in r.selection else f"{r.subject} {r.selection}"
            print(f"     {r.game:<9} {r.market[:44]:<44} {sel[:30]:<30} @ {r.odds:.2f}  P {r.p:.0%}")
    con.commit()
    print("\nBefore placing anything: confirm both starting goalies in every leg's game and rerun with --exclude "
          "\"AWY@HOM\" --replace for games where one is unknown.")


STRATEGIES = {
    "chance": "highest win chance by the book's price (legs 55-92%, any number of legs)",
    "bandit": "least margin paid, legs 60-88%, chances moved by the bandit's record of each kind of leg",
    "model":  "highest win chance among legs where the goals model rates the leg at least as likely as the book",
}


def _search(d: pd.DataFrame, target: float, score: str, tol: float = 0.20, width: int = 600, max_legs: int = 10):
    """One leg per game; slip pays target .. target * (1 + tol); best by sum of log(score column) (+ log odds for
    the margin objective). Beam search over the legs sorted by game."""
    d = d.reset_index(drop=True)
    lo, hi = math.log(target) - 1e-9, math.log(target * (1 + tol))
    lo_o, gm = np.log(d.odds.values), d.game.values
    sc = np.log(d.p.values) + lo_o if score == "p_margin" else np.log(d[score].values)
    beam, best = [((), 0.0, 0.0)], None
    for _ in range(max_legs):
        nxt = []
        for legs, o, v in beam:
            gs = {gm[i] for i in legs}
            for j in range(legs[-1] + 1 if legs else 0, len(d)):
                if gm[j] in gs or o + lo_o[j] > hi:
                    continue
                c = (legs + (j,), o + lo_o[j], v + sc[j])
                if c[1] >= lo:
                    best = c if best is None or c[2] > best[2] else best
                else:
                    nxt.append(c)
        if not nxt:
            break
        a_ = sorted(nxt, key=lambda c: c[2] + c[1], reverse=True)[:width // 2]
        b_ = sorted(nxt, key=lambda c: c[2], reverse=True)[:width // 2]
        beam = list({c[0]: c for c in a_ + b_}.values())
    return None if best is None else d.loc[list(best[0])]


def strategy_slips(night: str, targets, con, exclude=(), only=None) -> None:
    """The same targets built three ways, every slip stored and graded under '<target>x-<strategy>', so the report
    can say which way of building comes closest to what it promised. Within one strategy no leg is used twice;
    across strategies the same leg may appear (they are separate tests, and win or lose together on that leg)."""
    P = pd.read_sql("SELECT * FROM predictions WHERE night = ? AND status = 'open'", con, params=[night])
    P = P[P.captured == P.groupby("game").captured.transform("max")]
    fam = pd.Series([family(t, s_) for t, s_ in zip(P.type, P.scope)], index=P.index)
    P = P[(P.p_method == "no_vig") & ~P.type.isin(["UNPARSED", "DNB"]) & (fam != "player") & (P.odds >= 1.10)].copy()
    text = P.game + " " + P.market.fillna("") + " " + P.selection.fillna("") + " " + P.subject.fillna("")
    for n in exclude:
        P = P[~text.str.contains(n, case=False, regex=False)]
    P["subject"] = P.subject.fillna("")
    one = lambda x, col: x.sort_values(col, ascending=False).groupby(["game", "type", "scope", "subject", "side"]).head(1)
    pools = {"chance": one(P[P.p_book.between(0.55, 0.92)], "p_book"),
             "bandit": leg_pool(night, con, False, exclude),
             "model": one(P[P.p_model.notna() & (P.p_model >= P.p_book) & P.p_book.between(0.50, 0.92)], "p_model")}
    score = {"chance": "p_book", "bandit": "p_margin", "model": "p_model"}
    now, rows = pd.Timestamp.now("UTC").isoformat(), []
    # one leg in one slip of the night, across strategies too (2026-11-22: Tampa Bay to win was in three slips and
    # sank all three). Slips of other strategies already stored tonight keep their legs reserved.
    taken = {k for k, sid_ in con.execute("SELECT row_key, slip_id FROM slip_legs WHERE night = ?", [night])
             if "-" in sid_ and sid_.split("-", 1)[1] not in (only or pools)}
    for strat, pool in pools.items():
        if only and strat not in only:
            continue
        used = set()
        for t in targets:
            sid = f"{t:g}x-{strat}"
            legs, note = _search(pool[~pool.row_key.isin(used | taken)], t, score[strat]), ""
            if legs is None:
                legs, note = _search(pool[~pool.row_key.isin(used)], t, score[strat]), "  (shares a leg with another slip: too few games)"
            if legs is None:
                print(f"  {sid}: not reachable"); continue
            used |= set(legs.row_key)
            taken |= set(legs.row_key)
            o, p, pb = float(np.prod(legs.odds)), float(np.prod(legs.p)), float(np.prod(legs.p_book))
            pm = float(np.prod(legs.p_model.fillna(legs.p_book)))
            con.execute("DELETE FROM slips WHERE night = ? AND slip_id = ?", [night, sid])
            con.execute("DELETE FROM slip_legs WHERE night = ? AND slip_id = ?", [night, sid])
            con.execute("INSERT INTO slips (night, slip_id, target, odds, p, ev, legs, built) VALUES (?,?,?,?,?,?,?,?)",
                        [night, sid, t, round(o, 4), p, p * o - 1, len(legs), now])
            con.executemany("INSERT INTO slip_legs VALUES (?,?,?,?,?,?,?,?,?,?)",
                            [(night, sid, r.row_key, r.game, r.market, r.selection, r.odds, r.p, r.p_book, r.captured) for r in legs.itertuples()])
            rows.append(dict(slip=sid, legs=len(legs), pays=o, book=pb, model=pm, ev=p * o - 1))
            print(f"\n  {sid}: pays {o:.2f} | {len(legs)} legs | book {pb:.1%} | model {pm:.1%} | expected return {p * o - 1:+.1%}{note}")
            for r in legs.itertuples():
                sel = r.selection if not r.subject or r.subject in r.selection else f"{r.subject} {r.selection}"
                print(f"     {r.game:<8} {r.market[:44]:<44} {sel[:30]:<30} @ {r.odds:.2f}  book {r.p_book:.0%}  model "
                      + (f"{r.p_model:.0%}" if r.p_model == r.p_model else "--"))
    con.commit()
    if rows:
        print("\n" + pd.DataFrame(rows).sort_values("book", ascending=False).round(3).to_string(index=False))


def grade_slips(night: str, con) -> None:
    """All legs won = won; one lost = lost. A void leg (push, scratched player) drops out at price 1.00, as the book
    does. A slip stays open while any leg's game has no result."""
    S = pd.read_sql("SELECT * FROM slips WHERE night = ?", con, params=[night])
    if S.empty:
        return
    P = pd.read_sql("SELECT row_key, game, status, result FROM predictions WHERE night = ?", con, params=[night])
    have = set(P[P.status == "graded"].game)
    P = P.set_index("row_key")
    for s in S.itertuples():
        L = pd.read_sql("SELECT * FROM slip_legs WHERE night = ? AND slip_id = ?", con, params=[night, s.slip_id])
        if not set(L.game) <= have:
            print(f"  slip {s.slip_id}: waiting for {sorted(set(L.game) - have)}")
            continue
        res = [P.result.get(k) for k in L.row_key]
        if "lost" in res:
            out, paid = "lost", 0.0
        elif "won" not in res:
            out, paid = "void", 1.0
        else:
            out, paid = "won", float(np.prod([o for o, r in zip(L.odds, res) if r == "won"]))
        con.execute("UPDATE slips SET status = 'graded', result = ?, paid = ? WHERE night = ? AND slip_id = ?",
                    [out, paid, night, s.slip_id])
        print(f"  slip {s.slip_id}: {out} (stake 1 -> {paid:.2f}; said P {s.p:.1%}; legs {res})")
    con.commit()


# ------------------------------------------------------------------ report
def decided(con) -> pd.DataFrame:
    d = pd.read_sql("SELECT * FROM predictions WHERE status = 'graded' AND result IN ('won', 'lost')", con)
    d["y"] = (d.result == "won").astype(float)
    d["ret"] = d.y * d.odds - 1
    d["cluster"] = d.night + "|" + d.game
    d["family"] = [family(t, s) for t, s in zip(d.type, d.scope)]
    return d


def cstat(x: pd.DataFrame) -> dict:
    g = x.groupby("cluster").ret.mean()
    n = len(g)
    return dict(rows=len(x), games=n, hit=x.y.mean(), need=(1 / x.odds).mean(), ret=g.mean(),
                se=g.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan"))


def cbrier(p, y, cl) -> float:
    return float(((p - y) ** 2).groupby(cl).mean().mean()) if len(y) else float("nan")


WS, GAMES_PRIOR, FOLDS = np.linspace(0, 1, 41), 40, 7


def _best_w(pb, pm, y) -> float:
    return float(WS[int(np.argmin([((pb + w * (pm - pb) - y) ** 2).mean() for w in WS]))])


def evidence(x: pd.DataFrame) -> dict:
    """The NFL system's rule, paid for there with a weight that swung 0.93 -> 0.004 in a week: the weight is
    fitted on other games and scored on held-out games (folds of whole games); t = standard errors across games
    by which the held-out blend beats the book. w = w_fit x games/(games+40) x max(0, 1 - 4/t^2): nothing below
    t = 2."""
    G = x.cluster.nunique()
    pb, pm, y = x.p_book, x.p_model, x.y
    fold = x.cluster.map(lambda c: int(hashlib.md5(str(c).encode()).hexdigest(), 16) % FOLDS)
    bl = pd.Series(np.nan, index=x.index)
    for k in sorted(fold.unique()):
        tr, te = fold != k, fold == k
        w = _best_w(pb[tr].values, pm[tr].values, y[tr].values) if tr.sum() >= 20 else 0.0
        bl[te] = pb[te] + w * (pm[te] - pb[te])
    d = ((bl - y) ** 2 - (pb - y) ** 2).groupby(x.cluster).sum()
    sd = float(d.std(ddof=1)) if G > 1 else 0.0
    t = float(-d.sum() / (sd * math.sqrt(G))) if sd > 0 else 0.0
    w_fit = _best_w(pb.values, pm.values, y.values)
    return dict(games=G, w_fit=w_fit, t=t, w_used=round(w_fit * G / (G + GAMES_PRIOR) * (max(0.0, 1 - 4 / t ** 2) if t > 2 else 0.0), 3))


def refit(d: pd.DataFrame, con) -> None:
    rows, now = [], pd.Timestamp.now('UTC').isoformat()
    for fam, x in d.dropna(subset=["p_model"]).groupby("family"):
        nc = x.cluster.nunique()
        if nc < MIN_GAMES:
            print(f"  {fam}: {nc} games with a model price (< {MIN_GAMES}): weight stays {START_W[fam]}")
            continue
        e = evidence(x)
        rows.append((now, fam, nc, e["w_fit"], e["t"], e["w_used"]))        # the w_se column now holds t
        print(f"  {fam}: {nc} games, in-sample weight {e['w_fit']:.2f}, held-out t {e['t']:+.2f} -> used {e['w_used']:.3f}"
              + ("" if e["t"] > 2 else "  (model has not beaten the book on held-out games: weight 0)"))
    if rows:
        con.executemany("INSERT INTO weights VALUES (?,?,?,?,?,?)", rows)
        con.commit()


def report(con, test_night: str | None) -> None:
    d = decided(con)
    if d.empty:
        sys.exit("nothing graded yet")
    nights = sorted(d.night.unique())
    L = [f"# NHL ledger report\n\n{len(d):,} decided bets, {d.cluster.nunique()} games, nights {nights[0]} .. "
         f"{nights[-1]}. Games are the unit of evidence: bets on one game move together.\n"]
    st = pd.read_sql("SELECT status, count(*) n FROM predictions GROUP BY 1", con)
    L += ["## Ledger status\n", st.to_markdown(index=False), ""]

    L.append("## Negatives first: book vs model vs blend (Brier, lower is better; only rows with a model price)\n")
    m = d.dropna(subset=["p_model"])
    rows = [dict(type=t, n=len(x), games=x.cluster.nunique(), book=cbrier(x.p_book, x.y, x.cluster),
                 model=cbrier(x.p_model, x.y, x.cluster), blend=cbrier(x.p, x.y, x.cluster)) for t, x in m.groupby("type")]
    if rows:
        T = pd.DataFrame(rows).assign(model_minus_book=lambda z: z.model - z.book).sort_values("model_minus_book", ascending=False)
        L += [T.round(4).to_markdown(index=False), ""]
        dis = m[(m.p_model >= 0.5) != (m.p_book >= 0.5)]
        if len(dis):
            L.append(f"When model and book pick opposite sides: model right {(dis.y == (dis.p_model >= .5)).mean():.1%} "
                     f"of {len(dis)} rows in {dis.cluster.nunique()} games.\n")
    else:
        L.append("No model prices graded yet (price with --teams).\n")

    L.append("## Is the book's price right? Calibration of p_book (two-way priced rows)\n")
    nv = d[d.p_method == "no_vig"]
    b = pd.cut(nv.p_book, np.arange(0, 1.01, 0.1), include_lowest=True)
    C = nv.groupby(b, observed=True).agg(n=("y", "size"), games=("cluster", "nunique"), p_book=("p_book", "mean"), hit=("y", "mean"))
    C["gap"] = C.hit - C.p_book
    L += [C.round(3).to_markdown(), ""]

    L.append("## Return by odds band (flat stake, averaged per game)\n")
    d["band"] = pd.cut(d.odds, [1, 1.25, 1.5, 1.8, 2.2, 3, 5, 10, 1e9]).astype(str)
    L += [pd.DataFrame([dict(odds=k, **cstat(x)) for k, x in d.groupby("band", sort=False)]).round(3).to_markdown(index=False), ""]

    L.append("## Return by bet type and side\n")
    R = pd.DataFrame([dict(type=t, scope=sc, side=s, **cstat(x)) for (t, sc, s), x in d.groupby(["type", "scope", "side"])])
    L += [R.sort_values("ret").round(3).to_markdown(index=False), ""]

    # default split: the last 30% of nights are the test nights (one night alone tests nothing)
    test_night = test_night or (nights[int(len(nights) * 0.7)] if len(nights) >= 10 else nights[-1])
    L.append(f"## Pattern scan: found on nights before {test_night}, tested on {test_night} and later\n")
    a_, b_ = d[d.night < test_night], d[d.night >= test_night]
    S = []
    for dims in (["type", "scope", "side"], ["family", "side"], ["family", "band"], ["type", "band"]):
        for key, x in a_.groupby(dims):
            y = b_
            for c, v in zip(dims, key):
                y = y[y[c] == v]
            if x.cluster.nunique() < 15 or y.cluster.nunique() < 5 or len(x) < 60:
                continue
            sa, sb = cstat(x), cstat(y)
            S.append(dict(group=" / ".join(map(str, key)), rows_a=sa["rows"], games_a=sa["games"], ret_a=sa["ret"],
                          z_a=sa["ret"] / sa["se"] if sa["se"] else np.nan, games_b=sb["games"], ret_b=sb["ret"]))
    if S:
        S = pd.DataFrame(S)
        held = S[(S.z_a >= 2) & (S.ret_b > 0)]
        L.append(f"{len(S)} groups tested; chance alone gives about {len(S) * 0.023:.1f} at z >= 2. "
                 f"At z >= 2 and still positive on the test nights: {len(held)}.\n")
        L += [S.sort_values("z_a", ascending=False).head(15).round(3).to_markdown(index=False), ""]
    else:
        L.append("Not enough games yet (needs 15 games before and 5 from the test night on, per group).\n")

    L.append("## Gates (spec 7): a type is playable only when ALL hold\n")
    gate = []
    for t, x in d.groupby("type"):
        s = cstat(x)
        mm = x.dropna(subset=["p_model"])
        gate.append(dict(type=t, n=len(x), games=s["games"], n_ok=len(x) >= GATE_N and s["games"] >= 100,
                         ret=s["ret"], ret_minus_2se=s["ret"] - 2 * s["se"],
                         blend_le_book=bool(len(mm)) and cbrier(mm.p, mm.y, mm.cluster) <= cbrier(mm.p_book, mm.y, mm.cluster)))
    Gt = pd.DataFrame(gate)
    Gt["PLAYABLE"] = Gt.n_ok & (Gt.ret_minus_2se > 0)
    L += [Gt.round(3).to_markdown(index=False), "",
          f"Playable types: {', '.join(Gt[Gt.PLAYABLE].type) or 'NONE'}. CLV is not measured here (needs a second "
          f"capture near puck drop); until it is, PLAYABLE is necessary, not sufficient.\n"]
    # ---- can 3-7x slips be built from this book? Slips are products of legs, so the legs decide it.
    games = d.cluster.nunique()
    nv = d[d.p_method == "no_vig"].copy()
    nv["pband"] = pd.cut(nv.p_book, [LEG_P[0], 0.70, 0.80, LEG_P[1]]).astype(str)
    legs = nv[nv.p_book.between(*LEG_P) & (nv.odds >= LEG_MIN_ODDS) & (nv.family != "player")]
    L.append(f"## Slip legs: two-way rows the book rates {LEG_P[0]:.0%}-{LEG_P[1]:.0%} (what `slips` builds from)\n")
    if len(legs):
        rows = [dict(family="ALL", band=f"{LEG_P[0]}-{LEG_P[1]}", **cstat(legs))]
        rows += [dict(family=f, band=b, **cstat(x)) for (f, b), x in legs.groupby(["family", "pband"])]
        T = pd.DataFrame(rows)
        T["ret_minus_2se"], T["ret_plus_2se"] = T.ret - 2 * T.se, T.ret + 2 * T.se
        L += [T.round(3).to_markdown(index=False), ""]
        s = cstat(legs)
        lo_, hi_ = s["ret"] - 2 * s["se"], s["ret"] + 2 * s["se"]
        four = lambda r: (1 + r) ** 4 - 1                     # a 3-7x slip is about 3-6 such legs
        if games < CAL_GAMES:
            v = f"TOO EARLY ({games} of {CAL_GAMES} games for a first read). Nothing in this report is evidence yet."
        elif games < DECIDE_GAMES:
            v = (f"FIRST READ ONLY ({games} of {DECIDE_GAMES} games). Per-leg return {s['ret']:+.1%}, plausible range "
                 f"{lo_:+.1%} .. {hi_:+.1%}. Paper slips only.")
        elif lo_ > 0:
            v = (f"LEGS PROFITABLE beyond noise ({s['ret']:+.1%}, range {lo_:+.1%} .. {hi_:+.1%}, {games} games): a "
                 f"4-leg slip is worth about {four(s['ret']):+.1%}. Check the slip ledger and price movement below "
                 f"agree before staking.")
        elif hi_ < 0:
            v = (f"LEGS LOSE beyond noise ({s['ret']:+.1%}, range {lo_:+.1%} .. {hi_:+.1%}, {games} games): a 4-leg "
                 f"slip loses about {-four(s['ret']):.1%} of the stake each time. No 3-7x slip from this book is a "
                 f"good bet. Stop, or keep it on paper.")
        else:
            v = (f"NOT SEPARABLE FROM ZERO ({s['ret']:+.1%}, range {lo_:+.1%} .. {hi_:+.1%}, {games} games): no "
                 f"evidence of an edge. A 4-leg slip is worth between {four(lo_):+.1%} and {four(hi_):+.1%}.")
    else:
        v = f"TOO EARLY ({games} of {CAL_GAMES} games): no graded leg in the band yet."
    L += [f"**Verdict: {v}**\n"]
    L.insert(1, f"**Verdict on 3-7x slips: {v}**\n")

    SA = stat_arms(con)
    if len(SA):
        L += ["## Team stats against the book (legs priced 40-60%)\n", SA.round(3).to_markdown(index=False), "",
              "stats agree = the team stat sheet (teamstats.py) pointed the same way as the bet. A positive posterior "
              "for 'agree' would mean the book under-prices what the stats show; for 'disagree', that it over-prices it.\n"]
    L += ["## Bandit arms: which kind of leg has paid better or worse than its price\n", arms(con).round(3).to_markdown(index=False), "",
          f"prior = the book's own price (its margin), worth {ARM_K0} games. `slips` adds `shift` to every leg of the arm. "
          "p_best = chance the arm is truly the best one, given the games so far.\n"]
    L.append("## Slip ledger: every slip that was built, said vs happened\n")
    S = pd.read_sql("SELECT * FROM slips WHERE status = 'graded' AND result IN ('won', 'lost')", con)
    if len(S):
        rows = []
        for t, x in [("ALL", S)] + list(S.groupby("target")):
            n = len(x)
            rows.append(dict(target=t, slips=n, nights=x.night.nunique(), avg_odds=x.odds.mean(), said_wins=x.p.sum(),
                             wins=int((x.result == "won").sum()), ret=x.paid.mean() - 1,
                             se=x.paid.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")))
        L += [pd.DataFrame(rows).round(3).to_markdown(index=False), "",
              "A 5x slip wins about 1 time in 6: fewer than ~100 slips per target says nothing about return; "
              "`said_wins` against `wins` is the part that can be read earlier.\n"]
        S["strategy"] = S.slip_id.str.split("-", n=1).str[1].fillna("rule")
        rows = [dict(strategy=k, slips=len(x), said_wins=x.p.sum(), wins=int((x.result == "won").sum()),
                     legs_said=float((x.p ** (1 / x.legs)).mean()), ret=x.paid.mean() - 1) for k, x in S.groupby("strategy")]
        L += ["By strategy (built the same night, same targets; legs_said = average chance per leg):\n",
              pd.DataFrame(rows).round(3).to_markdown(index=False), ""]
        lg = pd.read_sql("SELECT l.slip_id, p.result, l.p_book FROM slip_legs l JOIN predictions p ON p.night = l.night AND "
                         "p.row_key = l.row_key WHERE p.result IN ('won', 'lost')", con)
        if len(lg):
            lg["strategy"] = lg.slip_id.str.split("-", n=1).str[1].fillna("rule")
            L += ["Legs by strategy (each leg graded on its own):\n",
                  lg.groupby("strategy").agg(legs=("result", "size"), said=("p_book", "mean"),
                                             won=("result", lambda r: (r == "won").mean())).round(3).to_markdown(), ""]
        mv = pd.read_sql("SELECT l.night, l.game, p.p_book - l.p_book AS move FROM slip_legs l JOIN predictions p "
                         "ON p.night = l.night AND p.row_key = l.row_key WHERE p.captured > l.captured", con)
        if len(mv):
            g = mv.groupby(mv.night + mv.game).move.mean()
            L.append(f"Price movement after the slip was built (later capture minus the price taken, in probability; "
                     f"positive = the market moved toward the pick): {g.mean():+.4f} +- "
                     f"{g.std(ddof=1) / math.sqrt(len(g)) if len(g) > 1 else float('nan'):.4f} over {len(g)} games.\n")
        else:
            L.append("Price movement: no leg has a later capture yet (price a second capture near puck drop under the "
                     "same date, after building the slips).\n")
    else:
        L.append("No graded slips yet.\n")

    L.append("## Calibration fit of the book price (used by `price` from the next night on)\n")
    tr, te = nv[nv.night < test_night], nv[nv.night >= test_night]
    if tr.cluster.nunique() >= CAL_GAMES and te.cluster.nunique() >= 30:
        a1, b1 = fit_calib(tr)
        pc = 1 / (1 + np.exp(-(a1 + b1 * _logit(te.p_book))))
        bb, bc = cbrier(te.p_book, te.y, te.cluster), cbrier(pc, te.y, te.cluster)
        keep = bc < bb                                        # stored only if it beat the plain price on later games
        con.execute("INSERT INTO calib VALUES (?,?,?,?)", [pd.Timestamp.now("UTC").isoformat(), games,
                                                           a1 if keep else 0.0, b1 if keep else 1.0])
        con.commit()
        L.append(f"Fitted on {tr.cluster.nunique()} games before {test_night} (a {a1:+.3f}, b {b1:.3f}; b > 1 = favourites "
                 f"win more often than the evenly de-margined price says), scored on {te.cluster.nunique()} later games: "
                 f"Brier book {bb:.4f}, calibrated {bc:.4f}. {'STORED' if keep else 'NOT better on later games: plain price kept'}.\n")
    else:
        L.append(f"Not fitted: needs {CAL_GAMES} games before the test night and 30 from it on "
                 f"(have {tr.cluster.nunique()} / {te.cluster.nunique()}). `price` uses the plain de-margined price.\n")

    out = Path(DB).with_name("nhl_ledger_report.md")
    out.write_text("\n".join(L), encoding="utf-8")
    print(f"Verdict on 3-7x slips: {v}")
    print(f"weights refit ({d.cluster.nunique()} games):")
    refit(d, con)
    print(out)


# ------------------------------------------------------------------ NHL public API
def _get(path: str):
    import urllib.request
    req = urllib.request.Request(f"{API}/{path}", headers={"User-Agent": "Mozilla/5.0"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())


def fetch_teams(date: str, out: str) -> None:
    """teams.csv (team,gp,gf,ga) from the standings as of `date`, and the teams playing `date` that also played the
    day before. Standings goals count one goal per shootout win: taken out again (shootoutWins / shootoutLosses).
    Field names read from live responses on 2026-11-16; the function was not run."""
    st = _get(f"standings/{date}")["standings"]
    ab = lambda t: t["teamAbbrev"]["default"] if isinstance(t["teamAbbrev"], dict) else t["teamAbbrev"]
    rows = [(ab(t), t["gamesPlayed"], t["goalFor"] - t.get("shootoutWins", 0), t["goalAgainst"] - t.get("shootoutLosses", 0))
            for t in st]
    Path(out).write_text("team,gp,gf,ga\n" + "\n".join(",".join(map(str, r)) for r in rows) + "\n", encoding="utf-8")
    side = lambda day: {g[k]["abbrev"] for g in _get(f"score/{day}").get("games", []) for k in ("homeTeam", "awayTeam")}
    b2b = sorted(side(date) & side((pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    print(f"{out}: {len(rows)} teams\nback-to-back on {date}: --b2b {','.join(b2b) or '(none)'}")


def _ingest(folder: str, table: str) -> pd.DataFrame:
    f = sorted(Path(folder).glob(f"nhl_*_{table}.csv"))
    if not f:
        sys.exit(f"no nhl_<season>_{table}.csv in {folder}: run nhl_ingest.py first (NHL_OUT={folder})")
    return pd.read_csv(f[-1], low_memory=False, dtype={"period": str})


def ingest_results(date: str, folder: str, out: str) -> None:
    """Results text for one date from nhl_ingest.py's tables. Better than `results` from the API: full player
    names, ice time (a skater with none is left out = void), and the tables were cross-checked by the ingest."""
    sch = _ingest(folder, "schedule")
    sch = sch[(sch.game_date.astype(str).str[:10] == date) & sch.ended_in.notna()]
    per, sk, gk = _ingest(folder, "period_scores"), _ingest(folder, "player_game_stats"), _ingest(folder, "goalie_game_stats")
    L = [f"# NHL results {date} from nhl_ingest tables in {folder}", "# skater: goals assists shots blocked-shots hits; goalie: saves goals-against"]
    for g in sch.itertuples():
        if "score_checks" in sch and not bool(g.score_checks):
            print(f"  {g.away_team}@{g.home_team}: period goals do not add up to the final in the ingest; left out")
            continue
        p = per[per.game_id == g.game_id]
        line = lambda t: [int(p[(p.team == t) & (p.period == k)].goals.sum()) for k in ("1", "2", "3")] + \
            [int(p[(p.team == t) & p.period.str.startswith("OT")].goals.sum())]
        L += ["", f"GAME {g.away_team}@{g.home_team}"] + [f"P {t} " + " ".join(map(str, line(t))) for t in (g.home_team, g.away_team)]
        if g.ended_in == "SO":
            L.append(f"SO {g.winner}")
        nm = lambda r: r.player_name if isinstance(r.player_name, str) and r.player_name else r.player_short
        for r in sk[(sk.game_id == g.game_id) & sk.played.astype(bool)].itertuples():
            L.append(f"{r.team} {nm(r)}: {int(r.goals)} {int(r.assists)} {int(r.shots_on_goal)} {int(r.blocked_shots)} {int(r.hits)}")
        for r in gk[(gk.game_id == g.game_id) & gk.played.astype(bool)].itertuples():
            L.append(f"G {r.team} {nm(r)}: {int(r.saves)} {int(r.goals_against)}")   # GA never has the shootout goal
        full = [t for t in (g.home_team, g.away_team)           # a goalie can score (Shesterkin, 2026-10-01): then the
                if int(sk[(sk.game_id == g.game_id) & (sk.team == t)].goals.sum()) == sum(line(t))]     # skater list is not all goals
        if full:
            L.append("COMPLETE " + " ".join(full))
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    games, errs = read_results(out)
    print(f"{out}: {len(games)} games, {len(errs)} errors")
    for e in errs:
        print("ERROR", e)


def ingest_teams(date: str, folder: str, out: str) -> None:
    """teams.csv from REGULATION scores of every game before `date` (no overtime or shootout goals at all), the
    back-to-back list, and this season's share of games tied after 60 minutes (compare with TIE_K)."""
    sch = _ingest(folder, "schedule")
    sch["d"] = sch.game_date.astype(str).str[:10]
    fin = sch[sch.ended_in.notna() & (sch.d < date)]
    a = pd.concat([fin[["home_team", "home_reg", "away_reg"]].set_axis(["team", "gf", "ga"], axis=1),
                   fin[["away_team", "away_reg", "home_reg"]].set_axis(["team", "gf", "ga"], axis=1)])
    t = a.groupby("team").agg(gp=("gf", "size"), gf=("gf", "sum"), ga=("ga", "sum")).reset_index()
    t["reg"] = 1                                              # marks the goals as regulation-only for team_rates()
    t.to_csv(out, index=False)
    side = lambda day: set(sch[sch.d == day].home_team) | set(sch[sch.d == day].away_team)
    b2b = sorted(side(date) & side((pd.Timestamp(date) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")))
    lg = t.gf.sum() / t.gp.sum()
    k = np.arange(15); pz = np.exp(-lg) * lg ** k / np.array([math.factorial(int(i)) for i in k])
    print(f"{out}: {len(t)} teams from {len(fin)} games, {lg:.2f} regulation goals per team per game")
    print(f"tied after 60 min: {(fin.ended_in != 'REG').mean():.1%} of {len(fin)} games; plain Poisson says {(pz ** 2).sum():.1%}, "
          f"the model with TIE_K = {TIE_K} says {TIE_K * (pz ** 2).sum() / (1 + (TIE_K - 1) * (pz ** 2).sum()):.1%}")
    print(f"back-to-back on {date}: --b2b {','.join(b2b) or '(none)'}")


def sync(folder: str, con) -> None:
    """Grade only what is new: every night in the ledger that still has a game without graded rows is checked
    against the ingest tables, and graded when the ingest has a finished game the ledger has not settled yet.
    Games still being played stay open; their slips wait."""
    sch = _ingest(folder, "schedule")
    sch = sch[sch.ended_in.notna()]
    done = set(sch.game_date.astype(str).str[:10] + "|" + sch.away_team + "@" + sch.home_team)
    have = {(n, g) for n, g in con.execute("SELECT night, game FROM synced")}       # settled from the full box score
    P = [(n, g) for n, g in con.execute("SELECT DISTINCT night, game FROM predictions") if (n, g) not in have]
    new = [(n, g) for n, g in P if f"{n}|{g}" in done]
    todo, waiting = sorted({n for n, _ in new}), sorted(f"{n} {g}" for n, g in P if f"{n}|{g}" not in done)
    print(f"ingest: {len(sch)} finished games, last {sch.game_date.max()}; nights with new results: {todo or 'none'}"
          + (f"; not finished or not ingested yet: {waiting}" if waiting else ""))
    for night in todo:
        out = str(Path(DB).with_name("results") / f"{night}.txt")
        ingest_results(night, folder, out)
        grade(night, out, con)
    con.executemany("INSERT OR IGNORE INTO synced VALUES (?, ?)", new)
    con.commit()
    if todo:
        try:                                                  # refresh the stats-vs-book arms from the new grades
            import contextlib, io
            import teamstats
            with contextlib.redirect_stdout(io.StringIO()):
                D = teamstats.lean(folder, DB)
            con.execute("DELETE FROM lean_legs")
            con.executemany("INSERT OR REPLACE INTO lean_legs VALUES (?,?,?,?,?,?,?,?)",
                            D[["night", "game", "kind", "agree", "strong", "p", "odds", "y"]].astype(object).values.tolist())
            con.commit()
        except Exception as e:
            print(f"  stats-vs-book arms not refreshed: {e}")
        report(con, None)


def fetch_results(date: str, out: str) -> None:
    """Writes the results text for one NHL date. Field names were read from live responses on 2026-11-16 (score and
    boxscore of a shootout game: the shootout goal is in the team score and in no skater's line, as read_results
    expects). The function itself has not been run end to end: check its first output against one box score.
    read_results() rejects anything inconsistent, so a wrong field shows up as an ERROR, not as a silent bad grade."""
    get = lambda u: _get(u.split("/v1/", 1)[1])
    L = [f"# NHL results {date} from api-web.nhle.com, fetched {pd.Timestamp.now('UTC'):%Y-%m-%d %H:%M}Z",
         "# skater: goals assists shots; goalie: saves goals-against"]
    for g in get(f"https://api-web.nhle.com/v1/score/{date}").get("games", []):
        if g.get("gameState") not in ("OFF", "FINAL"):
            print(f"  {g['awayTeam']['abbrev']}@{g['homeTeam']['abbrev']}: state {g.get('gameState')}, skipped")
            continue
        h, a = g["homeTeam"]["abbrev"], g["awayTeam"]["abbrev"]
        per = {h: [0, 0, 0, 0], a: [0, 0, 0, 0]}
        for goal in g.get("goals", []):
            pd_ = goal.get("periodDescriptor", {})
            if pd_.get("periodType") == "SO":
                continue
            per[goal["teamAbbrev"]][min(int(pd_.get("number", goal.get("period", 1))), 4) - 1] += 1
        L += ["", f"GAME {a}@{h}", f"P {h} " + " ".join(map(str, per[h])), f"P {a} " + " ".join(map(str, per[a]))]
        if g.get("gameOutcome", {}).get("lastPeriodType") == "SO":
            L.append(f"SO {h if g['homeTeam']['score'] > g['awayTeam']['score'] else a}")
        box = get(f"https://api-web.nhle.com/v1/gamecenter/{g['id']}/boxscore").get("playerByGameStats", {})
        for key, t in (("homeTeam", h), ("awayTeam", a)):
            for p in box.get(key, {}).get("forwards", []) + box.get(key, {}).get("defense", []):
                L.append(f"{t} {p['name']['default']}: {p.get('goals', 0)} {p.get('assists', 0)} {p.get('sog', 0)} "
                         f"{p.get('blockedShots', 0)} {p.get('hits', 0)}")
            for p in box.get(key, {}).get("goalies", []):
                if p.get("toi", "00:00") != "00:00":
                    L.append(f"G {t} {p['name']['default']}: {p.get('saves', 0)} {p.get('goalsAgainst', 0)}")
        L.append(f"COMPLETE {h} {a}")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text("\n".join(L) + "\n", encoding="utf-8")
    games, errs = read_results(out)
    print(f"{out}: {len(games)} games, {len(errs)} errors")
    for e in errs:
        print("ERROR", e)
    print("NOTE: box scores give first initial + surname ('A. Matthews'); player props match on surname + initial "
          "and refuse ambiguous matches. Check the no_result count after grading.")


# ------------------------------------------------------------------ selftest
def selftest() -> None:
    import tempfile
    od, mk, n = [], [], [0]

    def M(name, sels, sv=None):
        ids = []
        for nm, tid, price, cid, osv in sels:
            n[0] += 1
            od.append(dict(id=n[0], name=nm, typeId=tid, price=price, oddStatus=0, competitorId=cid, sv=osv))
            ids.append(n[0])
        mk.append(dict(id=1000 + len(mk), name=name, typeId=len(mk), desktopOddIds=[ids], sv=sv))

    T, B = "Toronto Maple Leafs", "Boston Bruins"
    M("Győztes (hosszabbítást és büntetőket beleértve)", [(T, 1, 1.80, 1, None), (B, 3, 2.05, 2, None)])
    M("1X2", [(T, 1, 2.30, 1, None), ("Döntetlen", 2, 4.10, None, None), (B, 3, 2.70, 2, None)])
    M("Gólok száma (hosszabbítást és büntetőket beleértve)", [("5,5 felett", 12, 1.75, None, "5.5"), ("5,5 alatt", 13, 2.10, None, "5.5"),
                                                             ("6,5 felett", 12, 2.40, None, "6.5"), ("6,5 alatt", 13, 1.58, None, "6.5")])
    M("Hendikep (hosszabbítást és büntetőket beleértve)", [(f"{T} (-1,5)", 1714, 2.90, 1, "-1.5"), (f"{B} (+1,5)", 1715, 1.42, 2, "+1.5")])
    M("Mindkét csapat szerez gólt", [("Igen", 74, 1.30, None, None), ("Nem", 76, 3.40, None, None)])
    M("1. harmad - 1X2", [(T, 1, 2.80, 1, None), ("Döntetlen", 2, 2.40, None, None), (B, 3, 3.10, 2, None)])
    M("1. harmad - gólok száma", [("1,5 felett", 12, 1.80, None, "1.5"), ("1,5 alatt", 13, 2.00, None, "1.5")])
    M(f"{T} gólok száma (hosszabbítást és büntetőket beleértve)", [("3,5 felett", 12, 2.20, None, "3.5"), ("3,5 alatt", 13, 1.65, None, "3.5")])
    M("Mindkét csapat szerez 2 vagy több gólt", [("Igen", 74, 1.48, None, None), ("Nem", 76, 2.55, None, None)])
    M(f"{B} pontok száma (hosszabbítást és büntetőket beleértve)", [("2,5 felett", 12, 1.55, None, "2.5"), ("2,5 alatt", 13, 2.40, None, "2.5")])
    M("Kétesély", [(f"{T} vagy döntetlen", 9, 1.50, 1, None), (f"{T} vagy {B}", 10, 1.24, None, None), (f"Döntetlen vagy {B}", 11, 1.45, 2, None)])
    M("Pontos eredmény", [("3:2", 500, 9.0, None, None)])
    M("Összes kapura lövés", [("60,5 felett", 12, 1.85, None, "60.5"), ("60,5 alatt", 13, 1.85, None, "60.5")])
    kid = dict(id=9001, childName="Auston Matthews", desktopOddIds=[[n[0] + 1], [n[0] + 2]])
    od += [dict(id=n[0] + 1, name="0,5 felett", typeId=12, price=1.90, oddStatus=0, sv="0.5"),
           dict(id=n[0] + 2, name="0,5 alatt", typeId=13, price=1.85, oddStatus=0, sv="0.5")]
    mk.append(dict(id=1999, name="Játékos gólok", typeId=77, desktopOddIds=[], childMarketIds=[9001]))
    mk.append(dict(id=1998, name="Játékos mentéseinek száma (hosszabbítást beleértve)", typeId=78, desktopOddIds=[], childMarketIds=[9002]))
    mk.append(dict(id=1997, name="Játékos kapott gól nélkül (hosszabbítást beleértve)", typeId=79, desktopOddIds=[], childMarketIds=[9003]))
    kid2 = dict(id=9002, childName="Joseph Woll", desktopOddIds=[[n[0] + 3, n[0] + 4]])      # saves ladder, shutout
    kid3 = dict(id=9003, childName="Joseph Woll", desktopOddIds=[[n[0] + 5, n[0] + 6]])
    od += [dict(id=n[0] + 3, name="25+", typeId=2425, price=1.50, oddStatus=0, sv="TOR|x"),
           dict(id=n[0] + 4, name="29+", typeId=2429, price=3.00, oddStatus=0, sv="TOR|x"),
           dict(id=n[0] + 5, name="Joseph Woll (TOR) 0,5 felett", typeId=2501, price=10.0, oddStatus=0, sv="0.5"),
           dict(id=n[0] + 6, name="Joseph Woll (TOR) 0,5 alatt", typeId=2502, price=1.01, oddStatus=0, sv="0.5")]
    kid4 = dict(id=9004, childName="Auston Matthews", desktopOddIds=[[n[0] + 7, n[0] + 8]])
    od += [dict(id=n[0] + 7, name="Auston Matthews (TOR) 1,5 felett", typeId=2501, price=1.70, oddStatus=0, sv="1.5"),
           dict(id=n[0] + 8, name="Auston Matthews (TOR) 1,5 alatt", typeId=2502, price=2.00, oddStatus=0, sv="1.5")]
    mk.append(dict(id=1996, name="Játékos blokkolt lövései (hosszabbítást beleértve)", typeId=80, desktopOddIds=[], childMarketIds=[9004]))
    ev = dict(markets=mk, odds=od, childMarkets=[kid, kid2, kid3, kid4],
              competitors=[dict(id=1, name=T, abbreviation="TOR"), dict(id=2, name=B, abbreviation="BOS")])
    tmp = Path(tempfile.mkdtemp())
    (tmp / "boards").mkdir()
    (tmp / "boards" / "TOR-BOS.net.json").write_text(json.dumps([dict(url="GetEventDetails", body=ev)]))
    (tmp / "teams.csv").write_text("team,gp,gf,ga\nTOR,18,62,50\nBOS,18,50,55\nMTL,18,54,54\n")
    # Toronto 3 (1,0,2) Boston 3 (0,2,1); Boston wins in the shootout -> final 3:4 with OT/SO, 3:3 regulation
    (tmp / "res.txt").write_text("GAME BOS@TOR\nP TOR 1 0 2\nP BOS 0 2 1\nSO BOS\nTOR A. Matthews: 1 0 5 2 3\nTOR B. Matthews: 0 0 1\nTOR Twin Name: 1 0 1\nTOR Twin Name: 0 0 2\nG TOR Joseph Woll: 27 3\n")
    global DB
    DB = str(tmp / "t.sqlite")
    con = connect()
    Bd = price("2026-01-01", str(tmp / "boards"), str(tmp / "teams.csv"), {"BOS"}, con)
    grade("2026-01-01", str(tmp / "res.txt"), con)
    g = pd.read_sql("SELECT type, scope, side, line, subject, result, status, p_book, p_model, p_method FROM predictions", con)
    R = {(r.type, r.scope, r.side, r.line): r.result for r in g.itertuples()}
    want = {("ML", "G", "home", None): "lost", ("ML", "G", "away", None): "won",          # shootout winner
            ("1X2", "G", "draw", None): "won", ("1X2", "G", "home", None): "lost",       # regulation only
            ("TOTAL", "G", "over", 5.5): "won", ("TOTAL", "G", "over", 6.5): "won",      # 3+3+1 SO goal = 7
            ("TOTAL", "G", "under", 6.5): "lost",
            ("HANDICAP", "G", "home", -1.5): "lost", ("HANDICAP", "G", "away", 1.5): "won",
            ("BTTS", "G", "yes", None): "won", ("1X2", "P1", "home", None): "won",       # P1 1:0
            ("TOTAL", "P1", "under", 1.5): "won", ("TEAM_TOTAL", "G", "under", 3.5): "won",   # TOR 3
            ("PL_GOALS", "G", "over", 0.5): "won", ("PL_GOALS", "G", "under", 0.5): "lost",
            ("BTTS2", "G", "yes", None): "won",                                          # 3:3 in regulation
            ("TEAM_TOTAL", "G", "over", 2.5): "won",                                     # BOS 3 + shootout goal = 4
            ("DC", "G", "1X", None): "won", ("DC", "G", "12", None): "lost", ("DC", "G", "X2", None): "won",
            ("PL_BLK", "G", "over", 1.5): "won",        # 2 blocked shots
            ("PL_SAVES", "G", "over", 24.5): "won", ("PL_SAVES", "G", "over", 28.5): "lost"}     # Woll 27 saves
    Rn = {(t, sc, s, None if l is None or (isinstance(l, float) and math.isnan(l)) else l): r for (t, sc, s, l), r in R.items()}
    bad = [(k, v, Rn.get(k)) for k, v in want.items() if Rn.get(k) != v]
    checks = [("settlement", not bad, bad),
              ("unparsed kept out", set(g[g.status == "unparsed"].type) == {"UNPARSED"} and (g.type == "UNPARSED").sum() == 5, None),      # incl. the two shutout rows
              ("no-vig sums to 1", abs(g[(g.type == "1X2") & (g.scope == "G")].p_book.sum() - 1) < 1e-9, None),
              ("ML model sums to 1", abs(g[g.type == "ML"].p_model.sum() - 1) < 1e-9, None),
              ("model favours TOR at home vs b2b BOS", float(g[(g.type == "ML") & (g.side == "home")].p_model.iloc[0]) > 0.5, None),
              ("total over/under model sums to 1", abs(g[(g.type == "TOTAL") & (g.scope == "G") & (g.line == 5.5)].p_model.sum() - 1) < 1e-9, None)]
    gm, errs = read_results(str(tmp / "res.txt"))
    twin = gm["BOS@TOR"]["sk"]
    checks.append(("two players with one name: both counted, neither matched",
                   sum(v["goals"] for v in twin.values()) == 2 and _find(twin, "Twin Name") is None, list(twin)))
    (tmp / "bad.txt").write_text("GAME BOS@TOR\nP TOR 1 0 2\nP BOS 0 2 1\n")
    _, errs2 = read_results(str(tmp / "bad.txt"))
    checks += [("good results accepted", not errs, errs), ("tie without OT/SO rejected", bool(errs2), None)]
    report(con, None)
    checks += _ingest_selftest(tmp) + _slip_selftest(tmp)
    for nm, ok, info in checks:
        print(("ok   " if ok else "FAIL ") + nm + (f"  {info}" if info and not ok else ""))
    print("PASS" if all(c[1] for c in checks) else "FAIL")
    sys.exit(0 if all(c[1] for c in checks) else 1)


def _ingest_selftest(tmp: Path) -> list:
    """nhl_ingest.py's tables (its column names) -> results text -> the same shootout game as above."""
    d = tmp / "ing"
    d.mkdir()
    w = lambda name, rows: pd.DataFrame(rows).to_csv(d / f"nhl_2026_{name}.csv", index=False)
    w("schedule", [dict(game_id=1, game_date="2026-01-01", home_team="TOR", away_team="BOS", ended_in="SO", winner="BOS",
                        home_reg=3, away_reg=3, score_checks=True),
                   dict(game_id=0, game_date="2025-12-31", home_team="BOS", away_team="MTL", ended_in="REG", winner="BOS",
                        home_reg=4, away_reg=1, score_checks=True)])
    w("period_scores", [dict(game_id=1, team=t, period=p, goals=g) for t, gs in (("TOR", (1, 0, 2, 0, 0)), ("BOS", (0, 2, 1, 0, 1)))
                        for p, g in zip(("1", "2", "3", "OT", "SO"), gs)])
    sk = lambda t, nm, short, pl, g, a, s: dict(game_id=1, team=t, player_name=nm, player_short=short, played=pl, goals=g,
                                                assists=a, shots_on_goal=s, blocked_shots=2, hits=3)
    w("player_game_stats", [sk("TOR", "Auston Matthews", "A. Matthews", True, 1, 0, 5), sk("TOR", "T One", "T. One", True, 2, 0, 3),
                            sk("TOR", "Sat Out", "S. Out", False, 0, 0, 0), sk("BOS", "B One", "B. One", True, 3, 0, 4)])
    w("goalie_game_stats", [dict(game_id=1, team="TOR", player_name="Joseph Woll", player_short="J. Woll", played=True,
                                 saves=27, goals_against=3, decision="O")])
    ingest_results("2026-01-01", str(d), str(tmp / "ing.txt"))
    games, errs = read_results(str(tmp / "ing.txt"))
    g = games.get("BOS@TOR", {})
    ok = (not errs and g.get("so") == "BOS" and g["per"]["TOR"] == [1, 0, 2, 0] and "Sat Out" not in g["sk"]
          and g["sk"]["Auston Matthews"]["blk"] == 2 and g["gk"]["Joseph Woll"] == dict(team="TOR", saves=27, ga=3))
    ingest_teams("2026-01-01", str(d), str(tmp / "ing_teams.csv"))
    t = pd.read_csv(tmp / "ing_teams.csv").set_index("team")
    return [("ingest tables -> results (shootout, scratch left out)", ok, errs or g),
            ("ingest tables -> teams.csv (regulation goals, games before the date only)", t.loc["BOS", "gf"] == 4 and len(t) == 2, None)]


def _slip_selftest(tmp: Path) -> list:
    """5 games, each with a 1.45 home favourite and a 1.50 under: build a 3x slip, grade a night where every leg
    wins and a night where every leg loses."""
    global DB
    DB = str(tmp / "s.sqlite")
    con = connect()
    bd = tmp / "b2"
    bd.mkdir()
    for i, (h, a) in enumerate([("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF"), ("GGG", "HHH"), ("III", "JJJ")]):
        hn, an, b = f"{h} Home", f"{a} Away", i * 10
        od = [dict(id=b + 1, name=hn, typeId=1, price=1.45, oddStatus=0, competitorId=1),
              dict(id=b + 2, name=an, typeId=3, price=2.85, oddStatus=0, competitorId=2),
              dict(id=b + 3, name="5,5 felett", typeId=12, price=2.60, oddStatus=0, sv="5.5"),
              dict(id=b + 4, name="5,5 alatt", typeId=13, price=1.50, oddStatus=0, sv="5.5")]
        mk = [dict(id=b + 100, name="Győztes (hosszabbítást és büntetőket beleértve)", typeId=1, desktopOddIds=[[b + 1, b + 2]]),
              dict(id=b + 101, name="Gólok száma (hosszabbítást és büntetőket beleértve)", typeId=2, desktopOddIds=[[b + 3, b + 4]])]
        ev = dict(markets=mk, odds=od, competitors=[dict(id=1, name=hn, abbreviation=h), dict(id=2, name=an, abbreviation=a)])
        (bd / f"{h}.net.json").write_text(json.dumps([dict(url="GetEventDetails", body=ev)]))
    pairs = [("AAA", "BBB"), ("CCC", "DDD"), ("EEE", "FFF"), ("GGG", "HHH"), ("III", "JJJ")]
    (tmp / "w.txt").write_text("".join(f"GAME {a}@{h}\nP {h} 1 1 1\nP {a} 0 1 0\n" for h, a in pairs))     # 3:1, total 4
    (tmp / "l.txt").write_text("".join(f"GAME {a}@{h}\nP {h} 1 1 1\nP {a} 2 1 1\n" for h, a in pairs))     # 3:4, total 7
    out = []
    for night, res, want in (("2026-02-01", "w.txt", "won"), ("2026-02-02", "l.txt", "lost")):
        price(night, str(bd), None, set(), con)
        slips(night, [3, 50], con)
        grade(night, str(tmp / res), con)
        s = pd.read_sql("SELECT * FROM slips WHERE night = ?", con, params=[night])
        lg = pd.read_sql("SELECT * FROM slip_legs WHERE night = ?", con, params=[night])
        one = len(s) == 1 and 3 <= s.odds[0] <= 3.6 and lg.game.nunique() == len(lg) == s.legs[0]
        out += [(f"slip {night}: one 3x slip, one leg per game, 50x not reachable", one, s.to_dict("records")),
                (f"slip {night}: P = product of legs, EV negative", one and abs(s.p[0] - lg.p.prod()) < 1e-9 and (s.ev[0] < 0 or night != "2026-02-01"), None),
                (f"slip {night}: graded {want}", one and s.result[0] == want and abs(s.paid[0] - (s.odds[0] if want == "won" else 0)) < 1e-3, None)]
    report(con, None)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    s = sub.add_parser("markets"); s.add_argument("boards")
    s = sub.add_parser("price"); s.add_argument("--date", required=True); s.add_argument("--boards", required=True)
    s.add_argument("--teams"); s.add_argument("--b2b", default="")
    s = sub.add_parser("results"); s.add_argument("--date", required=True); s.add_argument("--out", required=True)
    s.add_argument("--ingest", help="folder with nhl_ingest.py's CSV tables (preferred over the live API)")
    s = sub.add_parser("grade"); s.add_argument("--date", required=True); s.add_argument("--results", required=True)
    s = sub.add_parser("report"); s.add_argument("--test-night")
    s = sub.add_parser("sync"); s.add_argument("--ingest", required=True)
    s = sub.add_parser("strategies"); s.add_argument("--date", required=True)
    s.add_argument("--targets", nargs="+", type=float, default=[3, 7]); s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--only", nargs="*", choices=list(STRATEGIES), help="rebuild only these strategies")
    sub.add_parser("arms")
    s = sub.add_parser("compare"); s.add_argument("--date", required=True); s.add_argument("--ingest", required=True)
    s = sub.add_parser("teams"); s.add_argument("--date", required=True); s.add_argument("--out", default="teams.csv")
    s.add_argument("--ingest", help="folder with nhl_ingest.py's CSV tables")
    s = sub.add_parser("slips"); s.add_argument("--date", required=True)
    s.add_argument("--targets", nargs="+", type=float, default=[3, 5, 7]); s.add_argument("--exclude", nargs="*", default=[])
    s.add_argument("--players", action="store_true"); s.add_argument("--replace", action="store_true")
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    if a.cmd == "markets":
        B = pd.concat([pd.DataFrame(board_rows(f)) for f in sorted(Path(a.boards).glob("*.json"))])
        t = B.groupby(["type", "scope", "market"]).size().rename("rows").reset_index().sort_values(["type", "rows"], ascending=[True, False])
        pd.set_option("display.width", 220, "display.max_colwidth", 90, "display.max_rows", 2000)
        print(t.to_string(index=False))
        print(f"\n{(B.type != 'UNPARSED').mean():.0%} of {len(B):,} rows typed; UNPARSED rows are stored but never graded")
    elif a.cmd == "price":
        price(a.date, a.boards, a.teams, {x.strip().upper() for x in a.b2b.split(",") if x.strip()}, connect())
    elif a.cmd == "teams":
        ingest_teams(a.date, a.ingest, a.out) if a.ingest else fetch_teams(a.date, a.out)
    elif a.cmd == "slips":
        slips(a.date, a.targets, connect(), a.players, a.exclude, a.replace)
    elif a.cmd == "results":
        ingest_results(a.date, a.ingest, a.out) if a.ingest else fetch_results(a.date, a.out)
    elif a.cmd == "grade":
        grade(a.date, a.results, connect())
    elif a.cmd == "report":
        report(connect(), a.test_night)
    elif a.cmd == "arms":
        pd.set_option("display.width", 200)
        con = connect()
        print(arms(con).round(3).to_string(index=False))
        SA = stat_arms(con)
        if len(SA):
            print("\nteam stats against the book (legs priced 40-60%):")
            print(SA.round(3).to_string(index=False))
    elif a.cmd == "compare":
        import teamstats
        txt = teamstats.compare(a.ingest, a.date, DB)
        Path(DB).with_name(f"compare_{a.date}.md").write_text(txt, encoding="utf-8")
    elif a.cmd == "strategies":
        strategy_slips(a.date, a.targets, connect(), a.exclude, a.only)
    elif a.cmd == "sync":
        sync(a.ingest, connect())


if __name__ == "__main__":
    main()
