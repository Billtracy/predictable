"""Team-level pace x efficiency model.

Two weighted ridge regressions on every final game before `as_of`:

    pace_g      = mu_p + p[home] + p[away] + ctx_pace          (possessions per 48 min)
    eff_{o vs d} = mu_e + hca * home(o) + O[o] + D[d] + ctx_eff (points per 100 poss)

so a team is described by how fast it plays (p), how well it scores (O) and how well
it defends (D, negative = good). Per-game weight is exp(-ln2 * age / half_life),
times `prev_season_weight` for games from an earlier season (rosters turn over in the
summer). Team effects are ridge-shrunk toward 0 (league average); the penalty is in
"pseudo-games": ridge=10 means a team's rating starts as if it had 10 average games.

Prediction for 48 minutes:  total = pace * (eff_h + eff_a) / 100,
                            margin = pace * (eff_h - eff_a) / 100.
Total and margin are Normal. A game level after 48 minutes (margin rounds to 0) goes
to overtime, which adds about pace * 5/48 * (eff_h + eff_a) / 100 points per period;
totals settle including OT, so P(over) is a mixture over 0 / 1+ overtimes.
"""
import bisect
import math
from datetime import datetime, timedelta

import numpy as np
from scipy.stats import norm

LN2 = math.log(2)

DEFAULTS = dict(
    half_life=60.0,          # days
    prev_season_weight=0.5,
    pace_ridge=10.0,
    eff_ridge=10.0,
    sd_total=18.0,           # SD of the 48-minute total around the model mean
    sd_margin=12.5,
    second_ot_prob=0.12,     # P(another OT | already in OT)
    # Late-game play (fouling, playing for the tie) piles 48-minute margins up at 0:
    # 2022-23 had 6.0% overtime games vs 3.0% from the Normal margin's mass at 0.
    ot_mult=2.0,
)


def _dt(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


def local_day(iso):
    """US calendar date of a tip-off (ET/PT evening games are next-day in UTC)."""
    return (_dt(iso) - timedelta(hours=6)).date()


def possessions(g):
    ph = g["home_fga"] - g["home_oreb"] + g["home_tov"] + 0.44 * g["home_fta"]
    pa = g["away_fga"] - g["away_oreb"] + g["away_tov"] + 0.44 * g["away_fta"]
    return (ph + pa) / 2.0


def has_box(g):
    return g["status"] == "FT" and g.get("home_fga") is not None and g.get("away_fga") is not None


def rest_context(games):
    """game_id -> {'home_rest': days, 'away_rest': days, 'home_b2b': 0/1, ...} from the
    full schedule (final and upcoming). Rest = days since the team's previous game,
    capped at 4; the first game of a season counts as 4."""
    last = {}
    out = {}
    for g in sorted(games, key=lambda g: g["date"]):
        day = local_day(g["date"])
        ctx = {}
        for side in ("home", "away"):
            tid = g[f"{side}_id"]
            prev = last.get(tid)
            rest = 4 if prev is None or prev[1] != g["season"] else min((day - prev[0]).days, 4)
            ctx[f"{side}_rest"] = rest
            ctx[f"{side}_b2b"] = int(rest <= 1)
            last[tid] = (day, g["season"])
        out[g["id"]] = ctx
    return out


def _ridge(X, y, w, penalty):
    """argmin sum w (y - X b)^2 + sum penalty_j b_j^2."""
    Xw = X * w[:, None]
    A = X.T @ Xw + np.diag(penalty)
    return np.linalg.solve(A, Xw.T @ y)


class Design:
    """Regression rows for every final game with a box score, built once and sliced
    by date in each fit (a walk-forward backtest refits ~700 times)."""

    def __init__(self, games, context):
        hist = sorted((g for g in games if has_box(g)), key=lambda g: g["date"])
        teams = sorted({g["home_id"] for g in hist} | {g["away_id"] for g in hist})
        self.teams = {t: k for k, t in enumerate(teams)}
        n = self.n = len(teams)
        m = len(hist)
        self.dates = [g["date"] for g in hist]
        self.days = np.array([_dt(d).timestamp() / 86400.0 for d in self.dates])
        self.season = np.array([g["season"] for g in hist])

        # pace rows: [mu, p_team x n, b2b count]
        self.Xp = np.zeros((m, 1 + n + 1))
        self.yp = np.zeros(m)
        # efficiency rows, two per game: [mu, hca, O x n, D x n, off_b2b, def_b2b]
        self.Xe = np.zeros((2 * m, 2 + 2 * n + 2))
        self.ye = np.zeros(2 * m)
        for i, g in enumerate(hist):
            c = context[g["id"]]
            h, a = self.teams[g["home_id"]], self.teams[g["away_id"]]
            poss = possessions(g)
            minutes = 48 + 5 * (max(g["periods"] or 4, 4) - 4)
            self.yp[i] = poss * 48.0 / minutes
            self.Xp[i, 0] = 1.0
            self.Xp[i, 1 + h] += 1.0
            self.Xp[i, 1 + a] += 1.0
            self.Xp[i, 1 + n] = c["home_b2b"] + c["away_b2b"]
            for j, (o, d, pts, home, ob, db) in enumerate(
                    ((h, a, g["home_pts"], 1, c["home_b2b"], c["away_b2b"]),
                     (a, h, g["away_pts"], 0, c["away_b2b"], c["home_b2b"]))):
                r = 2 * i + j
                self.ye[r] = 100.0 * pts / poss
                self.Xe[r, 0] = 1.0
                self.Xe[r, 1] = home * (1 - g["neutral"])
                self.Xe[r, 2 + o] = 1.0
                self.Xe[r, 2 + n + d] = 1.0
                self.Xe[r, 2 + 2 * n] = ob
                self.Xe[r, 3 + 2 * n] = db

    def upto(self, as_of):
        """Number of rows strictly before as_of."""
        return bisect.bisect_left(self.dates, as_of)


class NBAModel:
    def __init__(self, **params):
        self.p = {**DEFAULTS, **params}

    # ----------------------------------------------------------------- fit
    def fit(self, design, as_of, season):
        """Fit on the design's games strictly before as_of (ISO). season: the season
        being predicted, for the previous-season down-weight."""
        m = design.upto(as_of)
        if m == 0:
            raise ValueError("no games before as_of")
        n = design.n
        self.teams = design.teams
        age = _dt(as_of).timestamp() / 86400.0 - design.days[:m]
        w = np.exp(-LN2 * age / self.p["half_life"])
        w = w * np.where(design.season[:m] == season, 1.0, self.p["prev_season_weight"])

        pen_p = np.r_[1e-6, np.full(n, self.p["pace_ridge"]), 1e-3]
        bp = _ridge(design.Xp[:m], design.yp[:m], w, pen_p)
        self.mu_pace, self.pace, self.b2b_pace = bp[0], bp[1:1 + n], bp[1 + n]

        pen_e = np.r_[1e-6, 1e-3, np.full(2 * n, self.p["eff_ridge"]), 1e-3, 1e-3]
        be = _ridge(design.Xe[:2 * m], design.ye[:2 * m], np.repeat(w, 2), pen_e)
        self.mu_eff, self.hca = be[0], be[1]
        self.off, self.dfn = be[2:2 + n], be[2 + n:2 + 2 * n]
        self.b2b_off, self.b2b_def = be[2 + 2 * n], be[3 + 2 * n]
        return self

    # ------------------------------------------------------------- predict
    def _team(self, arr, tid):
        k = self.teams.get(tid)
        return 0.0 if k is None else arr[k]

    def expected(self, g, ctx):
        """48-minute expectations for one game: pace, home/away points."""
        h, a = g["home_id"], g["away_id"]
        pace = (self.mu_pace + self._team(self.pace, h) + self._team(self.pace, a)
                + self.b2b_pace * (ctx["home_b2b"] + ctx["away_b2b"]))
        hca = self.hca * (1 - g["neutral"])
        eff_h = (self.mu_eff + hca + self._team(self.off, h) + self._team(self.dfn, a)
                 + self.b2b_off * ctx["home_b2b"] + self.b2b_def * ctx["away_b2b"])
        eff_a = (self.mu_eff + self._team(self.off, a) + self._team(self.dfn, h)
                 + self.b2b_off * ctx["away_b2b"] + self.b2b_def * ctx["home_b2b"])
        return pace, pace * eff_h / 100.0, pace * eff_a / 100.0

    def predict(self, g, ctx):
        pace, pts_h, pts_a = self.expected(g, ctx)
        reg_total, margin = pts_h + pts_a, pts_h - pts_a
        sd_m = self.p["sd_margin"]
        p_ot = self.p["ot_mult"] * (norm.cdf((0.5 - margin) / sd_m) - norm.cdf((-0.5 - margin) / sd_m))
        ot_pts = reg_total * 5.0 / 48.0
        n_ot = 1.0 / (1.0 - self.p["second_ot_prob"])       # expected OTs given >= 1
        return {
            "pace": pace, "home_pts": pts_h, "away_pts": pts_a,
            "reg_total": reg_total, "margin": margin, "p_ot": p_ot,
            "total": reg_total + p_ot * ot_pts * n_ot,      # expected final total
            "p_home_win": 1.0 - norm.cdf(-margin / sd_m),   # OT splits ~evenly
            "_ot_pts": ot_pts, "_n_ot": n_ot,
        }

    def p_over(self, pred, line):
        """P(final total > line), mixing regulation-only and overtime outcomes."""
        sd_t = self.p["sd_total"]
        reg = 1.0 - norm.cdf((line - pred["reg_total"]) / sd_t)
        ot_mean = pred["reg_total"] + pred["_ot_pts"] * pred["_n_ot"]
        ot_sd = math.sqrt(sd_t ** 2 + (0.35 * pred["_ot_pts"]) ** 2)
        ot = 1.0 - norm.cdf((line - ot_mean) / ot_sd)
        return (1 - pred["p_ot"]) * reg + pred["p_ot"] * ot

    def p_home_cover(self, pred, spread):
        """P(home margin + spread > 0); spread is the home handicap (-6.5 = home favoured)."""
        return 1.0 - norm.cdf((-spread - pred["margin"]) / self.p["sd_margin"])

    def ratings(self, info=None):
        """Per-team pace / offence / defence / net rating (per 100 poss), best net first."""
        rows = []
        for tid, k in self.teams.items():
            rows.append({"team_id": tid, "abbr": (info or {}).get(tid, {}).get("abbreviation", tid),
                         "pace": self.mu_pace + self.pace[k],   # vs an average opponent
                         "off": self.mu_eff + self.off[k], "def": self.mu_eff + self.dfn[k],
                         "net": self.off[k] - self.dfn[k]})
        return sorted(rows, key=lambda r: -r["net"])
