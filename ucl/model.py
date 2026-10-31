"""Joint attack/defense Poisson model over the domestic leagues + UCL.

Every team gets a rating = its league's effect + its own deviation:

    log λ_home = μ + shift[c] + home[c]·(1 - neutral)
                 + (A[league(h)] + a[h]) + (D[league(a)] + d[a])
    log λ_away = μ + shift[c] + (A[league(a)] + a[a]) + (D[league(h)] + d[h])

shift[c] is a scoring-rate offset for each European competition (0 for domestic
leagues, whose rate is already inside A + D). Domestic games pin down each team
relative to its own league; the European meetings
between leagues pin down the league effects A/D relative to each other. So league
strength falls out of the fit, with no UEFA coefficients needed. Team deviations
are ridge-penalised toward 0 (= league average), which is what a team with little
data falls back to. Clubs with no domestic league in ucl.db share the "other" league.

Matches are weighted by exp(-ln2 · age / half-life), and fit() only ever sees
matches strictly before `as_of`, so a backtest that refits per date is leak-free.

The same model fits corners (target="corners") from the matches that have corner
counts: UCL games, plus EPL copied from predictable.db. Other leagues' corner
levels are then identified through their clubs' UCL games only.
"""
import math
from datetime import datetime, timezone

import numpy as np
from scipy.optimize import minimize
from scipy.stats import nbinom, poisson

from ucl import config as C


def _days_between(earlier_iso, later_iso):
    a = datetime.fromisoformat(earlier_iso.replace("Z", "+00:00"))
    b = datetime.fromisoformat(later_iso.replace("Z", "+00:00"))
    if a.tzinfo is None:
        a = a.replace(tzinfo=timezone.utc)
    if b.tzinfo is None:
        b = b.replace(tzinfo=timezone.utc)
    return (b - a).total_seconds() / 86400.0


def team_leagues(matches):
    """Map team -> domestic league: the one it played most domestic games in,
    preferring the most recent season (handles promotion/relegation)."""
    counts = {}
    for m in matches:
        if m["competition"] in C.EUROPEAN:
            continue
        for tid in (m["home_id"], m["away_id"]):
            key = (m["season"], m["competition"])
            counts.setdefault(tid, {}).setdefault(key, 0)
            counts[tid][key] += 1
    return {tid: max(c, key=lambda k: (k[0], c[k]))[1] for tid, c in counts.items()}


TARGETS = {"goals": ("home_goals", "away_goals"), "corners": ("home_corners", "away_corners")}


class UCLModel:
    def __init__(self, half_life=C.HALF_LIFE_DAYS, team_ridge=C.TEAM_RIDGE,
                 league_ridge=C.LEAGUE_RIDGE, rho=C.DIXON_COLES_RHO, target="goals"):
        self.target = target
        self.half_life = half_life
        self.team_ridge = team_ridge
        self.league_ridge = league_ridge
        self.rho = rho
        self.fitted = False

    # ------------------------------------------------------------------ fitting

    def fit(self, matches, as_of, leagues):
        """matches: rows with competition/date/home_id/away_id/home_goals/away_goals/
        neutral/status. Only FT matches strictly before as_of are used.
        leagues: team -> league map from team_leagues() (season membership is
        known in advance, so using the full map is not look-ahead)."""
        hcol, acol = TARGETS[self.target]
        train = [m for m in matches if m["status"] == "FT" and m["date"] < as_of
                 and m[hcol] is not None and m[acol] is not None]
        if not train:
            raise ValueError(f"No completed matches before {as_of}")

        self.leagues = dict(leagues)
        self.league_names = list(C.DOMESTIC) + [C.OTHER_LEAGUE]
        lidx = {l: i for i, l in enumerate(self.league_names)}
        self.comps = list(C.COMPETITIONS)
        cidx = {c: i for i, c in enumerate(self.comps)}

        teams = sorted({m["home_id"] for m in train} | {m["away_id"] for m in train})
        self.team_index = {t: i for i, t in enumerate(teams)}
        nT, nL, nC = len(teams), len(self.league_names), len(self.comps)

        h = np.array([self.team_index[m["home_id"]] for m in train])
        a = np.array([self.team_index[m["away_id"]] for m in train])
        team_l = np.array([lidx[self.leagues.get(t, C.OTHER_LEAGUE)] for t in teams])
        lh, la = team_l[h], team_l[a]
        comp = np.array([cidx[m["competition"]] for m in train])
        eidx = {c: i for i, c in enumerate(C.EUROPEAN)}
        nE = len(C.EUROPEAN)
        # Match -> European competition index; domestic matches point at a dummy
        # slot (nE) that is fixed at zero.
        euro = np.array([eidx.get(m["competition"], nE) for m in train])
        is_euro = (euro < nE).astype(float)
        home_on = np.array([0.0 if m["neutral"] else 1.0 for m in train])
        yh = np.array([m[hcol] for m in train], dtype=float)
        ya = np.array([m[acol] for m in train], dtype=float)
        age = np.array([_days_between(m["date"], as_of) for m in train])
        w = np.exp(-math.log(2) * age / self.half_life)

        # Parameter layout: [mu, shift[nE], home[nC], A[nL], D[nL], a[nT], d[nT]].
        # Competitions with no training rows keep home = 0 (gradient is zero there).
        s_home = 1 + nE
        s_A = s_home + nC
        s_D = s_A + nL
        s_a = s_D + nL
        s_d = s_a + nT
        n_params = s_d + nT
        tr, lr = self.team_ridge, self.league_ridge

        def unpack(p):
            return (p[0], p[1:s_home], p[s_home:s_A], p[s_A:s_D], p[s_D:s_a], p[s_a:s_d], p[s_d:])

        def objective(p):
            mu, shift, home, A, D, att, dfn = unpack(p)
            base = mu + np.append(shift, 0.0)[euro]
            eta_h = base + home[comp] * home_on + A[lh] + att[h] + D[la] + dfn[a]
            eta_a = base + A[la] + att[a] + D[lh] + dfn[h]
            lam_h, lam_a = np.exp(eta_h), np.exp(eta_a)

            nll = np.sum(w * (lam_h - yh * eta_h)) + np.sum(w * (lam_a - ya * eta_a))
            nll += tr * (att @ att + dfn @ dfn) + lr * (A @ A + D @ D)

            # d(nll)/d(eta) per match, then scatter onto parameters.
            gh = w * (lam_h - yh)
            ga = w * (lam_a - ya)
            g = np.zeros(n_params)
            g[0] = gh.sum() + ga.sum()
            g[1:s_home] = np.bincount(euro, (gh + ga) * is_euro, nE + 1)[:nE]
            g[s_home:s_A] = np.bincount(comp, weights=gh * home_on, minlength=nC)
            g[s_A:s_D] = np.bincount(lh, gh, nL) + np.bincount(la, ga, nL) + 2 * lr * A
            g[s_D:s_a] = np.bincount(la, gh, nL) + np.bincount(lh, ga, nL) + 2 * lr * D
            g[s_a:s_d] = np.bincount(h, gh, nT) + np.bincount(a, ga, nT) + 2 * tr * att
            g[s_d:] = np.bincount(a, gh, nT) + np.bincount(h, ga, nT) + 2 * tr * dfn
            return nll, g

        p0 = np.zeros(n_params)
        p0[0] = math.log(max(0.1, np.average(np.r_[yh, ya], weights=np.r_[w, w])))
        res = minimize(objective, p0, jac=True, method="L-BFGS-B", options={"maxiter": 2000})
        if not res.success:
            print(f"  WARN: fit did not fully converge ({res.message})")

        mu, shift, home, A, D, att, dfn = unpack(res.x)
        self.mu = float(mu)
        self.shift = dict(zip(C.EUROPEAN, shift))
        self.home = dict(zip(self.comps, home))
        self.A = dict(zip(self.league_names, A))
        self.D = dict(zip(self.league_names, D))
        self.att = dict(zip(teams, att))
        self.dfn = dict(zip(teams, dfn))
        self.n_train = len(train)
        self.fitted = True
        return self

    # --------------------------------------------------------------- predicting

    def _strength(self, team):
        league = self.leagues.get(team, C.OTHER_LEAGUE)
        return (self.A[league] + self.att.get(team, 0.0),
                self.D[league] + self.dfn.get(team, 0.0))

    def expected_goals(self, home_id, away_id, competition=C.UCL, neutral=False):
        ha, hd = self._strength(home_id)
        aa, ad = self._strength(away_id)
        base = self.mu + self.shift.get(competition, 0.0)
        home = 0.0 if neutral else self.home[competition]
        return math.exp(base + home + ha + ad), math.exp(base + aa + hd)

    def score_grid(self, lam_h, lam_a, dixon_coles=True):
        """P(home=i, away=j) for i, j in 0..MAX_GOALS, normalised."""
        k = np.arange(C.MAX_GOALS + 1)
        grid = np.outer(poisson.pmf(k, lam_h), poisson.pmf(k, lam_a))
        if dixon_coles:
            r = self.rho
            grid[0, 0] *= 1 - lam_h * lam_a * r
            grid[0, 1] *= 1 + lam_h * r
            grid[1, 0] *= 1 + lam_a * r
            grid[1, 1] *= 1 - r
        return grid / grid.sum()

    def predict(self, home_id, away_id, competition=C.UCL, neutral=False):
        lam_h, lam_a = self.expected_goals(home_id, away_id, competition, neutral)
        return markets(self.score_grid(lam_h, lam_a), lam_h, lam_a)

    def predict_corners(self, home_id, away_id, competition=C.UCL, neutral=False):
        """For a model fitted with target="corners": expected_goals() then returns
        expected corners, since the structure is identical."""
        return corner_markets(*self.expected_goals(home_id, away_id, competition, neutral))

    # ------------------------------------------------------------ knockout ties

    def _goal_diff_dist(self, grid):
        """Distribution of (home - away) as {diff: p}."""
        n = grid.shape[0]
        i, j = np.indices(grid.shape)
        diffs = (i - j).ravel() + (n - 1)
        p = np.bincount(diffs, weights=grid.ravel(), minlength=2 * n - 1)
        return {d - (n - 1): float(p[d]) for d in range(2 * n - 1) if p[d] > 0}

    def _extra_time_win(self, home_id, away_id, neutral):
        """P(home side wins extra time + penalties), with 30 mins at a third of
        the 90-min rate and penalties as a coin flip."""
        lam_h, lam_a = self.expected_goals(home_id, away_id, C.UCL, neutral)
        et = self._goal_diff_dist(self.score_grid(lam_h / 3, lam_a / 3, dixon_coles=False))
        win = sum(p for d, p in et.items() if d > 0)
        draw = et.get(0, 0.0)
        return win + 0.5 * draw

    def tie_advance_prob(self, team1, team2, leg1_score=None, single_leg_neutral=False):
        """P(team1 advances). team1 hosts leg 1; team2 hosts leg 2 (where any extra
        time is played). leg1_score = (team1 goals, team2 goals) once played.
        single_leg_neutral=True scores a one-off tie such as the final."""
        if single_leg_neutral:
            lam_h, lam_a = self.expected_goals(team1, team2, C.UCL, neutral=True)
            diff = self._goal_diff_dist(self.score_grid(lam_h, lam_a))
            et = self._extra_time_win(team1, team2, neutral=True)
            return sum(p for d, p in diff.items() if d > 0) + diff.get(0, 0.0) * et

        if leg1_score is None:
            g1 = self.score_grid(*self.expected_goals(team1, team2, C.UCL))
            leg1 = self._goal_diff_dist(g1)                      # team1 - team2
        else:
            leg1 = {leg1_score[0] - leg1_score[1]: 1.0}
        g2 = self.score_grid(*self.expected_goals(team2, team1, C.UCL))
        leg2 = {-d: p for d, p in self._goal_diff_dist(g2).items()}  # flip to team1's view

        p_win = p_level = 0.0
        for d1, p1 in leg1.items():
            for d2, p2 in leg2.items():
                if d1 + d2 > 0:
                    p_win += p1 * p2
                elif d1 + d2 == 0:
                    p_level += p1 * p2
        # Extra time is at team2's ground; flip to team1's perspective.
        et_team1 = 1.0 - self._extra_time_win(team2, team1, neutral=False)
        return p_win + p_level * et_team1


def corner_markets(lam_h, lam_a, dispersion=C.CORNER_DISPERSION, lines=C.CORNER_LINES):
    """Total corners as a negative binomial with variance = dispersion x mean. Each
    side's count is overdispersed, but the two are negatively correlated (whoever
    dominates wins the corners), so the total is close to Poisson: EPL 24/25 and
    25/26 had variance/mean 1.17 and 1.06."""
    mean = lam_h + lam_a
    if dispersion > 1.0:
        n = mean / (dispersion - 1.0)  # scipy's NB(n, p): mean n(1-p)/p, var mean/p
        dist = nbinom(n, 1.0 / dispersion)
    else:
        dist = poisson(mean)
    return {
        "home_corners": lam_h,
        "away_corners": lam_a,
        "total_corners": mean,
        "over": {line: float(dist.sf(math.floor(line))) for line in lines},
    }


def markets(grid, lam_h, lam_a):
    i, j = np.indices(grid.shape)
    total = i + j
    return {
        "home_xg": lam_h,
        "away_xg": lam_a,
        "home_win": float(grid[i > j].sum()),
        "draw": float(grid[i == j].sum()),
        "away_win": float(grid[i < j].sum()),
        "btts": float(grid[(i > 0) & (j > 0)].sum()),
        "over": {line: float(grid[total > line].sum()) for line in (0.5, 1.5, 2.5, 3.5, 4.5)},
        "most_likely_score": "{}-{}".format(*np.unravel_index(grid.argmax(), grid.shape)),
    }
