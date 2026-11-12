"""Fitted attack/defence model for leagues with several seasons of history (EPL).

A competition opts in with a "league" key in config.COMPETITIONS. Every
competition sharing that key is one history: the model in ucl/model.py is fitted
by maximum likelihood on all of the league's completed matches before a cutoff,
weighted by recency, once for goals and once for corners. Unlike the ratio
ratings in models/team_ratings.py, its expected totals are unbiased (the league
mean is a fitted parameter) and its market probabilities come from an exact
distribution, not a simulation.

Clubs are keyed by SportRadar id where there is one: the same club has different
teams.id values across seasons ("Brighton" / "Brighton & Hove Albion").
"""
import os
import sys
from datetime import datetime, timezone

import numpy as np
from scipy.stats import nbinom, poisson

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import COMPETITIONS
from db.database import db_version, get_db
from ucl.model import UCLModel

FINISHED = ("FT", "AET", "PEN")

# (half-life in days, team ridge, league ridge) per target: the values tuned for
# the UCL model. On the EPL walk-forward (24/25 to 26/27), half-life 180-730 with
# team ridge 4-16 for goals and 8-32 for corners all score within noise of these
# (tip hit rate within a point or two, line Brier within 0.003), so they are kept.
FIT_PARAMS = {"goals": (365.0, 8.0, 5.0), "corners": (365.0, 16.0, 10.0)}

# Variance / mean of total corners around the model's expectation. EPL totals are
# close to Poisson (raw variance / mean 1.04-1.17 across 24/25 to 26/27), and
# 1.0-1.3 score within 0.001 line Brier of each other on the walk-forward.
CORNER_DISPERSION = 1.1

# Fewest training matches worth fitting on. Below this even the league mean is a
# guess (it only bites at the start of the first season with corner counts).
MIN_TRAINING_MATCHES = 50

GOAL_LINES = (0.5, 1.5, 2.5, 3.5, 4.5)
CORNER_LINES = (7.5, 8.5, 9.5, 10.5, 11.5, 12.5)
MAX_CORNERS = 30

_HISTORY = {}  # league -> (db version, training rows, {teams.id: club})
_FITS = {}  # (league, target, cutoff, training rows) -> fitted UCLModel
_MAX_FITS = 2048


def league_of(competition_id):
    """The fitted-model league a competition belongs to, or None if it uses the
    ratio ratings."""
    return (COMPETITIONS.get(competition_id) or {}).get("league")


def _history(league):
    """(training rows, {teams.id: club}) for the league, reread when the database changes."""
    version = db_version()
    cached = _HISTORY.get(league)
    if cached is None or cached[0] != version:
        cached = _HISTORY[league] = (version, _read_history(league), _read_clubs())
    return cached[1], cached[2]


def _read_clubs():
    with get_db() as db:
        return {row["id"]: row["club"] for row in db.execute(
            "SELECT id, COALESCE(sportradar_id, 'local:' || id) AS club FROM teams")}


def _read_history(league):
    """Completed matches of every season in the league, as UCLModel training rows."""
    seasons = [c for c, cfg in COMPETITIONS.items() if cfg.get("league") == league]
    marks = ",".join("?" for _ in seasons)
    with get_db() as db:
        matches = db.execute(
            f"""
            SELECT m.id, m.date, m.home_score, m.away_score,
                   COALESCE(h.sportradar_id, 'local:' || h.id) AS home_club,
                   COALESCE(a.sportradar_id, 'local:' || a.id) AS away_club
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition_id IN ({marks}) AND m.status IN ('FT', 'AET', 'PEN')
                  AND m.home_score IS NOT NULL AND m.away_score IS NOT NULL
            """,
            seasons,
        ).fetchall()
        corners = {}
        for row in db.execute(
            f"""
            SELECT s.match_id, COALESCE(t.sportradar_id, 'local:' || s.team_id) AS club, s.corners
            FROM match_stats s
            JOIN matches m ON m.id = s.match_id
            LEFT JOIN teams t ON t.id = s.team_id
            WHERE m.competition_id IN ({marks}) AND s.corners IS NOT NULL
            """,
            seasons,
        ):
            corners.setdefault(row["match_id"], []).append((row["club"], row["corners"]))

    rows = []
    for m in matches:
        # Corners count only with exactly one row for each of the two clubs.
        stats = corners.get(m["id"], ())
        counts = dict(stats)
        has_corners = len(stats) == 2 and set(counts) == {m["home_club"], m["away_club"]}
        rows.append({
            "competition": league, "date": m["date"], "status": "FT", "neutral": False,
            "home_id": m["home_club"], "away_id": m["away_club"],
            "home_goals": m["home_score"], "away_goals": m["away_score"],
            "home_corners": counts[m["home_club"]] if has_corners else None,
            "away_corners": counts[m["away_club"]] if has_corners else None,
        })
    return rows


def _fit(league, target, history, as_of_date):
    """The model fitted on matches strictly before as_of_date (or on everything
    played so far when it is None). None if there is too little to fit on yet."""
    # Live predictions: everything played, aged to the end of today (UTC).
    cutoff = as_of_date or datetime.now(timezone.utc).strftime("%Y-%m-%dT23:59:59+00:00")
    column = f"home_{target}"
    train = [m for m in history if m["date"] < cutoff and m[column] is not None]
    if len(train) < MIN_TRAINING_MATCHES:
        return None

    key = (league, target, cutoff, len(train))
    model = _FITS.get(key)
    if model is None:
        leagues = {m[side]: league for m in train for side in ("home_id", "away_id")}
        model = UCLModel(*FIT_PARAMS[target], target=target).fit(train, cutoff, leagues)
        if len(_FITS) >= _MAX_FITS:
            _FITS.clear()
        _FITS[key] = model
    return model


def _rates(competition_id, home_team_id, away_team_id, target, as_of_date):
    league = league_of(competition_id)
    history, clubs = _history(league)
    model = _fit(league, target, history, as_of_date)
    if model is None:
        return None
    # A club the model has not seen (newly promoted) is rated as league average.
    lam_h, lam_a = model.expected_goals(clubs.get(home_team_id), clubs.get(away_team_id), competition=league)
    return model, lam_h, lam_a


def predict_goals(competition_id, home_team_id, away_team_id, as_of_date=None):
    """Expected goals and the goal markets, in the shape models.poisson.simulate_match
    returns (probabilities as percentages). None when there is too little history."""
    fitted = _rates(competition_id, home_team_id, away_team_id, "goals", as_of_date)
    if fitted is None:
        return None
    model, lam_h, lam_a = fitted
    grid = model.score_grid(lam_h, lam_a)
    i, j = np.indices(grid.shape)
    top = np.argsort(grid, axis=None)[::-1][:10]
    score_probs = {(int(h), int(a)): float(grid[h, a]) * 100 for h, a in zip(*np.unravel_index(top, grid.shape))}
    best = next(iter(score_probs))
    return {
        "home_xg": lam_h,
        "away_xg": lam_a,
        "results": {
            "home_win_prob": float(grid[i > j].sum()) * 100,
            "draw_prob": float(grid[i == j].sum()) * 100,
            "away_win_prob": float(grid[i < j].sum()) * 100,
            "btts_prob": float(grid[(i > 0) & (j > 0)].sum()) * 100,
            "most_likely_score": f"{best[0]}-{best[1]}",
            "score_probs": score_probs,
            "total_probs": {},
            "total_expected": lam_h + lam_a,
            "over_lines": {line: float(grid[i + j > line].sum()) * 100 for line in GOAL_LINES},
        },
    }


def predict_corners(competition_id, home_team_id, away_team_id, as_of_date=None):
    """Expected corners and the total-corners markets, in the shape
    models.poisson.simulate_match returns. None when there is too little corner history."""
    fitted = _rates(competition_id, home_team_id, away_team_id, "corners", as_of_date)
    if fitted is None:
        return None
    _, lam_h, lam_a = fitted
    mean = lam_h + lam_a
    if CORNER_DISPERSION > 1.0:
        # scipy's NB(n, p): mean n(1-p)/p, variance mean/p.
        dist = nbinom(mean / (CORNER_DISPERSION - 1.0), 1.0 / CORNER_DISPERSION)
    else:
        dist = poisson(mean)
    pmf = dist.pmf(np.arange(MAX_CORNERS + 1))
    top = np.argsort(pmf)[::-1][:10]
    return {
        "home_expected": lam_h,
        "away_expected": lam_a,
        "results": {
            "home_win_prob": 0.0,
            "draw_prob": 0.0,
            "away_win_prob": 0.0,
            "btts_prob": 0.0,
            "most_likely_score": str(int(top[0])),
            "score_probs": {},
            "total_probs": {int(t): float(pmf[t]) * 100 for t in top},
            "total_expected": mean,
            "over_lines": {line: float(dist.sf(int(line))) * 100 for line in CORNER_LINES},
        },
    }
