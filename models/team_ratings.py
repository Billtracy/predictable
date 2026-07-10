import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from config import COMPETITIONS
from models.xg_model import get_team_proxy_xg

# Ratings are ratios centered on 1.0 (league average). Clamp keeps small samples sane.
RATING_KEYS = ("attack_goals", "defense_goals", "attack_corners", "defense_corners")

# Prior strength for shrinkage: how many "league-average" pseudo-matches to blend in.
# A team is regressed toward 1.0 until it has accumulated real matches. See _shrink().
SHRINKAGE_K = 4.0

# Max weight of the shots-conceded volume estimate inside the goals defense
# rating (mirrors the 60% proxy-xG share on the attack side). Scaled down by
# shot-stats coverage; see _compute_one. Per the seeded backtest A/B, this blend
# cut WC-2022 log loss ~0.04 and improved every binary goal-market Brier.
DEFENSE_VOLUME_WEIGHT = 0.6

# Cross-season prior cache: {prior_competition_id: {team_id: {"attack","defense"}}}.
# The prior is a full previous season, entirely before the current one, so it is
# leak-free and can be computed once and reused across all point-in-time calls.
_PRIOR_CACHE = {}
# Prior-season league-average cache: {prior_competition_id: avgs dict}. Used as the
# cold-start fallback for a season with no completed matches yet (e.g. GW1).
_PRIOR_AVG_CACHE = {}


def _shrink(rating, n_matches, k=SHRINKAGE_K, target=1.0):
    """Regress a ratio-rating toward `target` (default the league mean 1.0) based on
    sample size. With n=0 the result is exactly `target`; as n grows it approaches the
    raw rating. When a cross-season prior is available it is passed as `target`, so a
    team starts the season at last year's strength and converges to this year's form.
    """
    if n_matches <= 0:
        return target
    return (n_matches * rating + k * target) / (n_matches + k)


def _clamp(ratings):
    for key in RATING_KEYS:
        ratings[key] = max(0.1, min(3.0, ratings[key]))
    return ratings


def _fetch_averages(cur, competition_id, as_of_date=None):
    """Global per-team averages for the competition, optionally restricted to
    matches strictly BEFORE as_of_date (point-in-time, leak-free)."""
    date_clause = "AND m.date < ?" if as_of_date else ""

    params = [competition_id] + ([as_of_date] if as_of_date else [])
    cur.execute(
        f"""
        SELECT AVG(home_score + away_score) as avg_goals,
               AVG(home_score) as avg_home_goals,
               AVG(away_score) as avg_away_goals,
               COUNT(*) as n_matches
        FROM matches m
        WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        params,
    )
    row = cur.fetchone()
    if not row or row["avg_goals"] is None:
        return None

    cur.execute(
        f"""
        SELECT AVG(ms.corners) * 2 as avg_corners
        FROM matches m
        JOIN match_stats ms ON m.id = ms.match_id
        WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        params,
    )
    crow = cur.fetchone()
    avg_corners_total = crow["avg_corners"] if crow and crow["avg_corners"] else None

    cur.execute(
        f"""
        SELECT AVG(ms.shots_total) as avg_shots
        FROM matches m
        JOIN match_stats ms ON m.id = ms.match_id
        WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
              AND ms.shots_total IS NOT NULL {date_clause}
        """,
        params,
    )
    srow = cur.fetchone()
    avg_shots_per_team = srow["avg_shots"] if srow and srow["avg_shots"] else None

    return {
        "avg_goals_per_team": row["avg_goals"] / 2,
        "avg_corners_per_team": (avg_corners_total / 2) if avg_corners_total else 0,
        "avg_shots_per_team": avg_shots_per_team,
        "avg_home_goals": row["avg_home_goals"] or 0,
        "avg_away_goals": row["avg_away_goals"] or 0,
        "n_matches": row["n_matches"] or 0,
    }


def _prior_averages(cur, prior_competition_id):
    """Full prior-season league averages, cached. Fixed data, so compute once."""
    if prior_competition_id not in _PRIOR_AVG_CACHE:
        _PRIOR_AVG_CACHE[prior_competition_id] = _fetch_averages(cur, prior_competition_id)
    return _PRIOR_AVG_CACHE[prior_competition_id]


def _averages_with_fallback(cur, competition_id, as_of_date=None):
    """League averages for the competition, falling back to the prior season when the
    current one has no completed matches yet. This is what makes GW1 of a new season
    (and the very first fixture in a point-in-time backtest) predictable: team ratings
    already fall back to the cross-season prior, and this supplies the matching league
    averages + home advantage until real results accumulate.
    """
    avgs = _fetch_averages(cur, competition_id, as_of_date)
    if avgs is not None:
        return avgs
    prior_comp = (COMPETITIONS.get(competition_id) or {}).get("prior_competition")
    return _prior_averages(cur, prior_comp) if prior_comp else None


def _shots_per_match(cur, team_id, competition_id, conceded=False, as_of_date=None):
    """Average shots per match taken by (or, with conceded=True, faced by) a team,
    counting only matches that have shot stats. Returns (avg, n_matches_with_stats),
    or (None, 0) when no shot data exists for the team yet."""
    date_clause = "AND m.date < ?" if as_of_date else ""
    date_arg = [as_of_date] if as_of_date else []
    team_clause = "ms.team_id != ?" if conceded else "ms.team_id = ?"

    cur.execute(
        f"""
        SELECT AVG(ms.shots_total) as avg_shots, COUNT(*) as n
        FROM match_stats ms
        JOIN matches m ON ms.match_id = m.id
        WHERE m.competition_id = ? AND (m.home_team_id = ? OR m.away_team_id = ?)
              AND {team_clause} AND ms.shots_total IS NOT NULL
              AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        [competition_id, team_id, team_id, team_id] + date_arg,
    )
    row = cur.fetchone()
    if not row or row["avg_shots"] is None:
        return None, 0
    return row["avg_shots"], row["n"]


def _prior_goal_ratings(cur, prior_competition_id, diff_weight):
    """Goals-based attack/defense ratings for every team in the prior season, cached.

    Uses the full prior season (no as_of_date): it is entirely before the current
    season, so there is no look-ahead leak. Returns {team_id: {"attack","defense"}}
    on the same diff-weighted, league-relative scale as the live ratings, so it can
    be used directly as a shrinkage target.
    """
    if prior_competition_id in _PRIOR_CACHE:
        return _PRIOR_CACHE[prior_competition_id]

    cur.execute(
        """
        SELECT AVG(home_score + away_score) / 2.0 AS avg
        FROM matches
        WHERE competition_id = ? AND status IN ('FT', 'AET', 'PEN')
        """,
        (prior_competition_id,),
    )
    row = cur.fetchone()
    avg = row["avg"] if row else None

    ratings = {}
    if avg:
        cur.execute(
            """
            SELECT t.id AS team_id,
                   SUM(CASE WHEN m.home_team_id = t.id THEN m.home_score ELSE m.away_score END) AS gf,
                   SUM(CASE WHEN m.home_team_id = t.id THEN m.away_score ELSE m.home_score END) AS ga,
                   COUNT(*) AS n
            FROM matches m
            JOIN teams t ON t.id IN (m.home_team_id, m.away_team_id)
            WHERE m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN')
            GROUP BY t.id
            """,
            (prior_competition_id,),
        )
        for r in cur.fetchall():
            n = r["n"] or 0
            if n == 0:
                continue
            attack = (r["gf"] / n) / avg * diff_weight
            defense = (r["ga"] / n) / avg * diff_weight
            # Light shrink (full season is ~38 games, so this barely moves it) to
            # keep an extreme season from dominating the following year's start.
            ratings[r["team_id"]] = {
                "attack": _shrink(attack, n),
                "defense": _shrink(defense, n),
            }

    _PRIOR_CACHE[prior_competition_id] = ratings
    return ratings


def _compute_one(cur, team_id, competition_id, avgs, diff_weight, as_of_date=None):
    """Compute attack/defense ratings for a single team using only matches
    strictly before as_of_date (when given)."""
    date_clause = "AND m.date < ?" if as_of_date else ""
    date_arg = [as_of_date] if as_of_date else []

    # Goals scored / conceded
    cur.execute(
        f"""
        SELECT COUNT(m.id) as matches,
               SUM(CASE WHEN m.home_team_id = ? THEN m.home_score ELSE m.away_score END) as goals_for,
               SUM(CASE WHEN m.home_team_id = ? THEN m.away_score ELSE m.home_score END) as goals_against
        FROM matches m
        WHERE m.competition_id = ? AND (m.home_team_id = ? OR m.away_team_id = ?)
              AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        [team_id, team_id, competition_id, team_id, team_id] + date_arg,
    )
    g = cur.fetchone()
    n = g["matches"] or 0

    # Cross-season prior (goals only): last season's rating for this team, if any.
    prior_comp = COMPETITIONS.get(competition_id, {}).get("prior_competition")
    prior = _prior_goal_ratings(cur, prior_comp, diff_weight).get(team_id) if prior_comp else None
    atk_target = prior["attack"] if prior else 1.0
    def_target = prior["defense"] if prior else 1.0

    # No current-season data yet -> start from the cross-season prior (or neutral for
    # promoted teams with no prior-season rating). This is the cold-start fix.
    if n == 0:
        return _clamp({
            "attack_goals": atk_target,
            "defense_goals": def_target,
            "attack_corners": 1.0,
            "defense_corners": 1.0,
            "n_matches": 0,
        })

    goals_for = g["goals_for"] or 0
    goals_against = g["goals_against"] or 0

    # Corners won by this team (only matches that actually have stats rows)
    cur.execute(
        f"""
        SELECT SUM(ms.corners) as corners_for, COUNT(ms.corners) as corner_matches
        FROM match_stats ms
        JOIN matches m ON ms.match_id = m.id
        WHERE ms.team_id = ? AND m.competition_id = ? AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        [team_id, competition_id] + date_arg,
    )
    cf = cur.fetchone()
    corners_for = cf["corners_for"] or 0
    corner_matches = cf["corner_matches"] or 0

    # Total corners in this team's matches -> conceded = total - for
    cur.execute(
        f"""
        SELECT SUM(ms.corners) as total_corners
        FROM match_stats ms
        JOIN matches m ON ms.match_id = m.id
        WHERE m.competition_id = ? AND (m.home_team_id = ? OR m.away_team_id = ?)
              AND m.status IN ('FT', 'AET', 'PEN') {date_clause}
        """,
        [competition_id, team_id, team_id] + date_arg,
    )
    total_corners = cur.fetchone()["total_corners"] or 0
    corners_against = total_corners - corners_for

    proxy_xg = get_team_proxy_xg(team_id, competition_id, as_of_date)

    avg_goals = avgs["avg_goals_per_team"]
    avg_corners = avgs["avg_corners_per_team"]
    avg_shots = avgs.get("avg_shots_per_team")

    # Attack (goals): blend actual goals (40%) with proxy xG (60%) relative to league avg
    attack_metric = (0.4 * (goals_for / n)) + (0.6 * proxy_xg)
    attack_goals = (attack_metric / avg_goals) * diff_weight if avg_goals > 0 else 1.0

    # Defense (goals): raw goals-against is the noisiest signal in the model
    # (~1.3 goals/match sample). Where shot stats exist, blend in expected goals
    # against from shots conceded x league conversion rate — shot volume is far
    # more stable match-to-match than whether those shots happened to go in.
    defense_metric = goals_against / n
    shots_against_pm, shot_matches = _shots_per_match(
        cur, team_id, competition_id, conceded=True, as_of_date=as_of_date
    )
    if shots_against_pm is not None and avg_shots and avg_goals > 0:
        conversion = avg_goals / avg_shots  # league-average goals per shot
        xga_volume = shots_against_pm * conversion
        # Weight the volume-based estimate by how much of the sample it covers.
        w = DEFENSE_VOLUME_WEIGHT * min(1.0, shot_matches / n)
        defense_metric = (1 - w) * defense_metric + w * xga_volume
    defense_goals = (defense_metric / avg_goals) * diff_weight if avg_goals > 0 else 1.0

    cn = corner_matches if corner_matches > 0 else n
    attack_corners = ((corners_for / cn) / avg_corners) * diff_weight if avg_corners > 0 else 1.0
    defense_corners = ((corners_against / cn) / avg_corners) * diff_weight if avg_corners > 0 else 1.0

    # Regress toward league average based on how many matches back each rating.
    ratings = {
        "attack_goals": _shrink(attack_goals, n, target=atk_target),
        "defense_goals": _shrink(defense_goals, n, target=def_target),
        "attack_corners": _shrink(attack_corners, cn),
        "defense_corners": _shrink(defense_corners, cn),
        "n_matches": n,
    }
    return _clamp(ratings)


def _read_cached(cur, team_id, competition_id):
    cur.execute(
        """
        SELECT attack_goals, defense_goals, attack_corners, defense_corners
        FROM team_ratings
        WHERE team_id = ? AND competition_id = ?
        """,
        (team_id, competition_id),
    )
    row = cur.fetchone()
    return dict(row) if row else None


def build_context(competition_id, home_team_id, away_team_id, as_of_date=None):
    """Assemble everything a predictor needs for one matchup.

    When as_of_date is given, ratings + averages are recomputed point-in-time from
    matches strictly before that date (used by the backtest to avoid look-ahead
    leakage). Otherwise the pre-computed cached ratings are used (live prediction).

    Returns a dict with home_ratings, away_ratings, avgs, and home_advantage/away_advantage.
    """
    comp_config = COMPETITIONS.get(competition_id)
    diff_weight = comp_config.get("difficulty_weight", 1.0) if comp_config else 1.0

    with get_db() as db:
        cur = db.cursor()
        avgs = _averages_with_fallback(cur, competition_id, as_of_date)
        if avgs is None:
            return None

        if as_of_date is not None:
            home = _compute_one(cur, home_team_id, competition_id, avgs, diff_weight, as_of_date)
            away = _compute_one(cur, away_team_id, competition_id, avgs, diff_weight, as_of_date)
        else:
            home = _read_cached(cur, home_team_id, competition_id)
            away = _read_cached(cur, away_team_id, competition_id)

    home_adv, away_adv = fit_home_advantage(competition_id, avgs)

    return {
        "home_ratings": home,
        "away_ratings": away,
        "avgs": avgs,
        "home_advantage": home_adv,
        "away_advantage": away_adv,
    }


def fit_home_advantage(competition_id, avgs):
    """Estimate the home/away scoring multipliers from the data itself instead of
    hardcoding them. On neutral ground (World Cup) the split is ~50/50 so both
    factors collapse to ~1.0 naturally; in league play a real home edge emerges.

    Returns (home_advantage, away_advantage) as multipliers on the base expected
    goals. Falls back to 1.0/1.0 when there isn't enough data to trust the split.
    """
    avg_goals_per_team = avgs.get("avg_goals_per_team", 0)
    if not avg_goals_per_team or avgs.get("n_matches", 0) < 10:
        return 1.0, 1.0

    home_adv = avgs["avg_home_goals"] / avg_goals_per_team
    away_adv = avgs["avg_away_goals"] / avg_goals_per_team

    # Guard against noise: keep multipliers in a sane band.
    home_adv = max(0.7, min(1.5, home_adv))
    away_adv = max(0.7, min(1.5, away_adv))
    return home_adv, away_adv


def compute_team_ratings(competition_id):
    """Compute and cache full-data ratings for every team in a competition.
    Used by the fetch pipeline for live predictions."""
    comp_config = COMPETITIONS.get(competition_id)
    if not comp_config:
        print(f"Unknown competition: {competition_id}")
        return

    diff_weight = comp_config.get("difficulty_weight", 1.0)

    with get_db() as db:
        cur = db.cursor()

        avgs = _averages_with_fallback(cur, competition_id)
        if avgs is None:
            print(f"No match data found for {competition_id} to compute averages.")
            return

        cur.execute(
            """
            SELECT DISTINCT home_team_id FROM matches WHERE competition_id = ?
            UNION
            SELECT DISTINCT away_team_id FROM matches WHERE competition_id = ?
            """,
            (competition_id, competition_id),
        )
        teams = [r[0] for r in cur.fetchall()]

        print(f"Computing ratings for {len(teams)} teams in {competition_id}...")

        for team_id in teams:
            r = _compute_one(cur, team_id, competition_id, avgs, diff_weight)
            db.execute(
                """
                INSERT OR REPLACE INTO team_ratings
                (team_id, competition_id, attack_goals, defense_goals, attack_corners, defense_corners, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    team_id,
                    competition_id,
                    r["attack_goals"],
                    r["defense_goals"],
                    r["attack_corners"],
                    r["defense_corners"],
                ),
            )

        print(f"Successfully computed and cached ratings for {competition_id}.")


def get_team_rating(team_id, competition_id):
    """Retrieve cached ratings for a team."""
    with get_db() as db:
        cur = db.cursor()
        return _read_cached(cur, team_id, competition_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        compute_team_ratings(sys.argv[1])
    else:
        print("Usage: python models/team_ratings.py <competition_id>")
