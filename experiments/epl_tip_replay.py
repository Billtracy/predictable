"""Replay the EPL site's single-fixture tip rules on completed fixtures."""

from collections import defaultdict
from datetime import datetime, timezone

from db.database import get_db
from models.team_ratings import build_context
from models.xg_model import train_xg_model
from predictors.corners import corner_tip, matchup_corner_rates
from predictors.goals import goal_tip, matchup_goal_rates


FINISHED = ("FT", "AET", "PEN")
MATCHES_PER_GAMEWEEK = 10


def kickoff_day(date):
    """Return the UTC midnight cutoff used for a fixture's replay and baselines."""
    if date.endswith("Z"):
        date = date[:-1] + "+00:00"
    kickoff = datetime.fromisoformat(date)
    if kickoff.tzinfo is None:
        kickoff = kickoff.replace(tzinfo=timezone.utc)
    return kickoff.astimezone(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()


def _read_season(season):
    with get_db() as db:
        matches = db.execute(
            """
            SELECT m.id, m.date, m.status, m.home_score, m.away_score,
                   h.id AS home_id, h.name AS home_name, h.sportradar_id AS home_provider_id,
                   a.id AS away_id, a.name AS away_name, a.sportradar_id AS away_provider_id
            FROM matches m
            JOIN teams h ON h.id = m.home_team_id
            JOIN teams a ON a.id = m.away_team_id
            WHERE m.competition_id = ?
            ORDER BY m.date, m.id
            """,
            (season,),
        ).fetchall()
        corners = defaultdict(list)
        for row in db.execute(
            """
            SELECT s.match_id, s.team_id, t.sportradar_id, s.corners
            FROM match_stats s
            JOIN matches m ON m.id = s.match_id
            LEFT JOIN teams t ON t.id = s.team_id
            WHERE m.competition_id = ? AND s.corners IS NOT NULL
            """,
            (season,),
        ):
            corners[row["match_id"]].append(row)
    return matches, corners


def _valid_count(value):
    return isinstance(value, int) and value >= 0


def _corner_counts(match, rows):
    """Grade corners only with one valid count for each team in the fixture."""
    if len(rows) != 2:
        return None
    sides = {}
    for row in rows:
        candidates = [side for side in ("home", "away") if row["team_id"] == match[f"{side}_id"]]
        if not candidates and row["sportradar_id"]:
            candidates = [
                side for side in ("home", "away")
                if row["sportradar_id"] == match[f"{side}_provider_id"]
            ]
        if len(candidates) != 1 or candidates[0] in sides or not _valid_count(row["corners"]):
            return None
        sides[candidates[0]] = row["corners"]
    return sides["home"], sides["away"]


def _market(counts):
    return {"actual": None if counts is None else sum(counts), "counts": counts, "pick": None, "rates": None}


def _completed_fixtures(season, matches, corners):
    # The EPL page chunks its full date-ordered schedule into groups of ten.
    # Keep the same slots, including unplayed rows, and break kickoff ties by ID.
    for index, match in enumerate(matches):
        if match["status"] not in FINISHED or not match["date"]:
            continue
        home, away = match["home_score"], match["away_score"]
        goal_counts = (home, away) if _valid_count(home) and _valid_count(away) else None
        yield match, {
            "season": season,
            "round": index // MATCHES_PER_GAMEWEEK + 1,
            "match_id": match["id"],
            "date": match["date"],
            "home_club": match["home_provider_id"] or f"local:{match['home_id']}",
            "away_club": match["away_provider_id"] or f"local:{match['away_id']}",
            "goals": _market(goal_counts),
            "corners": _market(_corner_counts(match, corners.get(match["id"], ()))),
        }


def _add_forecasts(fixture, match, context, strategy):
    if not context or not context["home_ratings"] or not context["away_ratings"]:
        return
    # The match card is only published when both markets can be priced.
    if not context["avgs"]["avg_corners_per_team"]:
        return

    goal_rates = matchup_goal_rates(context)
    fixture["goals"].update(pick=goal_tip(sum(goal_rates)), rates=goal_rates)
    corner_rates = matchup_corner_rates(context)
    pick = corner_tip(strategy, sum(corner_rates), match["home_name"], match["away_name"])
    fixture["corners"].update(pick=pick, rates=corner_rates)


def load_epl_history(seasons):
    """Yield completed fixtures with outcome labels for historical baselines."""
    for season in seasons:
        matches, corners = _read_season(season)
        for _, fixture in _completed_fixtures(season, matches, corners):
            yield fixture


def load_epl_fixtures(seasons, strategy):
    """Yield completed fixtures with eligible forecasts and outcome labels.

    Fit the shot model once before each season and compute one context per fixture
    using only matches before the kickoff day.
    """
    for season in seasons:
        matches, corners = _read_season(season)
        start_date = next((match["date"] for match in matches if match["date"]), None)
        if start_date is None:
            continue
        shot_model = train_xg_model(as_of_date=kickoff_day(start_date))
        for match, fixture in _completed_fixtures(season, matches, corners):
            context = build_context(
                season,
                match["home_id"],
                match["away_id"],
                as_of_date=kickoff_day(match["date"]),
                shot_model=shot_model,
            )
            _add_forecasts(fixture, match, context, strategy)
            yield fixture
