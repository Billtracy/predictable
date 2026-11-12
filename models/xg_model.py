import math
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db

# SportRadar coordinates are treated as 0-100 percentages, with X=100 as the
# opponent goal line and Y=50 as the goal center.
PITCH_LENGTH = 100
PITCH_WIDTH = 100
GOAL_Y_CENTER = 50
SHOT_EVENT_TYPES = (
    "shot_off_target",
    "shot_on_target",
    "shot_saved",
    "shot_blocked",
    "goal",
    "score_change",
)
DEFAULT_SHOT_MODEL = object()
_XG_MODEL_CACHE = {"loaded": False, "model": None}


def calculate_distance(x, y):
    """Calculate distance to the center of the goal."""
    return math.sqrt((100 - x) ** 2 + (GOAL_Y_CENTER - y) ** 2)


def calculate_angle(x, y):
    """Calculate the visible angle of the goal from the shot position."""
    post1_y = 44.6
    post2_y = 55.4

    d1 = math.sqrt((100 - x) ** 2 + (post1_y - y) ** 2)
    d2 = math.sqrt((100 - x) ** 2 + (post2_y - y) ** 2)
    goal_width = post2_y - post1_y

    try:
        cos_angle = (d1**2 + d2**2 - goal_width**2) / (2 * d1 * d2)
        cos_angle = max(-1.0, min(1.0, cos_angle))
        return math.acos(cos_angle)
    except (ValueError, ZeroDivisionError):
        return 0


def _load_shots(db, as_of_date, *, team_id=None, competition_id=None):
    """Load coordinate shots with each team's attack oriented toward X=100."""
    event_types = ", ".join("?" for _ in SHOT_EVENT_TYPES)
    filters = [f"me.event_type IN ({event_types})"]
    params = list(SHOT_EVENT_TYPES)
    join = ""
    if competition_id is not None or as_of_date is not None:
        join = "JOIN matches m ON m.id = me.match_id"
    if team_id is not None:
        filters.append("me.team_id = ?")
        params.append(team_id)
    if competition_id is not None:
        filters.append("m.competition_id = ?")
        params.append(competition_id)
    if as_of_date is not None:
        filters.extend(("m.status IN ('FT', 'AET', 'PEN')", "m.date < ?"))
        params.append(as_of_date)

    # Preserve the direction heuristic: average X across the team's attempts in
    # the match, including those without a Y coordinate and excluding "goal".
    return db.execute(
        f"""
        WITH shots AS (
            SELECT me.match_id, me.event_type, me.x_coord, me.y_coord,
                   AVG(CASE WHEN me.event_type != 'goal' THEN me.x_coord END)
                       OVER (PARTITION BY me.match_id, me.team_id) AS avg_x
            FROM match_events me
            {join}
            WHERE {" AND ".join(filters)}
        )
        SELECT match_id, event_type, y_coord,
               CASE WHEN avg_x < 50 THEN 100 - x_coord ELSE x_coord END AS x_coord
        FROM shots
        WHERE x_coord IS NOT NULL AND y_coord IS NOT NULL
        """,
        params,
    ).fetchall()


def _shot_features(x, y):
    return [calculate_distance(x, y), calculate_angle(x, y)]


def train_xg_model(as_of_date=None):
    """
    Train a logistic regression model on historical shot data from the database.
    With a cutoff, use only completed matches dated strictly before it.
    Return None to select the baseline curve when training data is insufficient.
    """
    try:
        from sklearn.linear_model import LogisticRegression

        with get_db() as db:
            shots = _load_shots(db, as_of_date)

        if len(shots) < 100:
            print(f"Not enough shots for training ({len(shots)}). Using baseline model.")
            return None

        features = [_shot_features(shot["x_coord"], shot["y_coord"]) for shot in shots]
        labels = [int(shot["event_type"] in ("goal", "score_change")) for shot in shots]

        if len(set(labels)) < 2:
            print("Shot data has only one outcome class. Using baseline model.")
            return None

        model = LogisticRegression()
        model.fit(features, labels)
        print(f"Trained xG model on {len(features)} shots. Goals: {sum(labels)}, Non-goals: {len(labels) - sum(labels)}")
        print(f"Coefficients: intercept={model.intercept_[0]:.3f}, distance={model.coef_[0][0]:.4f}, angle={model.coef_[0][1]:.3f}")
        return model

    except ImportError:
        print("scikit-learn not installed. Using baseline lookup model.")
        return None
    except Exception as exc:
        print(f"Error training model: {exc}")
        import traceback
        traceback.print_exc()
        return None


# Baseline model coefficients: P(Goal) = 1 / (1 + e^-(B0 + B1*Distance + B2*Angle))
B0 = 1.2
B1 = -0.15
B2 = 2.5


def get_xg(x, y, model=None):
    """Get the expected goals value for a shot at coordinates (x, y)."""
    features = _shot_features(x, y)
    if model is not None:
        return model.predict_proba([features])[0][1]
    return _baseline_xg(*features)


def _baseline_xg(distance, angle):
    exponent = -(B0 + (B1 * distance) + (B2 * angle))
    exponent = max(-50, min(50, exponent))
    return 1 / (1 + math.exp(exponent))


def get_cached_xg_model():
    """Train or retrieve the in-process xG model cache."""
    if not _XG_MODEL_CACHE["loaded"]:
        _XG_MODEL_CACHE["model"] = train_xg_model()
        _XG_MODEL_CACHE["loaded"] = True
    return _XG_MODEL_CACHE["model"]


def get_team_event_xg(team_id, competition_id, model=None, as_of_date=None):
    """Return average xG per match from mapped SportRadar shot coordinates.
    With a cutoff, count only completed matches dated strictly before it."""
    with get_db() as db:
        shots = _load_shots(db, as_of_date, team_id=team_id, competition_id=competition_id)

    if not shots:
        return None
    covered_matches = len({shot["match_id"] for shot in shots})
    features = [_shot_features(shot["x_coord"], shot["y_coord"]) for shot in shots]
    if model is None:
        total_xg = sum(_baseline_xg(*feature) for feature in features)
    else:
        total_xg = sum(row[1] for row in model.predict_proba(features))
    return total_xg / covered_matches


def get_team_proxy_xg(team_id, competition_id, as_of_date=None, *, shot_model=DEFAULT_SHOT_MODEL):
    """
    Prefer event-coordinate xG when available; otherwise estimate from
    API-Sports aggregate shots-on-target data. With a cutoff, count only
    completed matches dated strictly before it. Historical replays should pass
    a model fitted before the cutoff, or None for the baseline curve.
    """
    if shot_model is DEFAULT_SHOT_MODEL:
        shot_model = get_cached_xg_model()
    event_xg = get_team_event_xg(team_id, competition_id, shot_model, as_of_date)
    if event_xg is not None:
        return event_xg

    date_clause = ""
    params = [team_id, competition_id]
    if as_of_date is not None:
        date_clause = "AND m.status IN ('FT', 'AET', 'PEN') AND m.date < ?"
        params.append(as_of_date)
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            f"""
            SELECT AVG(ms.shots_on_target)
            FROM match_stats ms
            JOIN matches m ON ms.match_id = m.id
            WHERE ms.team_id = ? AND m.competition_id = ? {date_clause}
            """,
            params,
        )
        row = cur.fetchone()

    if not row or row[0] is None:
        return 1.0

    # A generic proxy: 1 shot on target is roughly 0.3 xG in professional football.
    return row[0] * 0.3
