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


def _normalize_x(x, team_id, match_id, db):
    """Normalize X coordinate so all shots face toward X=100 (opponent's goal).
    SportRadar uses absolute pitch coords where each team attacks a different end.
    We detect the team's attacking direction from the average X of their shots in that match."""
    cur = db.cursor()
    cur.execute("""
        SELECT AVG(x_coord) as avg_x FROM match_events
        WHERE match_id = ? AND team_id = ?
        AND event_type IN ('shot_off_target', 'shot_on_target', 'shot_saved', 'shot_blocked', 'score_change')
    """, (match_id, team_id))
    row = cur.fetchone()
    avg_x = row["avg_x"] if row and row["avg_x"] is not None else 50
    
    # If average shot X < 50, the team is attacking toward X=0, so flip
    if avg_x < 50:
        return 100 - x
    return x


def train_xg_model():
    """
    Train a logistic regression model on historical shot data from the database.
    Falls back to the baseline curve when there is not enough event data.
    """
    try:
        from sklearn.linear_model import LogisticRegression

        with get_db() as db:
            cur = db.cursor()
            cur.execute(
                """
                SELECT me.x_coord, me.y_coord, me.event_type, me.outcome, me.team_id, me.match_id
                FROM match_events me
                WHERE me.event_type IN (
                    'shot_off_target',
                    'shot_on_target',
                    'shot_saved',
                    'shot_blocked',
                    'goal',
                    'score_change'
                )
                AND me.x_coord IS NOT NULL AND me.y_coord IS NOT NULL
                """
            )
            shots = cur.fetchall()

            if len(shots) < 100:
                print(f"Not enough shots for training ({len(shots)}). Using baseline model.")
                return None

            features = []
            labels = []
            for shot in shots:
                x = shot["x_coord"]
                y = shot["y_coord"]
                if x is None or y is None:
                    continue

                # Normalize X so all shots face toward X=100
                x = _normalize_x(x, shot["team_id"], shot["match_id"], db)

                dist = calculate_distance(x, y)
                angle = calculate_angle(x, y)
                is_goal = 1 if shot["event_type"] in ("goal", "score_change") else 0

                features.append([dist, angle])
                labels.append(is_goal)

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
    dist = calculate_distance(x, y)
    angle = calculate_angle(x, y)

    if model:
        return model.predict_proba([[dist, angle]])[0][1]

    exponent = -(B0 + (B1 * dist) + (B2 * angle))
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
    When as_of_date is given, only matches strictly before it are counted."""
    date_clause = "AND m.date < ?" if as_of_date else ""
    params = [team_id, competition_id] + ([as_of_date] if as_of_date else [])
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            f"""
            SELECT me.match_id, me.x_coord, me.y_coord, me.team_id
            FROM match_events me
            JOIN matches m ON me.match_id = m.id
            WHERE me.team_id = ?
              AND m.competition_id = ?
              AND me.event_type IN (
                    'shot_off_target',
                    'shot_on_target',
                    'shot_saved',
                    'shot_blocked',
                    'goal',
                    'score_change'
              )
              {date_clause}
            """,
            params,
        )
        shots = cur.fetchall()

        covered_matches = len({shot["match_id"] for shot in shots})
        if not shots or covered_matches == 0:
            return None

        total_xg = 0
        for shot in shots:
            x = _normalize_x(shot["x_coord"], shot["team_id"], shot["match_id"], db)
            total_xg += get_xg(x, shot["y_coord"], model)
        return total_xg / covered_matches


def get_team_proxy_xg(team_id, competition_id, as_of_date=None):
    """
    Prefer event-coordinate xG when available; otherwise estimate from
    API-Sports aggregate shots-on-target data. When as_of_date is given, only
    matches strictly before it contribute (point-in-time, leak-free).
    """
    model = get_cached_xg_model()
    event_xg = get_team_event_xg(team_id, competition_id, model, as_of_date)
    if event_xg is not None:
        return event_xg

    date_clause = "AND m.date < ?" if as_of_date else ""
    params = [team_id, competition_id] + ([as_of_date] if as_of_date else [])
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
