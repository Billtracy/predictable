import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db

SHOT_EVENT_TYPES = (
    "shot_off_target",
    "shot_on_target",
    "shot_saved",
    "shot_blocked",
    "goal",
    "score_change",
)


def report():
    """Per-competition data coverage: how many completed matches have stats rows
    (API-Sports) and shot-coordinate events (SportRadar). Read-only — safe to run
    anytime to check backfill progress."""
    with get_db() as db:
        cur = db.cursor()
        cur.execute(
            """
            SELECT competition_id, COUNT(*) as completed
            FROM matches
            WHERE status IN ('FT', 'AET', 'PEN')
            GROUP BY competition_id
            """
        )
        completed = {r["competition_id"]: r["completed"] for r in cur.fetchall()}

        cur.execute(
            """
            SELECT m.competition_id, COUNT(DISTINCT ms.match_id) as with_stats
            FROM match_stats ms
            JOIN matches m ON ms.match_id = m.id
            GROUP BY m.competition_id
            """
        )
        stats = {r["competition_id"]: r["with_stats"] for r in cur.fetchall()}

        placeholders = ",".join("?" * len(SHOT_EVENT_TYPES))
        cur.execute(
            f"""
            SELECT m.competition_id, COUNT(DISTINCT me.match_id) as with_events
            FROM match_events me
            JOIN matches m ON me.match_id = m.id
            WHERE me.event_type IN ({placeholders})
            GROUP BY m.competition_id
            """,
            SHOT_EVENT_TYPES,
        )
        events = {r["competition_id"]: r["with_events"] for r in cur.fetchall()}

    print(f"{'competition':<18} {'completed':>9} {'stats':>12} {'shot events':>12}")
    print("-" * 55)
    for comp in sorted(completed):
        n = completed[comp]
        s = stats.get(comp, 0)
        e = events.get(comp, 0)
        print(f"{comp:<18} {n:>9} {s:>6}/{n:<5} {e:>6}/{n:<5}")
    print("\nBackfill is complete when stats and shot events both reach the completed count.")


if __name__ == "__main__":
    report()
