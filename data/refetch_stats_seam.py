"""Remove the epl_2024 data seam.

match_stats for epl_2024 came from two providers split by date: Aug-early-Nov 2024
from API-Sports (identifiable by possession IS NOT NULL) and the rest from SportRadar
timelines (possession NULL). The two disagree by ~12-15% on shot/corner counts, a
pure provider artifact that biases point-in-time ratings.

This script re-derives the API-Sports rows from their SportRadar timelines using the
exact same tally logic as data/sportradar.fetch_sportradar_events, so every match ends
up on one consistent SportRadar basis. It overwrites (INSERT OR REPLACE) only the
seam rows; SportRadar-native rows are left alone.

Resumable and quota-aware (reuses fetch_match_timeline's 429 handling + quota banner);
aborts cleanly after 5 consecutive fetch failures.

Usage:
  python data/refetch_stats_seam.py epl_2024
  python data/refetch_stats_seam.py epl_2024 --dry-run   # list affected matches, no API calls
"""

import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.database import get_db
from data.sportradar import fetch_match_timeline, get_event_competitor_id, EVENT_TYPES


def _empty_tally():
    return {"shots": 0, "on": 0, "off": 0, "blocked": 0, "corners": 0, "goals": 0}


def _accumulate(tally, event_type):
    if event_type in ("shot_on_target", "shot_saved", "score_change"):
        tally["shots"] += 1
        tally["on"] += 1
    elif event_type == "shot_off_target":
        tally["shots"] += 1
        tally["off"] += 1
    elif event_type == "shot_blocked":
        tally["shots"] += 1
        tally["blocked"] += 1
    elif event_type == "corner_kick":
        tally["corners"] += 1
    if event_type == "score_change":
        tally["goals"] += 1


def refetch_seam(competition_id, dry_run=False):
    with get_db() as db:
        rows = db.execute(
            """
            SELECT DISTINCT m.id, m.sportradar_id, m.home_team_id, m.away_team_id,
                   m.home_score, m.away_score, m.date
            FROM matches m
            JOIN match_stats ms ON m.id = ms.match_id
            WHERE m.competition_id = ? AND ms.possession IS NOT NULL
            ORDER BY m.date
            """,
            (competition_id,),
        ).fetchall()

    print(f"Found {len(rows)} seam matches (API-Sports-sourced stats) in {competition_id}.")
    if dry_run:
        for r in rows[:10]:
            print(f"  {r['date'][:10]}  match {r['id']}  sr={r['sportradar_id']}")
        if len(rows) > 10:
            print(f"  ... and {len(rows) - 10} more")
        print("Dry run: no API calls made, nothing written.")
        return

    consecutive_failures = 0
    rewritten = 0
    for i, r in enumerate(rows):
        sr_id = r["sportradar_id"]
        if not sr_id:
            print(f"  match {r['id']}: no sportradar_id, cannot refetch. Skipping.")
            continue

        print(f"  ({i + 1}/{len(rows)}) match {r['id']} [{r['date'][:10]}] ...")
        time.sleep(1.5)
        tl = fetch_match_timeline(sr_id)
        if not tl:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                print("\n" + "!" * 60)
                print("ABORT: 5 timeline fetches failed in a row — quota likely exhausted.")
                print(f"Rewrote {rewritten} matches this run. Re-run to resume (this script")
                print("re-selects seam rows each time, so completed ones drop out).")
                print("!" * 60)
                return
            continue
        consecutive_failures = 0

        tally = {"home": _empty_tally(), "away": _empty_tally()}
        for ev in tl.get("timeline", []):
            et = ev.get("type", "")
            if et not in EVENT_TYPES:
                continue
            q = get_event_competitor_id(ev)
            if q in tally:
                _accumulate(tally[q], et)

        if tally["home"]["shots"] == 0 and tally["away"]["shots"] == 0:
            print(f"    empty/unusable timeline — leaving existing stats in place.")
            continue

        # Align SR home/away qualifiers to our fixture via the recorded score, guarding
        # against cross-provider home/away swaps.
        sr_h, sr_a = tally["home"]["goals"], tally["away"]["goals"]
        direct = (sr_h == r["home_score"] and sr_a == r["away_score"])
        swapped = (sr_h == r["away_score"] and sr_a == r["home_score"])
        if swapped and not direct:
            home_team, away_team = r["away_team_id"], r["home_team_id"]
            print("    (home/away swap detected via score — corrected)")
        else:
            home_team, away_team = r["home_team_id"], r["away_team_id"]

        with get_db() as db:
            for team_id, t in ((home_team, tally["home"]), (away_team, tally["away"])):
                db.execute(
                    """
                    INSERT OR REPLACE INTO match_stats
                    (match_id, team_id, shots_total, shots_on_target, shots_off_target,
                     blocked_shots, corners, possession, passes_total, passes_accurate,
                     fouls, offsides, saves)
                    VALUES (?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, NULL, NULL, NULL)
                    """,
                    (r["id"], team_id, t["shots"], t["on"], t["off"], t["blocked"], t["corners"]),
                )
        rewritten += 1
        h, a = tally["home"], tally["away"]
        print(f"    rewrote | H {h['shots']}S/{h['corners']}C  A {a['shots']}S/{a['corners']}C")

    print(f"\nDone: re-derived {rewritten} seam matches from SportRadar for {competition_id}.")
    print("Run `python data/coverage.py` and re-check the source split — it should be gone.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python data/refetch_stats_seam.py <competition_id> [--dry-run]")
        sys.exit(1)
    refetch_seam(sys.argv[1], "--dry-run" in sys.argv)
