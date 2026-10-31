"""Ingest schedules + scores from SportRadar into ucl.db.

Scores only (no timelines): ~2 API calls per competition-season, so the full
backfill of 7 competitions x 3 seasons is ~42 calls.

Usage (from the repo root):
    python -m ucl.ingest                 # every competition, every season
    python -m ucl.ingest ucl 2026        # one competition, one season
    python -m ucl.ingest --current       # just the current season (weekly refresh)
    python -m ucl.ingest corners         # corner counts for played UCL matches (1 call each)
    python -m ucl.ingest epl-corners     # copy EPL corners from predictable.db (no API calls)
"""
import os
import sqlite3
import sys
import time

import requests

from config import SPORTRADAR_KEY  # read-only: shares the API key, nothing else
from ucl.config import COMPETITIONS, CURRENT_SEASON, EUROPEAN, SEASONS
from ucl.db import get_db, init_db

BASE_URL = "https://api.sportradar.com/soccer-extended/trial/v4/en"
REQUEST_GAP_S = 1.5


def _get(path):
    time.sleep(REQUEST_GAP_S)
    for attempt in range(3):
        resp = requests.get(f"{BASE_URL}/{path}", params={"api_key": SPORTRADAR_KEY}, timeout=30)
        if resp.status_code != 429:
            break
        print(f"  Rate limited (429). Waiting 30s... (attempt {attempt + 1}/3)")
        time.sleep(30)

    allotted = resp.headers.get("x-plan-quota-allotted")
    used = resp.headers.get("x-plan-quota-current")
    if allotted and used:
        print(f"  [quota] monthly: {used}/{allotted} used")

    if resp.status_code != 200:
        print(f"  HTTP {resp.status_code} for {path}: {resp.text[:300]}")
        return None
    return resp.json()


def find_season(urn, year):
    """SR season id for a start year. League seasons are named like '25/26'."""
    data = _get(f"competitions/{urn}/seasons.json")
    if not data:
        return None
    candidates = {str(year), f"{year % 100:02d}/{(year + 1) % 100:02d}",
                  f"{year}/{(year + 1) % 100:02d}", f"{year}/{year + 1}"}
    for s in data.get("seasons", []):
        if str(s.get("year", "")) in candidates or any(c in str(s.get("name", "")) for c in candidates):
            return s["id"]
    names = ", ".join(s.get("name", "?") for s in data.get("seasons", [])[:8])
    print(f"  No season for {year} under {urn}. Available: {names}")
    return None


def _regulation_score(status):
    """90-minute score. Knockout games that go to extra time report the final
    score at top level; the regular-period scores are in period_scores."""
    periods = status.get("period_scores") or []
    regular = [p for p in periods if p.get("type") == "regular_period"]
    if regular:
        return (sum(p.get("home_score", 0) for p in regular),
                sum(p.get("away_score", 0) for p in regular))
    if status.get("match_status") in ("aet", "ap"):
        return None, None  # went past 90 but no breakdown: don't guess
    return status.get("home_score"), status.get("away_score")


def _parse(entry, competition, season):
    event = entry.get("sport_event", {})
    status = entry.get("sport_event_status", {})
    competitors = event.get("competitors") or []
    by_q = {c.get("qualifier"): c for c in competitors}
    home, away = by_q.get("home"), by_q.get("away")
    if not home or not away:
        return None, []

    ctx = event.get("sport_event_context") or {}
    stage = ctx.get("stage") or {}
    rnd = ctx.get("round") or {}
    round_name = rnd.get("name")
    two_legged = rnd.get("cup_round_number_of_sport_events") == 2

    closed = status.get("status") == "closed"
    reg_h, reg_a = _regulation_score(status) if closed else (None, None)
    match_status = status.get("match_status")
    decided_by = {"aet": "aet", "ap": "pen"}.get(match_status)

    teams = [(c["id"], c.get("name", "Unknown"), c.get("country"), c.get("abbreviation")) for c in (home, away)]
    row = {
        "id": event["id"],
        "competition": competition,
        "season": season,
        "date": event.get("start_time", ""),
        "home_id": home["id"],
        "away_id": away["id"],
        "status": "FT" if closed and reg_h is not None else "NS",
        "home_goals": reg_h,
        "away_goals": reg_a,
        "home_final": status.get("home_score") if closed else None,
        "away_final": status.get("away_score") if closed else None,
        "decided_by": decided_by,
        "winner_id": status.get("winner_id") if closed else None,
        # Second legs carry the aggregate winner; one-off knockouts (the final) the match winner.
        "tie_winner_id": (status.get("aggregate_winner_id") if two_legged
                          else status.get("winner_id") if round_name == "final" else None) if closed else None,
        "stage": stage.get("phase"),
        "round_name": round_name,
        "round_number": rnd.get("number"),
        "tie_id": rnd.get("cup_round_id"),
        "leg": rnd.get("cup_round_sport_event_number") if two_legged else None,
        "neutral": 1 if round_name == "final" else 0,
    }
    if closed and reg_h is None:
        print(f"  WARN {event['id']}: closed after extra time with no period scores; stored as NS")
    return row, teams


def ingest(competition, season):
    urn = COMPETITIONS[competition]["urn"]
    print(f"\n{competition} {season}/{(season + 1) % 100:02d} ({urn})")
    season_id = find_season(urn, season)
    if not season_id:
        return
    data = _get(f"seasons/{season_id}/schedules.json")
    if not data:
        return

    n_ft = n_ns = 0
    with get_db() as db:
        for entry in data.get("schedules", []):
            ev_id = entry.get("sport_event", {}).get("id")
            # Postponed games come back under a new event id; drop the stale row.
            if entry.get("sport_event_status", {}).get("status") in ("postponed", "cancelled", "abandoned"):
                db.execute("DELETE FROM matches WHERE id = ?", (ev_id,))
                continue
            row, teams = _parse(entry, competition, season)
            if not row:
                continue
            # Relegation playoffs bring in second-division clubs with no other data.
            if competition not in EUROPEAN and row["stage"] not in (None, "regular season"):
                continue
            for tid, name, country, abbr in teams:
                db.execute(
                    "INSERT INTO teams (id, name, country, abbreviation) VALUES (?, ?, ?, ?) "
                    "ON CONFLICT(id) DO UPDATE SET name = excluded.name, "
                    "country = COALESCE(excluded.country, teams.country), "
                    "abbreviation = COALESCE(excluded.abbreviation, teams.abbreviation)",
                    (tid, name, country, abbr),
                )
            cols = ", ".join(row)
            db.execute(f"INSERT OR REPLACE INTO matches ({cols}) VALUES ({', '.join('?' * len(row))})",
                       tuple(row.values()))
            if row["status"] == "FT":
                n_ft += 1
            else:
                n_ns += 1
        n_ties = _derive_tie_winners(db, competition, season)
    print(f"  {n_ft} completed, {n_ns} upcoming" + (f", {n_ties} ties decided" if n_ties else ""))


def _derive_tie_winners(db, competition, season):
    """The schedule feed has no aggregate winner, so set it on each second leg from
    the aggregate of the final scores (incl. extra time), falling back to the
    penalty-shootout winner of leg 2 when level."""
    rows = db.execute(
        "SELECT * FROM matches WHERE competition = ? AND season = ? AND tie_id IS NOT NULL "
        "AND leg IS NOT NULL AND status = 'FT'", (competition, season)).fetchall()
    ties = {}
    for r in rows:
        ties.setdefault(r["tie_id"], {})[r["leg"]] = r
    n = 0
    for t in ties.values():
        if 1 not in t or 2 not in t:
            continue
        l1, l2 = t[1], t[2]
        team1 = l1["home_final"] + l2["away_final"]
        team2 = l1["away_final"] + l2["home_final"]
        winner = (l1["home_id"] if team1 > team2 else l1["away_id"] if team2 > team1
                  else l2["winner_id"])
        if winner:
            db.execute("UPDATE matches SET tie_winner_id = ? WHERE id = ?", (winner, l2["id"]))
            n += 1
    return n


def fetch_corners(seasons=SEASONS):
    """Corner counts from each played UCL match's summary (league phase onward;
    qualifiers skipped). Resumable: only matches without corners are requested,
    and it stops after 5 failures in a row so a spent trial quota isn't hammered
    (the EPL refresh shares the key)."""
    with get_db() as db:
        todo = [r["id"] for r in db.execute(
            "SELECT id FROM matches WHERE competition = 'ucl' AND status = 'FT' AND stage <> 'qualification' "
            f"AND home_corners IS NULL AND season IN ({','.join('?' * len(seasons))}) ORDER BY date",
            tuple(seasons))]
    print(f"{len(todo)} UCL matches need corners")
    failures = done = 0
    for i, match_id in enumerate(todo, 1):
        data = _get(f"sport_events/{match_id}/summary.json")
        corners = {}
        for c in ((data or {}).get("statistics", {}).get("totals", {}).get("competitors", [])):
            corners[c.get("qualifier")] = (c.get("statistics") or {}).get("corner_kicks")
        if corners.get("home") is None or corners.get("away") is None:
            failures += 1
            if failures >= 5:
                print(f"Stopping: 5 failures in a row (quota?). {done} saved this run; re-run to resume.")
                return
            continue
        failures = 0
        with get_db() as db:
            db.execute("UPDATE matches SET home_corners = ?, away_corners = ? WHERE id = ?",
                       (corners["home"], corners["away"], match_id))
        done += 1
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
    print(f"Done: {done} matches")


def import_epl_corners():
    """Copy EPL corner counts from predictable.db (opened read-only). Both databases
    key matches by the SportRadar event id, so no API calls are needed. Stats rows
    are matched to home/away by the club's sr:competitor id, not the row id:
    predictable.db has duplicate rows for some clubs ('Ipswich' and 'Ipswich Town')
    and a few stats rows sit under the older one."""
    src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "predictable.db")
    con = sqlite3.connect(f"file:{src}?mode=ro", uri=True)
    rows = con.execute("""
        SELECT m.sportradar_id, hs.corners, aw.corners
        FROM matches m
        JOIN teams ht ON ht.id = m.home_team_id
        JOIN teams at ON at.id = m.away_team_id
        JOIN match_stats hs ON hs.match_id = m.id
        JOIN teams hst ON hst.id = hs.team_id AND hst.sportradar_id = ht.sportradar_id
        JOIN match_stats aw ON aw.match_id = m.id
        JOIN teams ast ON ast.id = aw.team_id AND ast.sportradar_id = at.sportradar_id
        WHERE m.competition_id LIKE 'epl_%' AND m.sportradar_id IS NOT NULL
          AND hs.corners IS NOT NULL AND aw.corners IS NOT NULL
    """).fetchall()
    con.close()
    with get_db() as db:
        n = sum(db.execute("UPDATE matches SET home_corners = ?, away_corners = ? "
                           "WHERE id = ? AND competition = 'epl'", (h, a, sr_id)).rowcount
                for sr_id, h, a in rows)
    print(f"Copied corners for {n} EPL matches ({len(rows)} available in predictable.db)")


def main(argv):
    init_db()
    if argv and argv[0] == "corners":
        fetch_corners([int(argv[1])] if len(argv) > 1 else SEASONS)
        return
    if argv and argv[0] == "epl-corners":
        import_epl_corners()
        return
    if argv and argv[0] == "--current":
        jobs = [(c, CURRENT_SEASON) for c in COMPETITIONS]
    elif argv:
        comps = [argv[0]]
        seasons = [int(argv[1])] if len(argv) > 1 else list(SEASONS)
        jobs = [(c, s) for c in comps for s in seasons]
    else:
        jobs = [(c, s) for c in COMPETITIONS for s in SEASONS]

    for comp, season in jobs:
        if comp not in COMPETITIONS:
            sys.exit(f"Unknown competition {comp!r}. Options: {', '.join(COMPETITIONS)}")
        ingest(comp, season)


if __name__ == "__main__":
    main(sys.argv[1:])
