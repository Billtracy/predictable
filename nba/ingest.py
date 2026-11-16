"""Ingest NBA games, box scores and betting lines from ESPN's public JSON API.

No key needed. Per season: ~270 scoreboard calls (one per date), then one summary
call (box score + player minutes) and one odds call per game. Re-running is cheap:
box scores are fetched once per final game, lines once per game after it is final
(plus every run for upcoming games, since they move).

Usage (from the repo root):
    python -m nba.ingest                 # every season in config.SEASONS
    python -m nba.ingest 2024            # one season (start year)
    python -m nba.ingest --current       # current season only (daily refresh)
"""
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta, timezone

import requests

from nba.config import CURRENT_SEASON, LINE_PROVIDERS, SEASONS
from nba.db import get_db, init_db

SITE = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CORE = "https://sports.core.api.espn.com/v2/sports/basketball/leagues/nba"
WORKERS = 6
NBA_TEAM_IDS = {str(i) for i in range(1, 31)}   # excludes All-Star / exhibition sides
GAME_TYPES = {2: "regular", 3: "playoff", 5: "playin"}  # 1 = preseason, skipped

_session = requests.Session()
_session.headers["User-Agent"] = "predictable-nba-ingest"


def _get(url, params=None):
    for attempt in range(4):
        try:
            resp = _session.get(url, params=params, timeout=30)
        except requests.RequestException as e:
            print(f"  {type(e).__name__} for {url}; retrying")
            time.sleep(2 * (attempt + 1))
            continue
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 404:
            return None
        print(f"  HTTP {resp.status_code} for {url}; retrying")
        time.sleep(5 * (attempt + 1))
    return None


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# --- schedule / scores -------------------------------------------------------

def _parse_event(ev):
    comp = ev["competitions"][0]
    game_type = GAME_TYPES.get(ev.get("season", {}).get("type"))
    if not game_type or comp.get("type", {}).get("abbreviation") == "ALLSTAR":
        return None, []
    sides = {c["homeAway"]: c for c in comp["competitors"]}
    if set(sides) != {"home", "away"} or not {sides[s]["team"]["id"] for s in sides} <= NBA_TEAM_IDS:
        return None, []
    status = comp["status"]["type"]
    if status.get("name") in ("STATUS_POSTPONED", "STATUS_CANCELED"):
        return {"id": ev["id"], "_drop": True}, []

    row = {
        "id": ev["id"],
        "season": ev["season"]["year"] - 1,
        "date": ev["date"],
        "game_type": game_type,
        "neutral": int(bool(comp.get("neutralSite"))),
        "home_id": sides["home"]["team"]["id"],
        "away_id": sides["away"]["team"]["id"],
        "status": "FT" if status.get("completed") else "NS",
    }
    if row["status"] == "FT":
        h_ls = [int(l["value"]) for l in sides["home"].get("linescores", [])]
        a_ls = [int(l["value"]) for l in sides["away"].get("linescores", [])]
        row.update(home_pts=int(sides["home"]["score"]), away_pts=int(sides["away"]["score"]),
                   periods=max(len(h_ls), 4))
        if len(h_ls) >= 4 and len(a_ls) >= 4:
            row.update(home_reg=sum(h_ls[:4]), away_reg=sum(a_ls[:4]))
    teams = [(s["team"]["id"], s["team"]["abbreviation"], s["team"]["displayName"]) for s in sides.values()]
    return row, teams


def _season_dates(season):
    start = date(season, 10, 1)
    end = date(season + 1, 6, 30)
    if season == CURRENT_SEASON:
        end = min(end, date.today() + timedelta(days=7))
    d = start
    while d <= end:
        yield d.strftime("%Y%m%d")
        d += timedelta(days=1)


def ingest_schedule(season):
    dates = list(_season_dates(season))
    with ThreadPoolExecutor(WORKERS) as pool:
        boards = list(pool.map(lambda d: _get(f"{SITE}/scoreboard", {"dates": d}), dates))
    n_ft = n_ns = 0
    with get_db() as db:
        for board in boards:
            for ev in (board or {}).get("events", []):
                row, teams = _parse_event(ev)
                if not row:
                    continue
                if row.get("_drop"):
                    db.execute("DELETE FROM lines WHERE game_id = ?", (row["id"],))
                    db.execute("DELETE FROM games WHERE id = ?", (row["id"],))
                    continue
                if row["season"] != season:
                    continue
                for tid, abbr, name in teams:
                    db.execute("INSERT INTO teams (id, abbreviation, name) VALUES (?, ?, ?) "
                               "ON CONFLICT(id) DO UPDATE SET abbreviation = excluded.abbreviation, "
                               "name = excluded.name", (tid, abbr, name))
                cols = ", ".join(row)
                updates = ", ".join(f"{c} = excluded.{c}" for c in row if c != "id")
                db.execute(f"INSERT INTO games ({cols}) VALUES ({', '.join('?' * len(row))}) "
                           f"ON CONFLICT(id) DO UPDATE SET {updates}", tuple(row.values()))
                n_ft += row["status"] == "FT"
                n_ns += row["status"] == "NS"
    print(f"  schedule: {n_ft} final, {n_ns} upcoming")


# --- box scores ----------------------------------------------------------------

def _stat(team_box, name):
    for s in team_box["statistics"]:
        if s["name"] == name:
            return s["displayValue"]
    return None


def _attempts(made_attempted):
    return int(made_attempted.split("-")[1]) if made_attempted else None


def _int(x):
    try:
        return int(str(x).lstrip("+"))
    except (TypeError, ValueError):
        return None


def _fetch_box(game_id):
    data = _get(f"{SITE}/summary", {"event": game_id})
    if not data or "boxscore" not in data:
        return game_id, None, []
    box = {}
    for t in data["boxscore"].get("teams", []):
        side = t.get("homeAway")
        if side not in ("home", "away"):
            continue
        box[f"{side}_fga"] = _attempts(_stat(t, "fieldGoalsMade-fieldGoalsAttempted"))
        box[f"{side}_fta"] = _attempts(_stat(t, "freeThrowsMade-freeThrowsAttempted"))
        box[f"{side}_oreb"] = _int(_stat(t, "offensiveRebounds"))
        # totalTurnovers includes team turnovers (shot clock etc.), which do end possessions.
        box[f"{side}_tov"] = _int(_stat(t, "totalTurnovers") or _stat(t, "turnovers"))
    players = []
    for team in data["boxscore"].get("players", []):
        tid = team["team"]["id"]
        for block in team.get("statistics", []):
            keys = block.get("keys") or []
            for a in block.get("athletes", []):
                if not a.get("athlete", {}).get("id"):
                    continue      # occasional placeholder entry with no player attached
                stats = dict(zip(keys, a.get("stats") or []))
                players.append((game_id, tid, a["athlete"]["id"], a["athlete"].get("displayName"),
                                int(bool(a.get("starter"))), _int(stats.get("minutes")),
                                _int(stats.get("points")), _int(stats.get("plusMinus")),
                                a.get("reason") if a.get("didNotPlay") else None))
    return game_id, box, players


def ingest_boxes(season):
    with get_db() as db:
        ids = [r["id"] for r in db.execute(
            "SELECT id FROM games WHERE season = ? AND status = 'FT' AND home_fga IS NULL", (season,))]
    if not ids:
        return
    print(f"  box scores: fetching {len(ids)}")
    done = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        for game_id, box, players in pool.map(_fetch_box, ids):
            if not box:
                continue
            with get_db() as db:
                sets = ", ".join(f"{k} = ?" for k in box)
                db.execute(f"UPDATE games SET {sets} WHERE id = ?", (*box.values(), game_id))
                db.executemany("INSERT OR REPLACE INTO player_games VALUES (?,?,?,?,?,?,?,?,?)", players)
            done += 1
            if done % 250 == 0:
                print(f"    {done}/{len(ids)}")
    print(f"  box scores: {done} stored")


# --- lines ---------------------------------------------------------------------

def _num(x):
    if x is None:
        return None
    try:
        return float(str(x).lstrip("ou+").replace("PK", "0"))
    except ValueError:
        return None


def _american(node, *path):
    for p in path:
        node = (node or {}).get(p)
    return _num((node or {}).get("american")) if isinstance(node, dict) else None


def _fetch_line(game_id):
    data = _get(f"{CORE}/events/{game_id}/competitions/{game_id}/odds")
    items = [i for i in (data or {}).get("items", [])
             if "live" not in i.get("provider", {}).get("name", "").lower() and i.get("overUnder") is not None]
    if not items:
        return game_id, None
    rank = {p: k for k, p in enumerate(LINE_PROVIDERS)}
    item = min(items, key=lambda i: rank.get(i["provider"]["name"], len(rank)))
    return game_id, {
        "provider": item["provider"]["name"],
        "total": _num(item.get("overUnder")),
        "spread": _num(item.get("spread")),
        "total_open": _american(item, "open", "total"),
        "spread_open": _american(item, "homeTeamOdds", "open", "pointSpread"),
        "over_odds": _num(item.get("overOdds")),
        "under_odds": _num(item.get("underOdds")),
    }


def ingest_lines(season):
    with get_db() as db:
        ids = [r["id"] for r in db.execute(
            "SELECT g.id FROM games g LEFT JOIN lines l ON l.game_id = g.id WHERE g.season = ? AND "
            "(g.status = 'NS' OR l.game_id IS NULL OR l.fetched_at < g.date)", (season,))]
    if not ids:
        return
    print(f"  lines: fetching {len(ids)}")
    stored = 0
    with ThreadPoolExecutor(WORKERS) as pool:
        results = list(pool.map(_fetch_line, ids))
    with get_db() as db:
        for game_id, line in results:
            if not line:
                continue
            db.execute("INSERT OR REPLACE INTO lines (game_id, provider, total, spread, total_open, "
                       "spread_open, over_odds, under_odds, fetched_at) VALUES (?,?,?,?,?,?,?,?,?)",
                       (game_id, line["provider"], line["total"], line["spread"], line["total_open"],
                        line["spread_open"], line["over_odds"], line["under_odds"], _now()))
            stored += 1
    print(f"  lines: {stored} stored ({len(ids) - stored} games without a line)")


def ingest(season):
    print(f"\nNBA {season}-{(season + 1) % 100:02d}")
    ingest_schedule(season)
    ingest_boxes(season)
    ingest_lines(season)


def main(argv):
    init_db()
    if "--current" in argv:
        seasons = [CURRENT_SEASON]
    elif argv:
        seasons = [int(a) for a in argv]
    else:
        seasons = list(SEASONS)
    for s in seasons:
        ingest(s)


if __name__ == "__main__":
    main(sys.argv[1:])
