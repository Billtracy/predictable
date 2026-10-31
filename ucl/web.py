"""Flask blueprint for the UCL page and its API. Registered by app.py; shares
nothing with the EPL endpoints except the Flask app object."""
import math
import os
import threading
from datetime import datetime, timezone

from flask import Blueprint, jsonify, send_from_directory

from ucl import config as C
from ucl.db import load_matches, team_info
from ucl.model import UCLModel, team_leagues
from ucl.tips import corner_tip, goal_tip, tip_prob, tip_won

ucl_bp = Blueprint("ucl", __name__)
STATIC_DIR = os.path.join(os.path.dirname(C.UCL_DIR), "static")

ROUND_LABELS = {
    "playoff_round": "Knockout play-offs",
    "round_of_16": "Round of 16",
    "quarterfinal": "Quarter-finals",
    "semifinal": "Semi-finals",
    "final": "Final",
}

# Fitted models keyed by as_of date. Cleared whenever ucl.db changes on disk.
_cache = {"mtime": None, "data": None, "fits": {}}
_lock = threading.Lock()


def _state():
    """Matches, team info and the fit cache, reloaded when ucl.db is re-ingested."""
    mtime = os.path.getmtime(C.UCL_DB_PATH)
    with _lock:
        if _cache["mtime"] != mtime:
            matches = load_matches()
            _cache.update(mtime=mtime, fits={}, data={
                "matches": matches, "teams": team_info(), "leagues": team_leagues(matches)})
        return _cache["data"]


def _fit(as_of, target="goals"):
    data = _state()
    with _lock:
        model = _cache["fits"].get((as_of, target))
    if model is None:
        params = ((C.HALF_LIFE_DAYS, C.TEAM_RIDGE, C.LEAGUE_RIDGE) if target == "goals" else
                  (C.CORNER_HALF_LIFE_DAYS, C.CORNER_TEAM_RIDGE, C.CORNER_LEAGUE_RIDGE))
        model = UCLModel(*params, target=target).fit(data["matches"], as_of, data["leagues"])
        with _lock:
            _cache["fits"][(as_of, target)] = model
    return model


def _matchdays(matches):
    """Current season's competition proper, grouped into matchdays in date order:
    league-phase rounds by number, knockouts by round + leg."""
    groups = {}
    for m in matches:
        if m["competition"] != C.UCL or m["season"] != C.CURRENT_SEASON or m["stage"] == "qualification":
            continue
        if m["round_name"] in ROUND_LABELS:
            key = f"{m['round_name']}-{m['leg'] or 1}"
            label = ROUND_LABELS[m["round_name"]] + (f" · Leg {m['leg']}" if m["leg"] else "")
        else:
            key = f"md{m['round_number']}"
            label = f"Matchday {m['round_number']}"
        groups.setdefault(key, {"key": key, "label": label, "matches": []})["matches"].append(m)
    days = sorted(groups.values(), key=lambda g: min(m["date"] for m in g["matches"]))
    for d in days:
        d["matches"].sort(key=lambda m: m["date"])
        d["start"] = d["matches"][0]["date"]
        d["end"] = d["matches"][-1]["date"]
        d["played"] = sum(m["status"] == "FT" for m in d["matches"])
    return days


def _as_of(day):
    """Played (or partly played) matchdays are predicted as of the morning of their
    first day, the same point-in-time view the backtest scores. Upcoming ones use
    everything known now."""
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    start = f"{day['start'][:10]}T00:00:00+00:00"
    return start if start <= now else now[:10] + "T00:00:00+00:00"


def _pct(p):
    return round(p * 100)


def _team(teams, tid):
    t = teams.get(tid, {})
    return {"name": t.get("name", "?"), "code": t.get("abbreviation") or t.get("name", "?")[:3].upper(),
            "country": t.get("country")}


@ucl_bp.route("/ucl")
def ucl_page():
    return send_from_directory(STATIC_DIR, "ucl.html")


@ucl_bp.route("/api/ucl/matchdays")
def ucl_matchdays():
    days = _matchdays(_state()["matches"])
    if not days:
        return jsonify({"success": False, "error": "No UCL data. Run `python -m ucl.ingest`."})
    current = next((i for i, d in enumerate(days) if d["played"] < len(d["matches"])), len(days) - 1)
    return jsonify({"success": True, "current": current, "matchdays": [
        {"key": d["key"], "label": d["label"], "start": d["start"], "end": d["end"],
         "played": d["played"], "total": len(d["matches"])} for d in days]})


@ucl_bp.route("/api/ucl/matchday/<key>")
def ucl_matchday(key):
    data = _state()
    day = next((d for d in _matchdays(data["matches"]) if d["key"] == key), None)
    if not day:
        return jsonify({"success": False, "error": f"Unknown matchday {key}"}), 404
    as_of = _as_of(day)
    model = _fit(as_of)
    cmodel = _fit(as_of, "corners")
    teams = data["teams"]

    fixtures = []
    for m in day["matches"]:
        neutral = bool(m["neutral"])
        p = model.predict(m["home_id"], m["away_id"], C.UCL, neutral)
        c = cmodel.predict_corners(m["home_id"], m["away_id"], C.UCL, neutral)
        g_tip = goal_tip(p["home_xg"] + p["away_xg"])
        c_tip = corner_tip(c["total_corners"])
        actual = None
        if m["status"] == "FT":
            corners = (m["home_corners"] + m["away_corners"]) if m["home_corners"] is not None else None
            actual = {"home": m["home_goals"], "away": m["away_goals"],
                      "home_final": m["home_final"], "away_final": m["away_final"],
                      "decided_by": m["decided_by"], "corners": corners}
        fixtures.append({
            "id": m["id"], "date": m["date"],
            "home": _team(teams, m["home_id"]), "away": _team(teams, m["away_id"]),
            "home_win": _pct(p["home_win"]), "draw": _pct(p["draw"]), "away_win": _pct(p["away_win"]),
            "goals": {
                "home_xg": round(p["home_xg"], 2), "away_xg": round(p["away_xg"], 2),
                "over": {str(k): _pct(v) for k, v in p["over"].items()},
                "btts": _pct(p["btts"]),
                "most_likely_score": p["most_likely_score"],
                "tip": g_tip, "tip_prob": _pct(tip_prob(g_tip, p["over"])),
                "tip_won": tip_won(g_tip, actual["home"] + actual["away"]) if actual else None,
            },
            "corners": {
                "home": round(c["home_corners"], 1), "away": round(c["away_corners"], 1),
                "total": round(c["total_corners"], 1),
                "over": {str(k): _pct(v) for k, v in c["over"].items()},
                "tip": c_tip, "tip_prob": _pct(tip_prob(c_tip, c["over"])),
                "tip_won": tip_won(c_tip, actual["corners"]) if actual and actual["corners"] is not None else None,
            },
            "actual": actual,
        })

    # Knockout ties: P(advance) for every tie touching this matchday.
    ties = []
    if day["key"].split("-")[0] in ROUND_LABELS:
        by_tie = {}
        for m in data["matches"]:
            if m["tie_id"] and m["leg"] and m["competition"] == C.UCL and m["season"] == C.CURRENT_SEASON:
                by_tie.setdefault(m["tie_id"], {})[m["leg"]] = m
        for m in day["matches"]:
            if m["round_name"] == "final":
                pa = model.tie_advance_prob(m["home_id"], m["away_id"], single_leg_neutral=True)
                t1, t2, leg1, winner = m["home_id"], m["away_id"], None, m["tie_winner_id"]
            elif m["leg"] == 1 or (m["leg"] == 2 and 1 in by_tie.get(m["tie_id"], {})):
                tie = by_tie[m["tie_id"]]
                first = tie[1]
                # Leg 1's result is only known to the model if it was played before as_of.
                leg1 = ((first["home_goals"], first["away_goals"])
                        if first["status"] == "FT" and first["date"] < as_of else None)
                t1, t2 = first["home_id"], first["away_id"]
                pa = model.tie_advance_prob(t1, t2, leg1_score=leg1)
                winner = tie.get(2, {}).get("tie_winner_id")
            else:
                continue
            ties.append({"team1": _team(teams, t1), "team2": _team(teams, t2),
                         "team1_advance": _pct(pa), "leg1_score": leg1,
                         "winner": None if not winner else (1 if winner == t1 else 2)})

    return jsonify({"success": True, "key": key, "label": day["label"], "as_of": as_of,
                    "fixtures": fixtures, "ties": ties})


@ucl_bp.route("/api/ucl/model")
def ucl_model():
    now = datetime.now(timezone.utc).date().isoformat() + "T00:00:00+00:00"
    model = _fit(now)
    names = {"epl": "Premier League", "laliga": "LaLiga", "seriea": "Serie A",
             "bundesliga": "Bundesliga", "ligue1": "Ligue 1", C.OTHER_LEAGUE: "Rest of Europe"}
    # Expected goal difference per game against an average side on neutral ground:
    # goals for exp(base + attack) minus goals against exp(base + defense).
    base = math.exp(model.mu + model.shift[C.UCL])

    def gd(att, dfn):
        return round(base * (math.exp(att) - math.exp(dfn)), 2)

    leagues = sorted(({"league": names[l], "gd": gd(model.A[l], model.D[l])} for l in model.league_names),
                     key=lambda r: -r["gd"])
    # This season's league-phase clubs (league effect included).
    data = _state()
    entrants = {m[side] for m in data["matches"] for side in ("home_id", "away_id")
                if m["competition"] == C.UCL and m["season"] == C.CURRENT_SEASON and m["stage"] == "regular season"}
    teams = []
    for tid in entrants:
        att, dfn = model._strength(tid)
        teams.append({**_team(data["teams"], tid), "gd": gd(att, dfn),
                      "league": names[model.leagues.get(tid, C.OTHER_LEAGUE)]})
    teams.sort(key=lambda t: -t["gd"])
    return jsonify({"success": True, "as_of": now, "n_train": model.n_train,
                    "home_advantage": round(model.home[C.UCL], 3), "leagues": leagues, "teams": teams})
