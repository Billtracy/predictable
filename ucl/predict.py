"""Predict upcoming UCL fixtures and live knockout ties.

Usage (from the repo root):
    python -m ucl.predict              # next matchday
    python -m ucl.predict --days 14    # everything in the next 14 days
    python -m ucl.predict --json out.json
"""
import argparse
import json
from datetime import datetime, timedelta, timezone

from ucl import config as C
from ucl.db import load_matches, team_names
from ucl.model import UCLModel, team_leagues
from ucl.tips import corner_tip, goal_tip


def upcoming(matches, now, days):
    ns = [m for m in matches if m["competition"] == C.UCL and m["status"] == "NS" and m["date"] >= now]
    if not ns:
        return []
    if days is None:  # next matchday: the first fixture's day plus the following day
        first = datetime.fromisoformat(ns[0]["date"].replace("Z", "+00:00"))
        cutoff = (first + timedelta(days=2)).date().isoformat()
    else:
        cutoff = (datetime.fromisoformat(now) + timedelta(days=days)).date().isoformat()
    return [m for m in ns if m["date"][:10] < cutoff]


def live_ties(matches):
    """Two-legged ties whose second leg is still to play: (leg1, leg2) pairs."""
    legs = {}
    for m in matches:
        if m["competition"] == C.UCL and m["season"] == C.CURRENT_SEASON and m["tie_id"] and m["leg"]:
            legs.setdefault(m["tie_id"], {})[m["leg"]] = m
    return [(t[1], t[2]) for t in legs.values() if 1 in t and 2 in t and t[2]["status"] == "NS"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int)
    ap.add_argument("--json", metavar="PATH")
    a = ap.parse_args()

    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    matches = load_matches()
    names = team_names()
    leagues = team_leagues(matches)
    model = UCLModel().fit(matches, now, leagues)
    cmodel = UCLModel(C.CORNER_HALF_LIFE_DAYS, C.CORNER_TEAM_RIDGE, C.CORNER_LEAGUE_RIDGE,
                      target="corners").fit(matches, now, leagues)
    fixtures = upcoming(matches, now, a.days)

    out = {"generated_at": now, "fixtures": [], "ties": []}
    for m in fixtures:
        p = model.predict(m["home_id"], m["away_id"], C.UCL, bool(m["neutral"]))
        c = cmodel.predict_corners(m["home_id"], m["away_id"], C.UCL, bool(m["neutral"]))
        out["fixtures"].append({"id": m["id"], "date": m["date"], "round": m["round_name"],
                                "home": names[m["home_id"]], "away": names[m["away_id"]],
                                "goals": p, "corners": c,
                                "goal_tip": goal_tip(p["home_xg"] + p["away_xg"]),
                                "corner_tip": corner_tip(c["total_corners"])})
    for leg1, leg2 in live_ties(matches):
        played = (leg1["home_goals"], leg1["away_goals"]) if leg1["status"] == "FT" else None
        pa = model.tie_advance_prob(leg1["home_id"], leg1["away_id"], leg1_score=played)
        out["ties"].append({"round": leg1["round_name"], "team1": names[leg1["home_id"]],
                            "team2": names[leg1["away_id"]], "leg1_score": played,
                            "team1_advance": pa})

    if a.json:
        with open(a.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Wrote {len(out['fixtures'])} fixtures, {len(out['ties'])} ties to {a.json}")
        return

    print(f"Fitted on {model.n_train} matches up to {now[:10]}\n")
    if not fixtures:
        print("No upcoming UCL fixtures in ucl.db (run `python -m ucl.ingest --current`).")
    for f in out["fixtures"]:
        print(f"{f['date'][:16].replace('T', ' ')}  {f['home']} v {f['away']}  ({f['round'] or 'league phase'})")
        g, c = f["goals"], f["corners"]
        print(f"   goals   xG {g['home_xg']:.2f}-{g['away_xg']:.2f}   O1.5 {g['over'][1.5] * 100:.0f}%   "
              f"O2.5 {g['over'][2.5] * 100:.0f}%   O3.5 {g['over'][3.5] * 100:.0f}%   tip {f['goal_tip']}")
        print(f"   corners {c['home_corners']:.1f}-{c['away_corners']:.1f} ({c['total_corners']:.1f})   "
              f"O8.5 {c['over'][8.5] * 100:.0f}%   O9.5 {c['over'][9.5] * 100:.0f}%   "
              f"O10.5 {c['over'][10.5] * 100:.0f}%   tip {f['corner_tip']}")
    for t in out["ties"]:
        score = f" (leg 1 {t['leg1_score'][0]}-{t['leg1_score'][1]})" if t["leg1_score"] else ""
        print(f"\n{t['round']}: {t['team1']} v {t['team2']}{score}  ->  "
              f"{t['team1']} advance {t['team1_advance'] * 100:.0f}%")


if __name__ == "__main__":
    main()
