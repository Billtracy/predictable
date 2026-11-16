"""Half-time live-odds logger + paper-trade evaluator for EPL total-corner markets.

At each EPL half-time it snapshots the in-play corner markets from API-Sports
(/odds/live, works on the free plan), reads the HT corner count from the
Sportradar live timeline, prices every total-corners line with the HT pricer
(experiments/ht_corner_pricer.py) and appends everything to data/ht_odds_log.jsonl.
Nothing is published or deployed; this is paper trading only.

Usage
-----
  python experiments/ht_odds_logger.py run                  # today's EPL fixtures (UTC date)
  python experiments/ht_odds_logger.py run --date 2026-11-21
  python experiments/ht_odds_logger.py run --max-calls 40   # API-Sports call budget (free tier: 100/day)
  python experiments/ht_odds_logger.py snapshot             # one /odds/live call now, any live EPL game
  python experiments/ht_odds_logger.py evaluate [--min-ev 0.03]

Run `run` before the first kickoff and leave it going (e.g. in tmux/nohup); it
sleeps until each kickoff slot's half-time window. After the round, ingest the
Sportradar timelines as usual so `evaluate` can settle bets from stored events.
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)
sys.path.append(os.path.join(ROOT, "experiments"))

from config import API_SPORTS_HOST, API_SPORTS_KEY  # noqa: E402
from db.database import get_db  # noqa: E402
from ht_corner_pricer import load_params, price_line, settle_ev  # noqa: E402

LOG_PATH = os.path.join(ROOT, "data", "ht_odds_log.jsonl")
TEAM_MAP_PATH = os.path.join(ROOT, "data", "ht_api_team_map.json")  # API-Sports team id -> our team id
COMPETITION = "epl_2026"
API_LEAGUE = 39
TOTAL_MARKETS = {"Total Corners", "Asian Corners", "Match Corners"}  # FT total-corner O/U lines

WINDOW_START_MIN = 46   # minutes after kickoff to start polling
WINDOW_END_MIN = 72     # give up on the slot after this
POLL_SECONDS = 150


class Budget:
    def __init__(self, max_calls):
        self.left = max_calls

    def spend(self):
        if self.left <= 0:
            raise RuntimeError("API-Sports call budget exhausted for this run")
        self.left -= 1


def now_utc():
    return datetime.now(timezone.utc)


def load_team_map():
    try:
        with open(TEAM_MAP_PATH) as f:
            return {int(k): int(v) for k, v in json.load(f).items()}
    except FileNotFoundError:
        return {}


def save_team_map(m):
    with open(TEAM_MAP_PATH, "w") as f:
        json.dump({str(k): v for k, v in sorted(m.items())}, f, indent=2)


def append_log(record):
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def logged_match_ids():
    try:
        with open(LOG_PATH) as f:
            return {json.loads(line)["match_id"] for line in f if line.strip()}
    except FileNotFoundError:
        return set()


# ------------------------------------------------------------------ data access

def fixtures_on(date_str):
    with get_db() as db:
        rows = db.execute("""
            SELECT m.id, m.date, m.sportradar_id, m.home_team_id, m.away_team_id,
                   th.name AS home, ta.name AS away
            FROM matches m
            JOIN teams th ON th.id = m.home_team_id
            JOIN teams ta ON ta.id = m.away_team_id
            WHERE m.competition_id = ? AND substr(m.date, 1, 10) = ?
            ORDER BY m.date
        """, (COMPETITION, date_str)).fetchall()
    return [dict(r) for r in rows]


def fetch_live_odds(budget):
    budget.spend()
    r = requests.get(f"https://{API_SPORTS_HOST}/odds/live",
                     headers={"x-apisports-key": API_SPORTS_KEY},
                     params={"league": API_LEAGUE}, timeout=30)
    print(f"  [api-sports] /odds/live HTTP {r.status_code}, daily remaining "
          f"{r.headers.get('x-ratelimit-requests-remaining')}, run budget left {budget.left}")
    if r.status_code != 200:
        return []
    j = r.json()
    if j.get("errors"):
        print(f"  API error: {j['errors']}")
        return []
    return j.get("response", [])


def ht_corners_sportradar(sr_event_id):
    """First-half corner count from the live Sportradar timeline (None on failure)."""
    from data.sportradar import fetch_match_timeline
    data = fetch_match_timeline(sr_event_id)
    if not data:
        return None
    return sum(1 for ev in data.get("timeline", [])
               if ev.get("type") == "corner_kick" and ev.get("period") == 1)


def pre_match_ft(fx):
    from models import fitted
    pred = fitted.predict_corners(COMPETITION, fx["home_team_id"], fx["away_team_id"], fx["date"])
    return None if pred is None else pred["home_expected"] + pred["away_expected"]


# ------------------------------------------------------------------ matching

def match_live_to_fixtures(live, fixtures, team_map):
    """Pair /odds/live entries with our fixtures: known team ids first, then by
    elimination when exactly one fixture and one live entry are left over."""
    pairs, used_live = {}, set()
    for fx in fixtures:
        for i, ev in enumerate(live):
            if i in used_live:
                continue
            h, a = ev["teams"]["home"]["id"], ev["teams"]["away"]["id"]
            if team_map.get(h) == fx["home_team_id"] or team_map.get(a) == fx["away_team_id"] \
                    or h == fx["home_team_id"] or a == fx["away_team_id"]:
                pairs[fx["id"]] = ev
                used_live.add(i)
                break
    left_fx = [fx for fx in fixtures if fx["id"] not in pairs]
    left_live = [ev for i, ev in enumerate(live) if i not in used_live]
    if len(left_fx) == 1 and len(left_live) == 1:
        pairs[left_fx[0]["id"]] = left_live[0]
    for fx in fixtures:  # learn ids for promoted teams with synthetic ids
        ev = pairs.get(fx["id"])
        if ev:
            team_map[ev["teams"]["home"]["id"]] = fx["home_team_id"]
            team_map[ev["teams"]["away"]["id"]] = fx["away_team_id"]
    return pairs


def is_half_time(ev):
    st = ev["fixture"]["status"]
    long = (st.get("long") or "").lower().replace(" ", "")
    return long == "halftime" or (st.get("short") or "").upper() == "HT"


def corner_lines(ev):
    """Flatten FT total-corner O/U markets into [{market, line, over, under, suspended}]."""
    out = []
    for m in ev.get("odds", []):
        if m["name"] not in TOTAL_MARKETS:
            continue
        by_line = defaultdict(dict)
        for v in m["values"]:
            if v.get("handicap") is None or v["value"] not in ("Over", "Under"):
                continue
            d = by_line[float(v["handicap"])]
            d[v["value"].lower()] = float(v["odd"])
            d["suspended"] = d.get("suspended", False) or bool(v.get("suspended"))
            d["main"] = d.get("main") or bool(v.get("main"))
        for line, d in sorted(by_line.items()):
            if "over" in d and "under" in d:
                out.append({"market": m["name"], "line": line, **d})
    return out


def price_snapshot(lines, ht, pre_ft, params):
    priced = []
    for ln in lines:
        p = price_line(params, ht, pre_ft or 0.0, ln["line"])
        ev_o = float(settle_ev(p["pmf"], ht, ln["line"], "over", ln["over"]))
        ev_u = float(settle_ev(p["pmf"], ht, ln["line"], "under", ln["under"]))
        priced.append({**ln, "p_over": p["p_over"], "p_under": p["p_under"],
                       "ev_over": ev_o, "ev_under": ev_u})
    return priced


def capture(fx, ev, params, status_note):
    ht = ht_corners_sportradar(fx["sportradar_id"]) if fx.get("sportradar_id") else None
    pre = pre_match_ft(fx)
    lines = corner_lines(ev)
    rec = {
        "match_id": fx["id"], "kickoff": fx["date"], "home": fx["home"], "away": fx["away"],
        "api_fixture_id": ev["fixture"]["id"], "captured_at": now_utc().isoformat(),
        "status": ev["fixture"]["status"], "capture_note": status_note,
        "score": [ev["teams"]["home"].get("goals"), ev["teams"]["away"].get("goals")],
        "ht_corners": ht, "pre_ft": pre, "params": params,
        "lines": price_snapshot(lines, ht, pre, params) if ht is not None else lines,
    }
    append_log(rec)
    print(f"  LOGGED {fx['home']} v {fx['away']}: HT corners {ht}, {len(lines)} lines ({status_note})")
    if ht is not None:
        for ln in rec["lines"]:
            best = max(("over", ln["ev_over"], ln["over"]), ("under", ln["ev_under"], ln["under"]), key=lambda s: s[1])
            flag = "  <-- value" if best[1] > 0.03 and not ln.get("suspended") else ""
            print(f"    {ln['market']:<14} {ln['line']:>5}  O {ln['over']:.2f} / U {ln['under']:.2f}  "
                  f"P(over) {ln['p_over']*100:4.1f}%  best {best[0]} EV {best[1]*100:+.1f}%{flag}")


# ------------------------------------------------------------------ commands

def cmd_run(date_str, max_calls):
    params = load_params()
    budget = Budget(max_calls)
    team_map = load_team_map()
    fixtures = [fx for fx in fixtures_on(date_str) if fx["id"] not in logged_match_ids()]
    if not fixtures:
        print(f"No un-logged {COMPETITION} fixtures on {date_str}.")
        return
    slots = defaultdict(list)
    for fx in fixtures:
        slots[fx["date"]].append(fx)
    print(f"{len(fixtures)} fixtures in {len(slots)} kickoff slots on {date_str}; budget {max_calls} calls.")

    for kickoff, slot in sorted(slots.items()):
        ko = datetime.fromisoformat(kickoff)
        start, end = ko + timedelta(minutes=WINDOW_START_MIN), ko + timedelta(minutes=WINDOW_END_MIN)
        if now_utc() > end:
            print(f"Slot {kickoff} already past its HT window, skipping.")
            continue
        wait = (start - now_utc()).total_seconds()
        names = ", ".join(f"{fx['home']} v {fx['away']}" for fx in slot)
        print(f"\nSlot {kickoff}: {names}")
        if wait > 0:
            print(f"  sleeping {wait/60:.0f} min until {start.isoformat()}")
            time.sleep(wait)

        pending = {fx["id"]: fx for fx in slot}
        while pending and now_utc() < end:
            try:
                live = fetch_live_odds(budget)
            except RuntimeError as e:
                print(f"  {e}; stopping.")
                save_team_map(team_map)
                return
            pairs = match_live_to_fixtures(live, list(pending.values()), team_map)
            for mid, ev in pairs.items():
                st = ev["fixture"]["status"]
                late = (st.get("elapsed") or 0) >= 47 and "second" in (st.get("long") or "").lower()
                if is_half_time(ev):
                    capture(pending.pop(mid), ev, params, "half-time")
                elif late and now_utc() > end - timedelta(minutes=2 * POLL_SECONDS / 60):
                    capture(pending.pop(mid), ev, params, f"missed HT, 2H min {st.get('elapsed')}")
            save_team_map(team_map)
            if pending and now_utc() < end:
                time.sleep(POLL_SECONDS)
        for fx in pending.values():
            print(f"  not captured: {fx['home']} v {fx['away']}")


def cmd_snapshot():
    params = load_params()
    team_map = load_team_map()
    live = fetch_live_odds(Budget(1))
    if not live:
        print("No live EPL fixtures in /odds/live right now.")
        return
    today = now_utc().strftime("%Y-%m-%d")
    pairs = match_live_to_fixtures(live, fixtures_on(today), team_map)
    save_team_map(team_map)
    fx_by_id = {fx["id"]: fx for fx in fixtures_on(today)}
    for mid, ev in pairs.items():
        capture(fx_by_id[mid], ev, params, f"manual snapshot, {ev['fixture']['status'].get('long')}")


def cmd_evaluate(min_ev):
    try:
        with open(LOG_PATH) as f:
            records = [json.loads(line) for line in f if line.strip()]
    except FileNotFoundError:
        print("No log yet.")
        return
    with get_db() as db:
        corners = defaultdict(lambda: [0, 0])
        for row in db.execute("SELECT match_id, raw_data FROM match_events WHERE event_type='corner_kick'"):
            p = json.loads(row["raw_data"]).get("period")
            if p in (1, 2):
                corners[row["match_id"]][p - 1] += 1

    settled = staked = pnl = hits = 0
    implied_mu2, ht_counts = [], []
    for rec in records:
        if rec["match_id"] not in corners:
            continue
        c1, c2 = corners[rec["match_id"]]
        ht = rec["ht_corners"] if rec["ht_corners"] is not None else c1
        ft = c1 + c2
        params = rec.get("params") or load_params()
        lines = price_snapshot([{k: v for k, v in ln.items() if k in ("market", "line", "over", "under", "suspended", "main")}
                                for ln in rec["lines"]], ht, rec.get("pre_ft"), params)
        settled += 1
        # main Total Corners line -> implied 2H expectation, to test "books extrapolate the 1H"
        main = [ln for ln in lines if ln["market"] == "Total Corners" and ln.get("main")] or \
               [ln for ln in lines if ln["market"] == "Total Corners"]
        if main:
            ln = min(main, key=lambda l: abs(1 / l["over"] - 1 / l["under"]))
            implied_mu2.append(ln["line"] - ht)
            ht_counts.append(ht)
        # one bet per match: the best-EV unsuspended line/side
        cands = [(ln[f"ev_{s}"], s, ln) for ln in lines if not ln.get("suspended") for s in ("over", "under")]
        if not cands:
            continue
        ev, side, ln = max(cands, key=lambda c: c[0])
        if ev < min_ev:
            continue
        odds, line = ln[side], ln["line"]
        frac = round(line % 1, 2)
        parts = (line - 0.25, line + 0.25) if frac in (0.25, 0.75) else (line,)
        ret = 0.0
        for L in parts:
            win = ft > L if side == "over" else ft < L
            push = ft == L
            ret += ((odds - 1) if win else 0 if push else -1) / len(parts)
        staked += 1
        pnl += ret
        hits += ret > 0
        print(f"  {rec['home']} v {rec['away']}: HT {ht} FT {ft} | {ln['market']} {side} {line} @ {odds} "
              f"EV {ev*100:+.1f}% -> {ret:+.2f}")

    print(f"\nSettled snapshots: {settled}/{len(records)}  bets (EV >= {min_ev*100:.0f}%): {staked}")
    if staked:
        print(f"Hit rate {hits/staked*100:.1f}%  P&L {pnl:+.2f} units  ROI {pnl/staked*100:+.1f}%")
    if len(ht_counts) >= 5:
        import numpy as np
        slope = np.polyfit(ht_counts, implied_mu2, 1)[0]
        print(f"Book's implied 2H corners vs HT count: slope {slope:+.3f} per 1H corner "
              f"(0 = no extrapolation, like the data; >0 = book chases 1H pace) over {len(ht_counts)} games")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--date", default=now_utc().strftime("%Y-%m-%d"))
    r.add_argument("--max-calls", type=int, default=40)
    sub.add_parser("snapshot")
    e = sub.add_parser("evaluate")
    e.add_argument("--min-ev", type=float, default=0.03)
    a = ap.parse_args()
    if a.cmd == "run":
        cmd_run(a.date, a.max_calls)
    elif a.cmd == "snapshot":
        cmd_snapshot()
    else:
        cmd_evaluate(a.min_ev)
