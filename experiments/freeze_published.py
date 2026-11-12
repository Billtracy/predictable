"""Freeze the EPL predictions a deployed version of the site showed for played matches.

The site does not log its tips: for a played match it re-predicts from data before
kickoff, so a model change would silently rewrite the published record. Run this
against a checkout of the commit that was live BEFORE the change, and the API
serves the frozen predictions for those matches from then on (app.py,
_published_fixture).

    git worktree add --detach /tmp/live <previously deployed commit>
    venv/bin/python experiments/freeze_published.py /tmp/live
    git worktree remove /tmp/live

Append-only: matches already in data/published_predictions.json are never
overwritten, so the record of an older model survives later freezes. Only
matches that have kicked off are frozen; upcoming ones follow the live model.
"""
import json
import os
import sys
import threading
from datetime import datetime, timezone

OUTPUT_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "published_predictions.json")


def snapshot(checkout):
    """{match_id: entry} for every played match, built by the checkout's own app.py."""
    checkout = os.path.abspath(checkout)
    # Import the checkout's modules, not this repo's.
    sys.path[:] = [checkout] + [p for p in sys.path if os.path.abspath(p or ".") != os.path.dirname(os.path.abspath(__file__))]
    os.chdir(checkout)
    threading.Thread.start = lambda self: None  # skip app.py's cache warm-up thread
    import numpy as np
    import app

    strategies = list(app.CORNER_STRATEGIES)
    matches = {}
    for gw in app._played_gameweeks():
        for strategy in strategies:
            np.random.seed(0)  # the old engine is Monte Carlo: fix one representative draw
            payload = app._build_gameweek(gw, strategy)
            for f in payload["fixtures"]:
                if f["status"] not in app.FINISHED:
                    continue
                entry = matches.setdefault(str(f["match_id"]), {
                    "gameweek": gw,
                    "goals": {k: v for k, v in f["goals"].items() if k != "bet_won"},
                    "corners": {k: v for k, v in f["corners"].items() if k not in ("bet_won", "recommended_bet")},
                    "corner_tips": {},
                })
                entry["corner_tips"][strategy] = f["corners"]["recommended_bet"]
    return matches, app.DEFAULT_CORNER_STRATEGY


def main():
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    checkout = sys.argv[1]
    commit = os.popen(f"git -C '{checkout}' rev-parse --short HEAD").read().strip() or "unknown"

    existing = {"versions": [], "matches": {}}
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE) as fh:
            existing = json.load(fh)

    matches, default_strategy = snapshot(checkout)
    added = {mid: {**entry, "version": commit} for mid, entry in matches.items() if mid not in existing["matches"]}
    existing["matches"].update(added)
    if added:
        existing["versions"].append({
            "commit": commit,
            "default_corner_strategy": default_strategy,
            "corner_strategies": sorted({s for e in added.values() for s in e["corner_tips"]}),
            "matches": len(added),
            "frozen_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })

    with open(OUTPUT_FILE, "w") as fh:
        json.dump(existing, fh, indent=1, sort_keys=True)
    print(f"Froze {len(added)} new matches from {commit} ({len(matches) - len(added)} already frozen) -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
