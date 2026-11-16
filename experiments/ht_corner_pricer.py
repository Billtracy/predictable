"""Half-time corner pricer: P(full-time total corners over/under a live line),
priced at the half-time whistle.

Model
-----
    FT corners = HT corners (known) + C2,   C2 ~ NegBin(mean mu2, variance phi * mu2)
    mu2        = a + b * pre_ft             (pre_ft = fitted pre-match expected FT corners)

The data showed second-half corners are essentially independent of the first half
(corr(1H, 2H) ~ 0, flat across HT score lines), so the first half only enters as the
already-banked count. Optional game-state terms are fitted and reported so that
finding is re-checked every time the params are refit.

Usage
-----
  python experiments/ht_corner_pricer.py fit        # walk-forward backtest + write params
  python experiments/ht_corner_pricer.py price HT PRE_FT LINE [OVER_ODDS UNDER_ODDS]
      e.g. python experiments/ht_corner_pricer.py price 6 10.2 11.5 1.85 1.95

Importable: load_params(), second_half_dist(), price_line(), settle_ev().
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import nbinom, poisson

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT)

PARAMS_PATH = os.path.join(ROOT, "data", "ht_corner_params.json")
MAX_C2 = 30
TRAIN_UNTIL = "2025-08-01"  # walk-forward split: fit on epl_2024, test on epl_2025+


# --------------------------------------------------------------------- pricing

def load_params(path=PARAMS_PATH):
    with open(path) as f:
        return json.load(f)


def second_half_mean(params, pre_ft):
    return max(0.5, params["a"] + params["b"] * pre_ft)


def second_half_pmf(mu2, phi):
    k = np.arange(MAX_C2 + 1)
    if phi > 1.0:
        pmf = nbinom(mu2 / (phi - 1.0), 1.0 / phi).pmf(k)
    else:
        pmf = poisson(mu2).pmf(k)
    return pmf / pmf.sum()


def _single_line_ev(pmf, ht, line, side, odds):
    """EV per unit stake for a half or whole line (whole lines push)."""
    totals = ht + np.arange(len(pmf))
    if side == "over":
        win, push = totals > line, totals == line
    else:
        win, push = totals < line, totals == line
    p_win, p_push = pmf[win].sum(), pmf[push].sum()
    p_lose = 1.0 - p_win - p_push
    return p_win * (odds - 1.0) - p_lose


def settle_ev(pmf, ht, line, side, odds):
    """EV per unit stake, handling Asian quarter lines (10.25 = half on 10, half on 10.5)."""
    frac = round(line % 1, 2)
    if frac in (0.25, 0.75):
        return 0.5 * (_single_line_ev(pmf, ht, line - 0.25, side, odds)
                      + _single_line_ev(pmf, ht, line + 0.25, side, odds))
    return _single_line_ev(pmf, ht, line, side, odds)


def price_line(params, ht_corners, pre_ft, line):
    """Probabilities and fair odds for FT total corners vs `line`, given the HT count."""
    pmf = second_half_pmf(second_half_mean(params, pre_ft), params["phi"])
    totals = ht_corners + np.arange(len(pmf))
    p_over = float(pmf[totals > line].sum())
    p_under = float(pmf[totals < line].sum())
    p_push = max(0.0, 1.0 - p_over - p_under)
    fair = lambda p: (1.0 - p_push) / p if p > 0 else float("inf")
    return {
        "mu2": second_half_mean(params, pre_ft),
        "expected_ft": ht_corners + second_half_mean(params, pre_ft),
        "p_over": p_over, "p_under": p_under, "p_push": p_push,
        "fair_over": fair(p_over), "fair_under": fair(p_under),
        "pmf": pmf,
    }


# --------------------------------------------------------------------- dataset

def build_dataset():
    """One row per EPL match with per-half corner counts from Sportradar events
    and a leak-free pre-match expected FT corner total from the fitted model."""
    from db.database import get_db
    from models import fitted

    with get_db() as db:
        matches = db.execute("""
            SELECT id, competition_id, date, home_team_id, away_team_id
            FROM matches
            WHERE competition_id LIKE 'epl_%' AND status IN ('FT', 'AET', 'PEN')
            ORDER BY date
        """).fetchall()
        events = db.execute("""
            SELECT e.match_id, e.event_type, e.raw_data
            FROM match_events e JOIN matches m ON m.id = e.match_id
            WHERE m.competition_id LIKE 'epl_%'
              AND e.event_type IN ('corner_kick', 'score_change')
        """).fetchall()

    per = defaultdict(lambda: {"c1": 0, "c2": 0, "gh": 0, "ga": 0})
    for row in events:
        r = json.loads(row["raw_data"])
        d, period = per[row["match_id"]], r.get("period")
        if row["event_type"] == "corner_kick":
            if period == 1:
                d["c1"] += 1
            elif period == 2:
                d["c2"] += 1
        elif period == 1:
            d["gh"] = r.get("home_score", d["gh"])
            d["ga"] = r.get("away_score", d["ga"])

    rows = []
    for m in matches:
        if m["id"] not in per:
            continue
        pred = fitted.predict_corners(m["competition_id"], m["home_team_id"],
                                      m["away_team_id"], m["date"])
        if pred is None:
            continue
        d = per[m["id"]]
        rows.append({
            "match_id": m["id"], "date": m["date"],
            "pre_ft": pred["home_expected"] + pred["away_expected"],
            "c1": d["c1"], "c2": d["c2"], "gd": d["gh"] - d["ga"],
        })
    return rows


# --------------------------------------------------------------------- fitting

def fit_params(rows, use_pre=True):
    """use_pre=False is the flat model: every match gets the league 2H mean."""
    pre = np.array([r["pre_ft"] for r in rows])
    c2 = np.array([r["c2"] for r in rows], dtype=float)
    if use_pre:
        A = np.c_[np.ones_like(pre), pre]
        (a, b), *_ = np.linalg.lstsq(A, c2, rcond=None)
    else:
        a, b = c2.mean(), 0.0
    mu = np.maximum(0.5, a + b * pre)
    phi = float(max(1.0, np.mean((c2 - mu) ** 2 / mu)))
    return {"a": float(a), "b": float(b), "phi": phi, "n": len(rows),
            "model": "pre_ft" if use_pre else "flat"}


def game_state_check(rows):
    """OLS of 2H corners on pre_ft + 1H corners + HT game state, for the report."""
    pre = np.array([r["pre_ft"] for r in rows])
    c1 = np.array([r["c1"] for r in rows], dtype=float)
    gd = np.array([r["gd"] for r in rows], dtype=float)
    c2 = np.array([r["c2"] for r in rows], dtype=float)
    X = np.c_[np.ones_like(pre), pre, c1, (gd == 0), np.abs(gd) >= 2]
    coef, *_ = np.linalg.lstsq(X, c2, rcond=None)
    resid = c2 - X @ coef
    se = np.sqrt(np.diag(np.var(resid) * np.linalg.inv(X.T @ X)))
    names = ["intercept", "pre_ft", "1H_corners", "HT_level", "HT_margin>=2"]
    return [(n, float(c), float(s)) for n, c, s in zip(names, coef, se)]


def _logloss(p, y):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def backtest(train, test, params, flat):
    """Score our P(over) on 'HT + k' half-lines vs two naive books, and simulate
    betting into a hypothetical book that extrapolates first-half pace."""
    mean1 = np.mean([r["c1"] for r in train])
    mean2 = np.mean([r["c2"] for r in train])
    out = {"lines": {}, "extrapolating_book": {}}

    for k in (3.5, 4.5, 5.5, 6.5):
        y, p_ours, p_flat = [], [], []
        for r in test:
            line = r["c1"] + k
            y.append(r["c1"] + r["c2"] > line)
            p_ours.append(price_line(params, r["c1"], r["pre_ft"], line)["p_over"])
            p_flat.append(price_line(flat, r["c1"], r["pre_ft"], line)["p_over"])
        y, p_ours, p_flat = map(np.array, (y, p_ours, p_flat))
        pick_over = p_ours > 0.5
        out["lines"][k] = {
            "over_rate": float(y.mean()),
            "logloss_ours": _logloss(p_ours, y),
            "logloss_flat": _logloss(p_flat, y),
            "pick_hit_rate": float(np.mean(np.where(pick_over, y, ~y))),
            "share_picked_over": float(pick_over.mean()),
        }

    # Hypothetical book: 2H mean = blend of pro-rata pre-match and 1H pace
    # extrapolation, priced with a 6% overround, line set at its own median.
    for w in (0.25, 0.5):
        staked, pnl, bets = 0.0, 0.0, 0
        for r in test:
            book_mu2 = (1 - w) * second_half_mean(params, r["pre_ft"]) + w * r["c1"] * (mean2 / mean1)
            book_pmf = second_half_pmf(max(0.5, book_mu2), params["phi"])
            k = np.searchsorted(np.cumsum(book_pmf), 0.5) + 0.5  # book's 2H median line
            bp_over = book_pmf[np.arange(MAX_C2 + 1) > k].sum()
            odds_over = 1.0 / (bp_over * 1.06)
            odds_under = 1.0 / ((1 - bp_over) * 1.06)
            ours = second_half_pmf(second_half_mean(flat, r["pre_ft"]), flat["phi"])
            best = max(("over", odds_over), ("under", odds_under),
                       key=lambda s: settle_ev(ours, 0, k, s[0], s[1]))
            if settle_ev(ours, 0, k, *best) > 0.03:
                won = r["c2"] > k if best[0] == "over" else r["c2"] < k
                staked += 1
                pnl += (best[1] - 1) if won else -1
                bets += 1
        out["extrapolating_book"][w] = {"bets": bets, "of": len(test),
                                        "roi": pnl / staked if staked else 0.0}
    return out


def cmd_fit():
    rows = build_dataset()
    train = [r for r in rows if r["date"] < TRAIN_UNTIL]
    test = [r for r in rows if r["date"] >= TRAIN_UNTIL]
    print(f"Dataset: {len(rows)} matches (train {len(train)}, test {len(test)})")

    p_train = fit_params(train)
    f_train = fit_params(train, use_pre=False)
    print(f"\nTrain fit (pre_ft): mu2 = {p_train['a']:.2f} + {p_train['b']:.3f}*pre_ft, phi = {p_train['phi']:.2f}")
    print(f"Train fit (flat):   mu2 = {f_train['a']:.2f}, phi = {f_train['phi']:.2f}")

    print("\nGame-state check (2H corners OLS, all data):  coef ± se")
    for name, c, s in game_state_check(rows):
        print(f"  {name:<14} {c:+.3f} ± {s:.3f}")

    res = backtest(train, test, p_train, f_train)
    print("\nOut-of-sample, line = HT corners + k (FT over/under):")
    print("  k     over%   logloss pre_ft  flat      pick hit%  picked-over%  (picks from pre_ft model)")
    for k, v in res["lines"].items():
        print(f"  {k:<5} {v['over_rate']*100:5.1f}   {v['logloss_ours']:.4f}       {v['logloss_flat']:.4f}"
              f"      {v['pick_hit_rate']*100:5.1f}     {v['share_picked_over']*100:5.1f}")
    print("\nHypothetical book that extrapolates 1H pace (6% margin), flat model bets when EV > 3%:")
    for w, v in res["extrapolating_book"].items():
        print(f"  extrapolation weight {w}: {v['bets']}/{v['of']} bets, ROI {v['roi']*100:+.1f}%")

    pre_wins = sum(v["logloss_ours"] < v["logloss_flat"] for v in res["lines"].values()) > len(res["lines"]) / 2
    print(f"\nOut-of-sample winner: {'pre_ft' if pre_wins else 'flat'} model")
    final = fit_params(rows, use_pre=pre_wins)
    final["fitted_on"] = f"{rows[0]['date'][:10]}..{rows[-1]['date'][:10]}"
    with open(PARAMS_PATH, "w") as f:
        json.dump(final, f, indent=2)
    print(f"Final params (all {final['n']} matches, {final['model']}): mu2 = {final['a']:.2f} + {final['b']:.3f}*pre_ft, "
          f"phi = {final['phi']:.2f}  -> {os.path.relpath(PARAMS_PATH, ROOT)}")


def cmd_price(argv):
    ht, pre_ft, line = int(argv[0]), float(argv[1]), float(argv[2])
    params = load_params()
    p = price_line(params, ht, pre_ft, line)
    print(f"HT {ht}, pre-match FT exp {pre_ft:.2f} -> 2H exp {p['mu2']:.2f}, FT exp {p['expected_ft']:.2f}")
    print(f"Line {line}: P(over) {p['p_over']*100:.1f}%  P(under) {p['p_under']*100:.1f}%  push {p['p_push']*100:.1f}%")
    print(f"Fair odds: over {p['fair_over']:.2f}  under {p['fair_under']:.2f}")
    if len(argv) >= 5:
        for side, odds in (("over", float(argv[3])), ("under", float(argv[4]))):
            print(f"  {side} @ {odds}: EV {settle_ev(p['pmf'], ht, line, side, odds)*100:+.1f}%")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "fit":
        cmd_fit()
    elif len(sys.argv) >= 5 and sys.argv[1] == "price":
        cmd_price(sys.argv[2:])
    else:
        print(__doc__)
