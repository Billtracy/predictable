"""Leak-free walk-forward backtest against the betting line.

For every game day, refit on all final games before that day's first tip-off,
predict the day's games, and score them against the DraftKings line:

  * totals / spread MAE: model vs closing line vs opening line
  * edge slope: OLS of (actual - line) on (model - line). 0 = the model adds nothing
    the line doesn't already know; 1 = trust the model fully over the line.
  * tip hit rate vs line (pushes excluded) overall and by |model - line| edge,
    against always-over / always-under and the -110 break-even.

Usage:
    python -m nba.backtest                    # seasons 2023-2026, default params
    python -m nba.backtest 2025 2026          # chosen seasons
    python -m nba.backtest --param half_life=45 --param eff_ridge=5
"""
import sys
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from nba.config import BREAK_EVEN
from nba.db import load_games
from nba.model import DEFAULTS, Design, NBAModel, local_day, rest_context

EVAL_SEASONS = (2023, 2024, 2025, 2026)
EDGES = (0, 2, 4, 6, 8)


def walk_forward(games, seasons=EVAL_SEASONS, design=None, context=None, lag_hours=0, **params):
    context = context or rest_context(games)
    design = design or Design(games, context)
    model = NBAModel(**params)
    by_day = defaultdict(list)
    for g in games:
        if g["season"] in seasons and g["status"] == "FT":
            by_day[(g["season"], local_day(g["date"]))].append(g)

    recs = []
    for (season, _), day_games in sorted(by_day.items(), key=lambda kv: kv[0][1]):
        as_of = min(g["date"] for g in day_games)
        if lag_hours:
            # Opening lines go up before the previous night's games finish; a lag of
            # 24h fits only on what was known when the opener was posted.
            as_of = (datetime.fromisoformat(as_of.replace("Z", "+00:00"))
                     - timedelta(hours=lag_hours)).strftime("%Y-%m-%dT%H:%MZ")
        if design.upto(as_of) < 200:
            continue
        model.fit(design, as_of, season)
        for g in day_games:
            pred = model.predict(g, context[g["id"]])
            rec = {"id": g["id"], "season": season, "type": g["game_type"],
                   "total": g["home_pts"] + g["away_pts"], "margin": g["home_pts"] - g["away_pts"],
                   "reg_total": (g["home_reg"] or 0) + (g["away_reg"] or 0) if g["home_reg"] is not None else None,
                   "ot": int((g["periods"] or 4) > 4),
                   "m_total": pred["total"], "m_reg_total": pred["reg_total"], "m_margin": pred["margin"],
                   "p_ot": pred["p_ot"],
                   "line_total": g.get("line_total"), "line_spread": g.get("line_spread"),
                   "open_total": g.get("line_total_open"), "open_spread": g.get("line_spread_open")}
            if rec["line_total"] is not None:
                rec["p_over"] = model.p_over(pred, rec["line_total"])
            if rec["line_spread"] is not None:
                rec["p_cover"] = model.p_home_cover(pred, rec["line_spread"])
            recs.append(rec)
    return recs


# ------------------------------------------------------------------ reporting

def _mae(xs):
    return float(np.mean(np.abs(xs))) if len(xs) else float("nan")


def slope(actual, line, model):
    """OLS slope of (actual - line) on (model - line), with its standard error."""
    x = np.asarray(model) - np.asarray(line)
    y = np.asarray(actual) - np.asarray(line)
    x = x - x.mean()
    b = float((x @ (y - y.mean())) / (x @ x))
    resid = (y - y.mean()) - b * x
    se = float(np.sqrt((resid @ resid) / (len(x) - 2) / (x @ x)))
    return b, se


def _fmt_rate(w, l):
    n = w + l
    if not n:
        return "     -        "
    p = w / n
    se = (p * (1 - p) / n) ** 0.5
    return f"{100 * p:5.1f}% ±{100 * se:3.1f} n={n:<5d}"


def total_report(recs, line_key, label):
    rows = [r for r in recs if r.get(line_key) is not None]
    if not rows:
        return
    a = np.array([r["total"] for r in rows])
    ln = np.array([r[line_key] for r in rows])
    md = np.array([r["m_total"] for r in rows])
    b, se = slope(a, ln, md)
    print(f"\n  TOTALS vs {label}  ({len(rows)} games)")
    print(f"    MAE  model {_mae(a - md):5.2f}   line {_mae(a - ln):5.2f}   "
          f"bias model {np.mean(md - a):+5.2f}  line {np.mean(ln - a):+5.2f}")
    print(f"    edge slope {b:+.3f} ± {se:.3f}   (0 = no info beyond the line)")
    print(f"    always over  {_fmt_rate(*_count(rows, line_key, lambda r: 'over'))}")
    print(f"    always under {_fmt_rate(*_count(rows, line_key, lambda r: 'under'))}")
    for e in EDGES:
        rule = lambda r, e=e: ("over" if r["m_total"] > r[line_key] else "under") \
            if abs(r["m_total"] - r[line_key]) >= max(e, 1e-9) else None
        print(f"    model |edge|>={e:<2d}  {_fmt_rate(*_count(rows, line_key, rule))}")


def _count(rows, line_key, rule):
    w = l = 0
    for r in rows:
        side = rule(r)
        if side is None or r["total"] == r[line_key]:
            continue
        over = r["total"] > r[line_key]
        hit = over if side == "over" else not over
        w += hit
        l += not hit
    return w, l


def spread_report(recs, line_key, label):
    rows = [r for r in recs if r.get(line_key) is not None]
    if not rows:
        return
    a = np.array([r["margin"] for r in rows])
    ln = np.array([-r[line_key] for r in rows])   # line's implied home margin
    md = np.array([r["m_margin"] for r in rows])
    b, se = slope(a, ln, md)
    print(f"\n  SPREAD vs {label}  ({len(rows)} games)")
    print(f"    MAE  model {_mae(a - md):5.2f}   line {_mae(a - ln):5.2f}")
    print(f"    edge slope {b:+.3f} ± {se:.3f}")
    for e in EDGES:
        w = l = 0
        for r, act, imp, m in zip(rows, a, ln, md):
            if act == imp or abs(m - imp) < max(e, 1e-9):
                continue
            hit = (act > imp) == (m > imp)
            w += hit
            l += not hit
        print(f"    model |edge|>={e:<2d}  {_fmt_rate(w, l)}")


def calibration(recs):
    rows = [r for r in recs if r.get("p_over") is not None and r["total"] != r["line_total"]]
    if not rows:
        return
    p = np.array([r["p_over"] for r in rows])
    y = np.array([r["total"] > r["line_total"] for r in rows], float)
    print(f"\n  P(over) Brier {np.mean((p - y) ** 2):.4f}  (coin flip 0.2500)")
    reg = [r for r in recs if r["reg_total"] is not None]
    rt = np.array([r["reg_total"] - r["m_reg_total"] for r in reg])
    mg = np.array([r["margin"] - r["m_margin"] for r in recs])
    print(f"  residual SD: 48-min total {rt.std():.2f}  margin {mg.std():.2f}   "
          f"(model uses {DEFAULTS['sd_total']}/{DEFAULTS['sd_margin']} unless overridden)")
    print(f"  overtime: actual {np.mean([r['ot'] for r in recs]):.3f}  "
          f"predicted {np.mean([r['p_ot'] for r in recs]):.3f}")


def report(recs, title=""):
    print(f"\n=== {title or 'NBA walk-forward'}: {len(recs)} games ===")
    total_report(recs, "line_total", "CLOSING line")
    total_report(recs, "open_total", "OPENING line")
    spread_report(recs, "line_spread", "CLOSING line")
    spread_report(recs, "open_spread", "OPENING line")
    calibration(recs)
    print(f"\n  break-even at -110: {100 * BREAK_EVEN:.1f}%")


def _parse(argv):
    seasons, params = [], {}
    it = iter(argv)
    for a in it:
        if a == "--param":
            k, v = next(it).split("=")
            params[k] = float(v)
        elif a == "--lag":
            params["lag_hours"] = float(next(it))
        else:
            seasons.append(int(a))
    return tuple(seasons) or EVAL_SEASONS, params


def main(argv):
    seasons, params = _parse(argv)
    games = load_games()
    recs = walk_forward(games, seasons, **params)
    report(recs, f"seasons {', '.join(map(str, seasons))}  params {params or 'default'}")
    for s in seasons:
        sub = [r for r in recs if r["season"] == s]
        if sub:
            rows = [r for r in sub if r.get("line_total") is not None]
            w, l = _count(rows, "line_total", lambda r: "over" if r["m_total"] > r["line_total"] else "under")
            print(f"  {s}: totals tips vs close {_fmt_rate(w, l)}")


if __name__ == "__main__":
    main(sys.argv[1:])
