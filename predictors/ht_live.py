"""On-demand live HT corner tip: the 'Get live tip' button on an EPL match card.

A click asks the server for the match's live state. The server reads the
Sportradar timeline (cached TIMELINE_TTL seconds per match, so a crowd of
clicks costs one API call a minute), and at half-time prices the rest of the
match with predictors/ht_corners.py.

The first HT tip for a match is frozen in STORE_PATH and served to every later
click, so the published HT record can never change after the fact.
STORE_PATH lives in runtime/, a volume that survives redeploys.
"""

import json
import os
import threading
import time
from datetime import datetime, timedelta, timezone

from predictors.ht_corners import PARAMS, ht_corner_tip, tip_probability_at_ht

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE_PATH = os.environ.get('HT_TIPS_PATH', os.path.join(ROOT, 'runtime', 'ht_tips_live.json'))
TIMELINE_TTL = 60
LIVE_WINDOW = timedelta(minutes=150)  # kickoff .. kickoff + 150 min counts as "in play"

_store_lock = threading.Lock()
_store_cache = {'mtime': None, 'tips': {}}
_timeline_cache = {}   # sportradar id -> (fetched_at, json)
_fetch_locks = {}      # sportradar id -> Lock, so concurrent clicks share one fetch
_fetch_locks_lock = threading.Lock()


# ------------------------------------------------------------------ frozen store

def load_live_tips():
    """{match_id (str): frozen tip}, re-read only when the file changes."""
    try:
        mtime = os.path.getmtime(STORE_PATH)
    except OSError:
        return {}
    with _store_lock:
        if _store_cache['mtime'] != mtime:
            with open(STORE_PATH) as f:
                _store_cache['tips'], _store_cache['mtime'] = json.load(f), mtime
        return _store_cache['tips']


def store_version():
    try:
        return os.path.getmtime(STORE_PATH)
    except OSError:
        return None


def _freeze(match_id, record):
    """Save a tip unless one is already frozen; return the frozen one."""
    with _store_lock:
        tips = {}
        if os.path.exists(STORE_PATH):
            with open(STORE_PATH) as f:
                tips = json.load(f)
        if str(match_id) not in tips:
            tips[str(match_id)] = record
            os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)
            tmp = STORE_PATH + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(tips, f, indent=1)
            os.replace(tmp, STORE_PATH)
        return tips[str(match_id)]


# ------------------------------------------------------------------ live state

def _default_fetch(sr_event_id):
    from data.sportradar import fetch_match_timeline
    return fetch_match_timeline(sr_event_id)


def _timeline(sr_event_id, fetch):
    with _fetch_locks_lock:
        lock = _fetch_locks.setdefault(sr_event_id, threading.Lock())
    with lock:
        hit = _timeline_cache.get(sr_event_id)
        if hit and time.time() - hit[0] < TIMELINE_TTL:
            return hit[1]
        data = fetch(sr_event_id)
        if data:
            _timeline_cache[sr_event_id] = (time.time(), data)
        return data


def read_timeline_state(timeline_json):
    """(state, first-half corners, (home, away) score, clock) where state is
    'first_half' | 'halftime' | 'second_half' | 'ended' | 'not_started' | 'unknown'."""
    status = timeline_json.get('sport_event_status') or {}
    match_status = (status.get('match_status') or '').lower()
    events = timeline_json.get('timeline', [])
    types = {e.get('type') for e in events}
    second_started = any(e.get('type') == 'period_start' and e.get('period') == 2 for e in events)
    if match_status in ('ended', 'closed') or 'match_ended' in types:
        state = 'ended'
    elif match_status == 'halftime' or ('break_start' in types and not second_started):
        state = 'halftime'
    elif second_started or match_status == '2nd_half':
        state = 'second_half'
    elif match_status == '1st_half' or 'match_started' in types:
        state = 'first_half'
    elif match_status in ('not_started', '') and not events:
        state = 'not_started'
    else:
        state = 'unknown'
    corners = sum(1 for e in events if e.get('type') == 'corner_kick' and e.get('period') == 1)
    score = [status.get('home_score'), status.get('away_score')]
    clock = (status.get('clock') or {}).get('played')
    return state, corners, score, clock


def in_live_window(kickoff_iso, now=None):
    now = now or datetime.now(timezone.utc)
    kickoff = datetime.fromisoformat(kickoff_iso.replace('Z', '+00:00'))
    return kickoff <= now <= kickoff + LIVE_WINDOW


def live_tip(match, pre_tip=None, fetch=_default_fetch, now=None):
    """Response for the live-tip button. `match` needs id, date, sportradar_id."""
    now = now or datetime.now(timezone.utc)

    def with_pre(payload):
        if pre_tip and payload.get('ht_corners') is not None:
            payload['pre_tip'] = pre_tip
            payload['pre_tip_prob'] = round(100 * tip_probability_at_ht(pre_tip, payload['ht_corners']), 1)
        return payload

    frozen = load_live_tips().get(str(match['id']))
    if frozen:
        return with_pre({'state': 'tip', 'frozen': True, **frozen})
    if not in_live_window(match['date'], now):
        return {'state': 'not_live', 'message': 'Live tips are available while the match is in play.'}
    if not match.get('sportradar_id'):
        return {'state': 'unavailable', 'message': 'No live feed for this match.'}

    data = _timeline(match['sportradar_id'], fetch)
    if not data:
        return {'state': 'unavailable', 'message': 'Live feed unavailable right now. Try again in a minute.'}
    state, corners, score, clock = read_timeline_state(data)

    if state == 'halftime':
        record = {**ht_corner_tip(corners), 'ht_score': score,
                  'captured_at': now.isoformat(), 'params': PARAMS}
        return with_pre({'state': 'tip', 'frozen': False, **_freeze(match['id'], record)})
    messages = {
        'not_started': 'Not kicked off yet. The tip unlocks at half-time.',
        'first_half': 'First half in play. The tip unlocks at half-time.',
        'second_half': 'Second half under way. The half-time tip window has closed.',
        'ended': 'Match finished.',
        'unknown': 'Live state not clear yet. Try again in a minute.',
    }
    return {'state': state, 'message': messages[state], 'score': score, 'clock': clock,
            'first_half_corners': corners if state == 'first_half' else None}
