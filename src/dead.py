"""Step 2 - dead ball: per-frame log-odds of "D" (positive = dead).

Each rule adds or removes evidence. To try a new rule: add it here with its parameter in
config.PARAMS and judge it on the external bench (python -m bench.run), never on the public
5 minutes.
"""

import numpy as np

from . import config as C
from .ball import nan_runs


def _run_length_at_least(mask, n):
    """True on the frames that belong to a True run of length >= n."""
    out = np.zeros_like(mask)
    s, e = nan_runs(mask)
    for a, b in zip(s, e):
        if b - a >= n:
            out[a:b] = True
    return out


def dead_logit(f, tags, p=C.PARAMS, pitch=None):
    """pitch = (length, width) in metres; None = the challenge match (config.PITCH)."""
    n = len(f["ball_xy"])
    L, W = C.PITCH if pitch is None else pitch
    z = np.full(n, p["dead_prior_logit"], dtype=np.float64)

    # 1) ball off the pitch
    x, y = f["ball_xy"][:, 0], f["ball_xy"][:, 1]
    m = p["out_margin_m"]
    with np.errstate(invalid="ignore"):
        outside = (x < -m) | (x > L + m) | (y < -m) | (y > W + m)
    z[outside] += 4.0

    # 2) pace of the players
    ps = np.nan_to_num(f["player_speed"], nan=p["fast_players_mps"])
    z[ps < p["slow_players_mps"]] += 2.0
    z[(ps >= p["slow_players_mps"]) & (ps < 1.4)] += 1.0
    z[ps > p["fast_players_mps"]] -= 1.0

    # 3) ball standing still for a while
    with np.errstate(invalid="ignore"):
        still = f["ball_seen"] & (f["ball_speed"] < p["still_ball_mps"])
    z[_run_length_at_least(still, int(p["still_ball_s"] * C.FPS))] += 1.5

    # 4) ball lost for a long time
    z[_run_length_at_least(~f["ball_raw_seen"], int(p["long_gap_s"] * C.FPS))] += 0.5

    # 5) Smart Tagging: the first part of a SET PIECES tag is the set-up of the restart
    tail = int(p["set_piece_tail_s"] * C.FPS)
    for t in tags:
        if t["code"] == "SET PIECES":
            z[max(t["start"], 0):max(t["end"] - tail, 0)] += 1.0
    return z
