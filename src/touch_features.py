"""Explicit ball-touch detection and touch features (layer 1 of the v4 model, src/model.py).

Validated on the external bench (bench/v4.py, docs/BENCH_V4.md, docs/BENCH_V4_FIX.md). A touch
candidate is a sharp change of the ball's velocity vector with a player close to the ball. Everything is computed from the tracking alone: no possession
label, no event, nothing from the challenge match. The thresholds are set from physics and the
tracking noise, before any experiment, and are never tuned:

  - velocities are taken on the RAW ball (never on interpolated stretches), lightly smoothed
    (centred mean over SMOOTH frames) and differentiated over K frames on each side; a candidate
    needs every frame of [t - K - 1, t + K + 1] observed, so the edges of a gap (where an
    interpolation would show a fake kink) can never produce one;
  - with ~0.2 m of position noise, the change of velocity measured this way has a noise of
    ~1.4 m/s per axis; MIN_DV = 4 m/s is ~3 sigma, while a pass, a reception or a dribbling
    touch changes the ball's velocity by 3-25 m/s;
  - a player must be within RADIUS_M of the ball (control radius 2 m plus noise), otherwise the
    kink is a bounce or noise; the touch goes to the team of the nearest player (exact ties: no
    team);
  - candidates closer than NMS frames are one touch (the strongest is kept).

Features (per frame) follow the naming of src/frame_features.py so that swap_teams() handles
them: "x_" features change sign when the teams are swapped, "g_" features do not.
"""

import warnings

import numpy as np
import pandas as pd

FPS = 25
SMOOTH = 3
K = 4
MIN_DV = 4.0
RADIUS_M = 2.5
NMS = 8
CAP_S = 30.0


def _velocity_change(ball):
    """(dv (N,), smoothed ball (N,2)). NaN wherever the window is not fully observed."""
    b = pd.DataFrame(np.asarray(ball, dtype=np.float64))
    bs = b.rolling(SMOOTH, center=True, min_periods=SMOOTH).mean().to_numpy()
    n = len(bs)
    vin = np.full((n, 2), np.nan)
    vout = np.full((n, 2), np.nan)
    vin[K:] = (bs[K:] - bs[:-K]) * (FPS / K)
    vout[:-K] = (bs[K:] - bs[:-K]) * (FPS / K)
    return np.linalg.norm(vout - vin, axis=1), bs


def detect(atd):
    """Touches from the tracking only: dict of arrays frame, team (+1/-1/0), dv (m/s), dist (m)."""
    ball = np.asarray(atd["ball"], dtype=np.float64)
    players = np.asarray(atd["players"], dtype=np.float64)
    team = np.asarray(atd["team"])
    dv, bs = _velocity_change(ball)
    cand = np.flatnonzero(np.nan_to_num(dv, nan=0.0) >= MIN_DV)
    empty = {"frame": np.zeros(0, int), "team": np.zeros(0, np.int8),
             "dv": np.zeros(0), "dist": np.zeros(0)}
    if not len(cand):
        return empty
    d = np.linalg.norm(players[cand] - bs[cand][:, None, :], axis=2)          # (C, S)
    dh = np.where(team[None, :] > 0, d, np.nan)
    da = np.where(team[None, :] < 0, d, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN rows -> NaN, silently
        mh = np.nanmin(dh, axis=1)
        ma = np.nanmin(da, axis=1)
    mh = np.where(np.isnan(mh), np.inf, mh)
    ma = np.where(np.isnan(ma), np.inf, ma)
    near = np.minimum(mh, ma)
    ok = near <= RADIUS_M
    cand, mh, ma, near = cand[ok], mh[ok], ma[ok], near[ok]
    side = np.where(mh < ma, 1, np.where(ma < mh, -1, 0)).astype(np.int8)
    strength = dv[cand]
    # non-maximum suppression: strongest first, deterministic tie-break on the frame
    order = np.lexsort((cand, -strength))
    taken = np.zeros(len(ball), dtype=bool)            # frames of the touches kept so far
    keep = []
    for i in order:
        f = cand[i]
        if taken[max(f - NMS, 0):f + NMS + 1].any():
            continue
        taken[f] = True
        keep.append(i)
    keep = np.array(keep, dtype=int)
    keep = keep[np.argsort(cand[keep], kind="stable")]
    return {"frame": cand[keep], "team": side[keep], "dv": np.minimum(strength[keep], 40.0),
            "dist": near[keep]}


NAMES = ["x_tlast_team", "x_tnext_team", "x_tbal_prev50", "x_tbal_next50", "x_tbal_prev125",
         "x_tbal_next125", "g_tsince", "g_tuntil", "g_tagree", "g_tprev_switch",
         "g_tsince_switch", "g_tn125", "g_tlast_dv", "g_tnext_dv", "g_tlast_dist",
         "g_tnext_dist", "g_tvis_prev50", "g_tvis_next50"]


def features(atd, touches=None):
    """(N, len(NAMES)) float32 touch features and NAMES."""
    t = detect(atd) if touches is None else touches
    n = len(atd["ball"])
    fr, tm = t["frame"], t["team"].astype(np.float64)
    idx = np.arange(n)
    nt = len(fr)
    # last touch at or before each frame, next touch strictly after it
    last = np.searchsorted(fr, idx, side="right") - 1
    nxt = np.searchsorted(fr, idx, side="right")
    has_last = last >= 0
    has_next = nxt < nt
    li = np.clip(last, 0, max(nt - 1, 0))
    ni = np.clip(nxt, 0, max(nt - 1, 0))

    def pick(arr, which, has, fill):
        if nt == 0:
            return np.full(n, fill, dtype=np.float64)
        return np.where(has, arr[which], fill).astype(np.float64)

    last_team = pick(tm, li, has_last, 0.0)
    next_team = pick(tm, ni, has_next, 0.0)
    since = np.where(has_last, (idx - pick(fr, li, has_last, 0)) / FPS, CAP_S)
    until = np.where(has_next, (pick(fr, ni, has_next, 0) - idx) / FPS, CAP_S)
    prev_team = pick(tm, np.clip(last - 1, 0, max(nt - 1, 0)), last >= 1, 0.0)

    # time since the last change of team between two consecutive assigned touches
    assigned = tm != 0
    fa, ta = fr[assigned], tm[assigned]
    sw_frames = fa[1:][ta[1:] != ta[:-1]] if len(fa) > 1 else np.zeros(0, int)
    since_sw = np.full(n, 2 * CAP_S)
    if len(sw_frames):
        k = np.searchsorted(sw_frames, idx, side="right") - 1
        since_sw = np.where(k >= 0, (idx - sw_frames[np.maximum(k, 0)]) / FPS, 2 * CAP_S)

    imp_team = np.zeros(n)
    imp_one = np.zeros(n)
    if nt:
        np.add.at(imp_team, fr, tm)
        np.add.at(imp_one, fr, 1.0)
    ct = np.concatenate(([0.0], np.cumsum(imp_team)))       # ct[i] = sum over frames < i
    co = np.concatenate(([0.0], np.cumsum(imp_one)))

    def window(c, a, b):
        """sum over frames in [idx + a, idx + b)."""
        lo = np.clip(idx + a, 0, n)
        hi = np.clip(idx + b, 0, n)
        return c[hi] - c[lo]

    seen = (~np.isnan(np.asarray(atd["ball"])[:, 0])).astype(np.float64)
    cs = np.concatenate(([0.0], np.cumsum(seen)))

    cols = {
        "x_tlast_team": last_team,
        "x_tnext_team": next_team,
        "x_tbal_prev50": window(ct, -49, 1),
        "x_tbal_next50": window(ct, 1, 51),
        "x_tbal_prev125": window(ct, -124, 1),
        "x_tbal_next125": window(ct, 1, 126),
        "g_tsince": np.minimum(since, CAP_S),
        "g_tuntil": np.minimum(until, CAP_S),
        "g_tagree": last_team * next_team,
        "g_tprev_switch": last_team * prev_team,
        "g_tsince_switch": np.minimum(since_sw, 2 * CAP_S),
        "g_tn125": window(co, -124, 126),
        "g_tlast_dv": pick(t["dv"], li, has_last, np.nan),
        "g_tnext_dv": pick(t["dv"], ni, has_next, np.nan),
        "g_tlast_dist": pick(t["dist"], li, has_last, np.nan),
        "g_tnext_dist": pick(t["dist"], ni, has_next, np.nan),
        "g_tvis_prev50": window(cs, -49, 1) / 50.0,
        "g_tvis_next50": window(cs, 1, 51) / 50.0,
    }
    return np.stack([cols[k] for k in NAMES], axis=1).astype(np.float32), list(NAMES)
