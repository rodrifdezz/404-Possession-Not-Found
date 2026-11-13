"""Step 1 - clean the ball: fill short gaps and compute its speed."""

import warnings

import numpy as np

from . import config as C


def nan_runs(mask):
    """(start, exclusive_end) of every True run in a boolean array."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def interpolate_short_gaps(ball, max_gap):
    """Linear interpolation of gaps of <= max_gap frames with the ball seen on both sides."""
    out = ball.copy()
    missing = np.isnan(ball[:, 0])
    starts, ends = nan_runs(missing)
    for s, e in zip(starts, ends):
        if e - s <= max_gap and s > 0 and e < len(ball):
            t = (np.arange(s, e) - (s - 1)) / float(e - (s - 1))
            out[s:e] = ball[s - 1] + t[:, None] * (ball[e] - ball[s - 1])
    return out


def rolling_median(x, w):
    if w <= 1:
        return x
    half = w // 2
    padded = np.concatenate([np.full(half, np.nan), x, np.full(half, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(padded, w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)     # all-NaN windows -> NaN, silently
        return np.nanmedian(windows, axis=1)


def speed(xy, smooth=1):
    """Speed in m/s (NaN where there is no position)."""
    v = np.full(len(xy), np.nan, dtype=np.float32)
    v[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) * C.FPS
    return rolling_median(v, smooth).astype(np.float32)


def clean_ball(ball, p=C.PARAMS):
    filled = interpolate_short_gaps(ball, int(round(p["ball_max_gap_s"] * C.FPS)))
    return filled, speed(filled, p["ball_speed_smooth"])
