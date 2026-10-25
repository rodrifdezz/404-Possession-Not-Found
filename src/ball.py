"""Pieza 1 - limpiar el balón: rellenar huecos cortos y calcular su velocidad."""

import numpy as np

from . import config as C


def nan_runs(mask):
    """(inicio, fin_exclusivo) de cada racha True en un array booleano."""
    padded = np.concatenate(([False], mask, [False])).astype(np.int8)
    d = np.diff(padded)
    return np.flatnonzero(d == 1), np.flatnonzero(d == -1)


def interpolate_short_gaps(ball, max_gap):
    """Interpolación lineal de huecos de <= max_gap frames con balón visto a ambos lados."""
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
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            return np.nanmedian(windows, axis=1)


def speed(xy, smooth=1):
    """Velocidad en m/s (NaN donde no hay posición)."""
    v = np.full(len(xy), np.nan, dtype=np.float32)
    v[1:] = np.linalg.norm(np.diff(xy, axis=0), axis=1) * C.FPS
    return rolling_median(v, smooth).astype(np.float32)


def clean_ball(ball, p=C.PARAMS):
    filled = interpolate_short_gaps(ball, int(round(p["ball_max_gap_s"] * C.FPS)))
    return filled, speed(filled, p["ball_speed_smooth"])
