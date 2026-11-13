"""Per-frame variables computed from the tracking (nothing is decided yet)."""

import warnings

import numpy as np

from . import config as C


def _nanmin(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmin(a, axis=axis)


def _nanmean(a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmean(a, axis=axis)


def compute(atd, ball_xy, ball_speed):
    players, team = atd["players"], atd["team"]
    dist = np.linalg.norm(players - ball_xy[:, None, :], axis=2)        # (N, slots)
    d_home = _nanmin(np.where(team[None, :] > 0, dist, np.nan), axis=1)
    d_away = _nanmin(np.where(team[None, :] < 0, dist, np.nan), axis=1)

    step = np.full(players.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * C.FPS
    step[step > 12] = np.nan              # tracker identity jumps, not real sprints
    player_speed = _nanmean(step, axis=1)

    return {
        "ball_xy": ball_xy,
        "ball_seen": ~np.isnan(ball_xy[:, 0]),
        "ball_raw_seen": ~np.isnan(atd["ball"][:, 0]),
        "ball_speed": ball_speed,
        "d_home": d_home,
        "d_away": d_away,
        "player_speed": player_speed,
    }
