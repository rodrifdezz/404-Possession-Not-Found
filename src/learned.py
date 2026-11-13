"""Learned evidence (v3): per-frame features + two logistic models, numpy only.

Replaces the hand-weighted rules of dead.py / team.py with two logistic regressions whose weights
are fitted ONLY on external matches (bench/train.py, IDSSE/DFL) and stored in models/:

    dead model   log-odds that the ball is dead                       (any frame)
    team model   log-odds that the home team has the ball, if alive   (antisymmetric: swapping
                 home and away flips the sign of every team feature, and the model has no
                 intercept, so neither team is favoured)

Every feature is computed from the tracking alone, with no player identities (slots swap), and
summarised over centred windows of 1, 3 and 9 s (the labels are produced offline, so the future
is fair game).
"""

import json
import os
import warnings

import numpy as np

from . import ball as ball_mod
from . import config as C

WINDOWS = (25, 75, 225)          # centred context windows, frames (1, 3, 9 s)
CONTROL_M = 1.5                  # a player this close to the ball can be touching it
TOUCH_MEMORY_S = 5.0             # decay of the "last / next team near the ball" memory
MAX_GAP_S = 10.0                 # cap for "time since / until the ball was seen"


# ------------------------------------------------------------------ helpers

def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fn(*args, **kwargs)


def centred_mean(x, w):
    """Mean of x over a centred window of w frames (shorter at the edges). x has no NaN."""
    n, k = len(x), w // 2
    c = np.concatenate(([0.0], np.cumsum(x, dtype=np.float64)))
    i = np.arange(n)
    lo, hi = np.clip(i - k, 0, n), np.clip(i + k + 1, 0, n)
    return (c[hi] - c[lo]) / (hi - lo)


def _since(mask, fps):
    """Seconds since mask was last True (capped)."""
    i = np.arange(len(mask))
    last = np.maximum.accumulate(np.where(mask, i, -10 ** 9))
    return np.clip((i - last) / fps, 0, MAX_GAP_S)


def _until(mask, fps):
    return _since(mask[::-1], fps)[::-1]


def _decayed_sign(event, sign, fps, reverse=False):
    """Sign of the most recent (or next) event, fading with the time elapsed since it."""
    if reverse:
        return _decayed_sign(event[::-1], sign[::-1], fps)[::-1]
    i = np.arange(len(event))
    last = np.maximum.accumulate(np.where(event, i, -1))
    s = np.where(last >= 0, sign[np.maximum(last, 0)], 0.0)
    return s * np.exp(-(i - last) / (TOUCH_MEMORY_S * fps))


# ------------------------------------------------------------------ features

def features(atd, pitch=None, halves=None, p=None):
    """(dead_X, team_X): float32 matrices (N, n_dead) and (N, n_team), in the order of
    DEAD_FEATURES and TEAM_FEATURES."""
    p = C.PARAMS if p is None else p
    L, W = C.PITCH if pitch is None else pitch
    halves = C.HALVES if halves is None else halves
    fps = C.FPS
    players, team = atd["players"], atd["team"]
    n = len(players)
    home, away = team > 0, team < 0

    # --- ball
    raw_seen = ~np.isnan(atd["ball"][:, 0])
    bxy, bspeed = ball_mod.clean_ball(atd["ball"], p)
    seen = ~np.isnan(bxy[:, 0])
    bx, by = bxy[:, 0], bxy[:, 1]
    with np.errstate(invalid="ignore"):
        out_dist = np.fmax(np.fmax(-bx, bx - L), np.fmax(-by, by - W))
    outside = np.clip(np.nan_to_num(out_dist, nan=0.0), 0, 5)
    inside_margin = np.clip(np.nan_to_num(-out_dist, nan=0.0), 0, 10)
    centre = np.clip(np.nan_to_num(np.hypot(bx - L / 2, by - W / 2), nan=20.0), 0, 20)
    speed = np.clip(np.nan_to_num(bspeed, nan=0.0), 0, 40)
    since, until = _since(raw_seen, fps), _until(raw_seen, fps)

    # --- players
    step = np.full(players.shape[:2], np.nan, dtype=np.float32)
    step[1:] = np.linalg.norm(np.diff(players, axis=0), axis=2) * fps
    step[step > 12] = np.nan                     # identity jumps, not running
    valid = ~np.isnan(step)
    n_valid = np.maximum(valid.sum(axis=1), 1)
    med_speed = np.nan_to_num(_quiet(np.nanmedian, step, axis=1), nan=0.0)
    frac_slow = ((step < 1.5) & valid).sum(axis=1) / n_valid
    frac_fast = ((step > 4.0) & valid).sum(axis=1) / n_valid
    visible = ~np.isnan(players[:, :, 0])
    n_vis = np.clip(visible.sum(axis=1) / 22.0, 0, 1.5)

    px = players[:, :, 0]
    hx = _quiet(np.nanmean, np.where(home, px, np.nan), axis=1)
    ax = _quiet(np.nanmean, np.where(away, px, np.nan), axis=1)
    home_left = hx < ax
    left = px < L / 2
    own_side = np.where(home[None, :], left == home_left[:, None], left != home_left[:, None])
    frac_own = lambda m: (own_side & visible & m).sum(axis=1) / np.maximum((visible & m).sum(axis=1), 1)
    formation = np.nan_to_num((frac_own(home) + frac_own(away)) / 2, nan=0.5)

    def spread(m):
        c = _quiet(np.nanmean, np.where(m[None, :, None], players, np.nan), axis=1)
        d = np.linalg.norm(players - c[:, None, :], axis=2)
        return np.nan_to_num(_quiet(np.nanmean, np.where(m, d, np.nan), axis=1), nan=0.0)

    spread_h, spread_a = spread(home), spread(away)

    # --- players around the ball
    dist = np.linalg.norm(players - bxy[:, None, :], axis=2)
    dh = _quiet(np.nanmin, np.where(home, dist, np.nan), axis=1)
    da = _quiet(np.nanmin, np.where(away, dist, np.nan), axis=1)
    dh, da = np.nan_to_num(dh, nan=30.0), np.nan_to_num(da, nan=30.0)
    closest = np.clip(np.minimum(dh, da), 0, 15)
    diff = np.clip(da - dh, -10, 10) * seen
    near = np.where(closest < 2.0, np.sign(da - dh), 0.0) * seen
    count = lambda m, r: ((dist < r) & m).sum(axis=1)
    cnt3 = (count(home, 3) - count(away, 3)) * seen
    cnt6 = (count(home, 6) - count(away, 6)) * seen
    touch = seen & (closest < CONTROL_M)
    side = np.sign(da - dh)
    last_touch = _decayed_sign(touch, side, fps)
    next_touch = _decayed_sign(touch, side, fps, reverse=True)

    # ball progress towards the away goal (+) or the home goal (-); sides inferred per half
    progress = np.zeros(n)
    for a, b in halves:
        hl = _quiet(np.nanmean, hx[a:b + 1]) < _quiet(np.nanmean, ax[a:b + 1])
        progress[a:b + 1] = (bx[a:b + 1] - L / 2) / (L / 2) * (1.0 if hl else -1.0)
    progress = np.clip(np.nan_to_num(progress, nan=0.0), -1.2, 1.2)

    dead_cols = {
        "ball_seen": seen.astype(float), "ball_raw_seen": raw_seen.astype(float),
        "ball_speed": speed, "ball_outside_m": outside, "ball_inside_margin_m": inside_margin,
        "ball_centre_dist_m": centre, "s_since_ball": since, "s_until_ball": until,
        "players_median_speed": med_speed, "players_frac_slow": frac_slow,
        "players_frac_fast": frac_fast, "players_visible": n_vis, "formation_own_half": formation,
        "spread_mean_m": (spread_h + spread_a) / 2, "closest_player_m": closest,
        "abs_team_gap_m": np.abs(diff),
    }
    for name in ("ball_seen", "ball_speed", "players_median_speed", "players_frac_slow",
                 "ball_outside_m", "formation_own_half", "closest_player_m"):
        for w in WINDOWS:
            dead_cols["{}_w{}".format(name, w)] = centred_mean(dead_cols[name], w)

    team_cols = {
        "team_gap_m": diff, "nearest_team": near, "count3_diff": cnt3, "count6_diff": cnt6,
        "last_touch_team": last_touch, "next_touch_team": next_touch,
        "ball_progress": progress, "spread_diff_m": spread_h - spread_a,
    }
    for name in ("team_gap_m", "nearest_team", "count3_diff", "ball_progress"):
        for w in WINDOWS:
            team_cols["{}_w{}".format(name, w)] = centred_mean(team_cols[name], w)

    assert list(dead_cols) == DEAD_FEATURES and list(team_cols) == TEAM_FEATURES
    to_matrix = lambda cols: np.column_stack(list(cols.values())).astype(np.float32)
    return to_matrix(dead_cols), to_matrix(team_cols)


def _names(base, windowed):
    return base + ["{}_w{}".format(n, w) for n in windowed for w in WINDOWS]


DEAD_FEATURES = _names(
    ["ball_seen", "ball_raw_seen", "ball_speed", "ball_outside_m", "ball_inside_margin_m",
     "ball_centre_dist_m", "s_since_ball", "s_until_ball", "players_median_speed",
     "players_frac_slow", "players_frac_fast", "players_visible", "formation_own_half",
     "spread_mean_m", "closest_player_m", "abs_team_gap_m"],
    ["ball_seen", "ball_speed", "players_median_speed", "players_frac_slow", "ball_outside_m",
     "formation_own_half", "closest_player_m"])
TEAM_FEATURES = _names(
    ["team_gap_m", "nearest_team", "count3_diff", "count6_diff", "last_touch_team",
     "next_touch_team", "ball_progress", "spread_diff_m"],
    ["team_gap_m", "nearest_team", "count3_diff", "ball_progress"])


# ------------------------------------------------------------------ logistic models

def fit_logistic(X, y, intercept=True, ridge=1.0, iters=30):
    """L2-regularised logistic regression by Newton's method. Deterministic.

    Features are standardised first (centred only when there is an intercept, so that a model
    without intercept stays antisymmetric). Returns a dict that predict() understands.
    """
    X = X.astype(np.float64)
    mean = X.mean(axis=0) if intercept else np.zeros(X.shape[1])
    scale = X.std(axis=0) if intercept else np.sqrt((X ** 2).mean(axis=0))
    scale[scale < 1e-9] = 1.0
    Z = (X - mean) / scale
    if intercept:
        Z = np.column_stack([np.ones(len(Z)), Z])
    w = np.zeros(Z.shape[1])
    reg = np.full(Z.shape[1], ridge)
    if intercept:
        reg[0] = 0.0
    for _ in range(iters):
        q = 1.0 / (1.0 + np.exp(-(Z @ w)))
        grad = Z.T @ (q - y) + reg * w
        hess = (Z * (q * (1 - q))[:, None]).T @ Z + np.diag(reg)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-7:
            break
    b = float(w[0]) if intercept else 0.0
    coef = w[1:] if intercept else w
    return {"mean": mean.tolist(), "scale": scale.tolist(), "coef": coef.tolist(), "bias": b}


def predict(model, X):
    """Log-odds for every row of X."""
    mean, scale = np.asarray(model["mean"]), np.asarray(model["scale"])
    return ((X - mean) / scale) @ np.asarray(model["coef"]) + model["bias"]


# ------------------------------------------------------------------ use in the pipeline

def load_model(path=None):
    path = path or os.path.join(C.ROOT, C.EVIDENCE_MODEL)
    with open(path, encoding="utf-8") as fh:
        model = json.load(fh)
    if model["dead_features"] != DEAD_FEATURES or model["team_features"] != TEAM_FEATURES:
        raise ValueError("{} was trained with other features; retrain it "
                         "(python -m bench.train)".format(path))
    return model


def evidence(atd, model, pitch=None, halves=None, p=None):
    """(team_llr, dead_logit) per frame from the learned models (no Smart Tagging)."""
    dead_X, team_X = features(atd, pitch, halves, p)
    return predict(model["team"], team_X), predict(model["dead"], dead_X)
