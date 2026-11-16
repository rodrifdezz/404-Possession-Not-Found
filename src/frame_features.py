"""Per-frame features for the learned evidence (src/model.py). They decide nothing: they
describe the play.

They come in one block PER TEAM (home and away, each in its own attacking direction) plus a
global block. That makes the model symmetric: swapping the two blocks is swapping the teams,
which is how the training set is augmented (bench/train_model.py) and how predictions are
averaged (src/model.py).

Many features do NOT need the ball (shape of the teams, speeds, pressure...), which is what
lets the model say something on the 41 % of frames where the tracker has no ball.

No player identities are used (slots swap): only positions and the team of each slot.
"""

import warnings

import numpy as np
import pandas as pd

FPS = 25


def _nanfun(fun, a, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return fun(a, axis=axis)


def central_velocity(xy, k):
    """Velocity (vector, m/s) by central difference over +-k frames. xy (N,...,2)."""
    v = np.full(xy.shape, np.nan, dtype=np.float32)
    v[k:-k] = (xy[2 * k:] - xy[:-2 * k]) * (FPS / (2.0 * k))
    return v


def orientation(players, team, halves, n):
    """+1 where HOME attacks towards increasing x, -1 otherwise (from where each team stands).

    Frames outside the halves take the direction of the nearest half, so that swapping the teams
    flips the sign everywhere."""
    sign = np.ones(n, dtype=np.float32)
    centre = np.empty(len(halves))
    value = np.empty(len(halves), dtype=np.float32)
    for i, (a, b) in enumerate(halves):
        xs = players[a:b + 1, :, 0]
        home_x = _nanfun(np.nanmean, xs[:, team > 0], None)
        away_x = _nanfun(np.nanmean, xs[:, team < 0], None)
        value[i] = 1.0 if home_x < away_x else -1.0
        centre[i] = (a + b) / 2.0
    if len(halves):
        nearest = np.abs(np.arange(n)[:, None] - centre[None, :]).argmin(axis=1)
        sign = value[nearest]
        for i, (a, b) in enumerate(halves):
            sign[a:b + 1] = value[i]
    return sign


def _team_block(players, pv, mask, ball, bv, attack_sign, pitch_len):
    """Features of one team. attack_sign: +1 if it attacks towards increasing x."""
    n = len(ball)
    pm = np.where(mask[None, :, None], players, np.nan)
    d = np.linalg.norm(pm - ball[:, None, :], axis=2)                 # (N,S)
    dd = np.where(np.isnan(d), np.inf, d)
    order = np.argsort(dd, axis=1)
    i1 = order[:, 0]
    rows = np.arange(n)
    d1 = dd[rows, i1]
    d2 = dd[rows, order[:, 1]]
    d1 = np.where(np.isfinite(d1), d1, np.nan)
    d2 = np.where(np.isfinite(d2), d2, np.nan)
    v1 = pv[rows, i1]
    relv = np.linalg.norm(bv - v1, axis=1)
    sp1 = np.linalg.norm(v1, axis=1)
    n3 = np.sum(d < 3.0, axis=1).astype(np.float32)
    n8 = np.sum(d < 8.0, axis=1).astype(np.float32)

    # position in its own attacking direction: 0 = own goal line, pitch_len = the opponent's
    x_att = np.where(attack_sign[:, None] > 0, pm[:, :, 0], pitch_len - pm[:, :, 0])
    ball_att = np.where(attack_sign > 0, ball[:, 0], pitch_len - ball[:, 0])
    cx = _nanfun(np.nanmean, x_att, 1)
    sx = _nanfun(np.nanstd, x_att, 1)
    sy = _nanfun(np.nanstd, pm[:, :, 1], 1)
    behind = _nanfun(np.nansum, (x_att < ball_att[:, None]).astype(np.float32), 1)
    behind = np.where(np.isnan(ball_att), np.nan, behind)
    spd = np.linalg.norm(np.where(mask[None, :, None], pv, np.nan), axis=2)
    mspd = _nanfun(np.nanmean, spd, 1)
    # mean team velocity along its attacking direction (pushing up / dropping back)
    vx = _nanfun(np.nanmean, np.where(mask[None, :], pv[:, :, 0], np.nan), 1) * attack_sign
    hi = _nanfun(np.nanmax, x_att, 1)          # most advanced player
    return {
        "d1": d1, "d2": d2, "relv": relv, "sp1": sp1, "n3": n3, "n8": n8,
        "cx": cx, "cx_rel": cx - ball_att, "sx": sx, "sy": sy, "behind": behind,
        "mspd": mspd, "vx": vx, "hi": hi, "ball_att": ball_att,
    }


def _roll(x, w, how, center=True):
    s = pd.Series(x)
    r = s.rolling(w, center=center, min_periods=1)
    return getattr(r, how)().to_numpy(dtype=np.float32)


def _shift_roll(x, w, how, future):
    """Statistic over the w previous frames (future=False) or the w next ones (True)."""
    s = pd.Series(x)
    if future:
        s = s[::-1]
    r = getattr(s.rolling(w, min_periods=1), how)().shift(1)
    out = r.to_numpy(dtype=np.float32)
    return out[::-1] if future else out


def _run_since(mask):
    """Frames since the last True (0 if True now)."""
    idx = np.where(mask, np.arange(len(mask)), -1)
    last = np.maximum.accumulate(idx)
    out = np.arange(len(mask)) - last
    out[last < 0] = 10 ** 6
    return out.astype(np.float32)


TEAM_TEMPORAL = ["d1", "relv", "n3"]
GLOBAL_TEMPORAL = ["ball_speed", "mspd_all", "ball_seen"]


def compute(atd, ball_xy, halves, pitch=(105.3, 68.0)):
    """Feature matrix (N, F) float32 and the list of names. ball_xy: the cleaned ball."""
    players = atd["players"].astype(np.float32)
    team = atd["team"]
    n = len(ball_xy)
    ball = ball_xy.astype(np.float32)
    seen = ~np.isnan(ball[:, 0])
    raw_seen = ~np.isnan(atd["ball"][:, 0])

    pv = central_velocity(players, 3)
    pv[np.linalg.norm(pv, axis=-1) > 12] = np.nan          # track jumps, not sprints
    bv = central_velocity(ball, 2)
    ball_speed = np.linalg.norm(bv, axis=1)
    ba = np.full(n, np.nan, dtype=np.float32)
    ba[3:-3] = np.linalg.norm(bv[6:] - bv[:-6], axis=1) * (FPS / 6.0)

    sign = orientation(players, team, halves, n)
    H = _team_block(players, pv, team > 0, ball, bv, sign, pitch[0])
    A = _team_block(players, pv, team < 0, ball, bv, -sign, pitch[0])

    spd_all = np.linalg.norm(pv, axis=2)
    mspd_all = _nanfun(np.nanmean, spd_all, 1)
    L, W = pitch
    bx, by = ball[:, 0], ball[:, 1]
    edge = np.minimum.reduce([bx, L - bx, by, W - by])      # < 0 off the pitch
    G = {
        "ball_seen": seen.astype(np.float32),
        "raw_seen": raw_seen.astype(np.float32),
        "since_seen": np.minimum(_run_since(raw_seen), 500),
        "until_seen": np.minimum(_run_since(raw_seen[::-1])[::-1], 500),
        "ball_speed": ball_speed,
        "ball_acc": ba,
        "edge": edge,
        "edge_x": np.minimum(bx, L - bx),
        "edge_y": np.minimum(by, W - by),
        "mspd_all": mspd_all,
        "n_players": np.sum(~np.isnan(players[:, :, 0]), axis=1).astype(np.float32),
    }
    # how long the ball has been almost still
    still = seen & (np.nan_to_num(ball_speed, nan=99) < 0.7)
    G["still_run"] = np.minimum(_run_since(~still), 500)

    feats, names = [], []

    def add(name, x):
        feats.append(np.asarray(x, dtype=np.float32))
        names.append(name)

    for prefix, blk in (("h_", H), ("a_", A)):
        for k, v in blk.items():
            add(prefix + k, v)
        for k in TEAM_TEMPORAL:
            for w in (10, 25, 75):
                add("{}{}_min{}".format(prefix, k, w) if k == "d1" else
                    "{}{}_mean{}".format(prefix, k, w),
                    _roll(blk[k], w, "min" if k == "d1" else "mean"))
            for fut in (False, True):
                add("{}{}_{}25".format(prefix, k, "next" if fut else "prev"),
                    _shift_roll(blk[k], 25, "min" if k == "d1" else "mean", fut))
    for k, v in G.items():
        add("g_" + k, v)
    for k in GLOBAL_TEMPORAL:
        for w in (25, 75, 250):
            add("g_{}_mean{}".format(k, w), _roll(G[k], w, "mean"))
    add("g_ball_speed_max25", _roll(G["ball_speed"], 25, "max"))
    add("g_ball_acc_max10", _roll(G["ball_acc"], 10, "max"))
    add("g_edge_min50", _roll(G["edge"], 50, "min"))
    # home - away differences (they help the trees)
    add("x_d1_diff", H["d1"] - A["d1"])
    add("x_n3_diff", H["n3"] - A["n3"])
    add("x_cx_diff", H["cx"] - A["cx"])
    add("x_vx_diff", H["vx"] - A["vx"])
    add("x_d1min25_diff", _roll(H["d1"], 25, "min") - _roll(A["d1"], 25, "min"))
    X = np.stack(feats, axis=1)
    return X, names


def swap_teams(X, names):
    """The same matrix with home and away swapped."""
    idx = {nme: i for i, nme in enumerate(names)}
    Y = X.copy()
    for nme, i in idx.items():
        if nme.startswith("h_"):
            Y[:, i] = X[:, idx["a_" + nme[2:]]]
        elif nme.startswith("a_"):
            Y[:, i] = X[:, idx["h_" + nme[2:]]]
        elif nme.startswith("x_"):
            Y[:, i] = -X[:, i]
    return Y
