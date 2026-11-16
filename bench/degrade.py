"""Damage a clean external match so that it resembles the challenge's automatic tracking.

What we know about the challenge tracking comes from the official documentation (DATA.md), not
from the ground truth or the video:
  - the ball is missing on 41 % of the graded frames;
  - it goes missing "disproportionately" on the frames that matter: occlusions, crowded boxes and
    balls in the air;
  - positions are noisy and players disappear for a while (46 slots for 22 players).

We do not know how long the ball gaps last, so the bench uses two scenarios (short and long gaps)
and picks parameters that work in both, instead of betting on one.
"""

import numpy as np

BALL_MISSING = 0.41          # DATA.md: 1 - 94,214 / 158,857 graded frames
SCENARIOS = {"short_gaps": 0.6, "long_gaps": 3.0}   # mean duration of a ball gap (s)
FPS = 25


def risk(m):
    """Relative weight of losing the ball on each frame (1 = normal)."""
    L, W = m["pitch"]
    x, y = m["ball"][:, 0], m["ball"][:, 1]
    d = np.linalg.norm(m["players"] - m["ball"][:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        crowded = np.sum(d < 3.0, axis=1) >= 3
        in_air = m["ball_z"] > 1.5
        outside = (x < 0) | (x > L) | (y < 0) | (y > W)
    return 1.0 + 2.0 * crowded + 2.0 * in_air + 2.0 * outside


def _ball_gaps(m, rng, mean_gap_s):
    """Mask of frames without ball: exponentially long gaps until 41 % is reached."""
    n = len(m["ball"])
    graded = m["labels"] != "X"
    w = risk(m) * graded
    p = w / w.sum()
    missing = np.zeros(n, dtype=bool)
    target = BALL_MISSING * graded.sum()
    while (missing & graded).sum() < target:
        starts = rng.choice(n, size=200, p=p)
        lengths = np.maximum(1, rng.exponential(mean_gap_s * FPS, size=200).astype(int))
        for s, k in zip(starts, lengths):
            missing[s:s + k] = True
            if (missing & graded).sum() >= target:
                break
    return missing


def _smooth_noise(rng, shape, sigma_m, corr_frames):
    """Slow position noise (like a tracker's error), not white noise per frame: white noise
    would invent speeds of several m/s that no real tracker produces."""
    raw = rng.normal(size=shape)
    k = np.ones(corr_frames) / corr_frames
    smooth = np.apply_along_axis(lambda v: np.convolve(v, k, mode="same"), 0, raw)
    return (smooth * sigma_m / max(smooth.std(), 1e-9)).astype(np.float32)


def _player_dropouts(m, rng, rate=0.07, mean_s=2.0):
    """Each player disappears now and then (~7 % of the time, ~2 s each time)."""
    n, slots = m["players"].shape[:2]
    gone = np.zeros((n, slots), dtype=bool)
    n_runs = int(rate * n / (mean_s * FPS))
    for j in range(slots):
        starts = rng.integers(0, n, size=n_runs)
        lengths = rng.exponential(mean_s * FPS, size=n_runs).astype(int) + 1
        for s, k in zip(starts, lengths):
            gone[s:s + k, j] = True
    return gone


def degrade(m, scenario, seed):
    """Degraded copy of match m. The ground truth (m["labels"]) is not touched."""
    rng = np.random.default_rng(seed)
    players = m["players"] + _smooth_noise(rng, m["players"].shape, 0.2, 25)
    players[_player_dropouts(m, rng)] = np.nan
    ball = m["ball"] + _smooth_noise(rng, m["ball"].shape, 0.2, 5)
    ball[_ball_gaps(m, rng, SCENARIOS[scenario])] = np.nan
    return {"players": players, "team": m["team"], "ball": ball}


# ------------------------------------------------------------------ training degradations
# A second, independent way of damaging a clean match, used ONLY to build the training mixture of
# the learned evidence (bench/train_model.py): Markov-chain ball visibility, white position noise,
# player dropouts in bursts and tracks fragmented over 46 slots. Training on several different
# degradations (these presets plus the two scenarios above) keeps the model from relying on the
# quirks of any single one; the real tracking is none of them.

NORMAL = {"ball_visible": 0.59, "ball_gap_mean_s": 1.2, "crowd_boost": 2.5, "ball_noise_m": 0.25,
          "player_noise_m": 0.15, "player_missing": 0.07, "player_gap_mean_s": 2.0,
          "n_slots": 46, "fragment_every_s": 240.0}
EASY = dict(NORMAL, ball_visible=0.68, ball_gap_mean_s=0.8, ball_noise_m=0.15,
            player_noise_m=0.1, player_missing=0.04)
HARD = dict(NORMAL, ball_visible=0.5, ball_gap_mean_s=2.0, ball_noise_m=0.5,
            player_noise_m=0.3, player_missing=0.12, fragment_every_s=60.0)
PRESETS = {"normal": NORMAL, "easy": EASY, "hard": HARD}


def degrade_markov(m, preset, seed):
    """Degraded copy of match m with one of PRESETS. The ground truth is not touched."""
    p = PRESETS[preset]
    rng = np.random.default_rng(seed)
    pl = m["players"].astype(np.float64)
    ball = m["ball"].astype(np.float64)
    n, s, _ = pl.shape
    pl = pl + rng.normal(0, p["player_noise_m"], pl.shape)
    ball = ball + rng.normal(0, p["ball_noise_m"], ball.shape)

    # ball: visible/hidden Markov chain, lost more often with players around it
    d = np.linalg.norm(pl - ball[:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        crowd = np.nansum(d < 3.0, axis=1)
    q_end = 1.0 / (p["ball_gap_mean_s"] * FPS)                      # leaving a gap
    q_start = q_end * (1 - p["ball_visible"]) / p["ball_visible"]   # equilibrium -> visible share
    hazard = q_start * (1 + p["crowd_boost"] * np.clip(crowd - 1, 0, 4) / 4.0)
    hazard /= hazard.mean() / q_start
    u = rng.random(n)
    seen = np.empty(n, dtype=bool)
    state = True
    for t in range(n):
        state = (u[t] >= hazard[t]) if state else (u[t] < q_end)
        seen[t] = state
    ball[~seen] = np.nan

    # players lost in bursts
    pe = 1.0 / (p["player_gap_mean_s"] * FPS)
    ps = pe * p["player_missing"] / (1 - p["player_missing"])
    r = rng.random((n, s))
    st = np.zeros(s, dtype=bool)
    for t in range(n):
        st = np.where(st, r[t] >= pe, r[t] < ps)
        pl[t, st] = np.nan

    # fragmented tracks: a player jumps to another free slot of his team
    team = m["team"]
    per_team = p["n_slots"] // 2
    out = np.full((n, p["n_slots"], 2), np.nan)
    out_team = np.array([1] * per_team + [-1] * per_team, dtype=np.int8)
    assign = {}
    for group, offset in (([i for i in range(s) if team[i] > 0], 0),
                          ([i for i in range(s) if team[i] < 0], per_team)):
        for j, i in enumerate(group):
            assign[i] = offset + (j % per_team)
    change = rng.random((n, s)) < 1.0 / (p["fragment_every_s"] * FPS)
    src = np.arange(s)
    dst = np.array([assign[i] for i in range(s)])
    for t in range(n):
        for i in np.flatnonzero(change[t]):
            offset = 0 if team[i] > 0 else per_team
            used = set(dst.tolist())
            free = [k for k in range(offset, offset + per_team) if k not in used]
            if free:
                dst[i] = free[int(rng.integers(len(free)))]
        # several bench players can share a slot: the one that is on the pitch keeps it
        row = np.full((p["n_slots"], 2), np.nan)
        ok = ~np.isnan(pl[t, :, 0])
        row[dst[ok]] = pl[t, src[ok]]
        out[t] = row
    return {"players": out.astype(np.float32), "team": out_team, "ball": ball.astype(np.float32)}
