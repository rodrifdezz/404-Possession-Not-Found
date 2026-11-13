"""Damage a clean external match so that it resembles the challenge's automatic tracking.

What we know about the challenge tracking comes from the official documentation (DATA.md), not
from the ground truth or the video:
  - the ball is missing on 41 % of the graded frames;
  - it goes missing "disproportionately" on the frames that matter: occlusions, crowded boxes and
    balls in the air;
  - positions are noisy and players disappear for a while (46 slots for 22 players).

We do not know how long the ball gaps last, nor whether the ball is lost more during play (the
occlusions DATA.md mentions) or during stoppages (ball off camera, players crowding a set piece).
So there are three scenarios, and a choice is only kept if it works in all of them:

  short_gaps     gaps of ~0.6 s, more likely in crowds, in the air and off the pitch
  long_gaps      the same with gaps of ~3 s
  stoppage_gaps  gaps of ~3 s, much more likely while the ball is dead
"""

import numpy as np

BALL_MISSING = 0.41          # DATA.md: 1 - 94,214 / 158,857 graded frames
SCENARIOS = {                # mean gap duration (s), extra weight of losing the ball when dead
    "short_gaps": (0.6, 0.0),
    "long_gaps": (3.0, 0.0),
    "stoppage_gaps": (3.0, 6.0),
}
FPS = 25


def risk(m, dead_weight=0.0):
    """Relative weight of losing the ball on each frame (1 = normal)."""
    L, W = m["pitch"]
    x, y = m["ball"][:, 0], m["ball"][:, 1]
    d = np.linalg.norm(m["players"] - m["ball"][:, None, :], axis=2)
    with np.errstate(invalid="ignore"):
        crowded = np.sum(d < 3.0, axis=1) >= 3
        in_air = m["ball_z"] > 1.5
        outside = (x < 0) | (x > L) | (y < 0) | (y > W)
    w = 1.0 + 2.0 * crowded + 2.0 * in_air + 2.0 * outside
    if dead_weight:
        w = w + dead_weight * (m["labels"] == "D")
    return w


def _ball_gaps(m, rng, mean_gap_s, dead_weight=0.0):
    """Mask of frames without ball: exponentially long gaps until 41 % is reached."""
    n = len(m["ball"])
    graded = m["labels"] != "X"
    w = risk(m, dead_weight) * graded
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
    ball[_ball_gaps(m, rng, *SCENARIOS[scenario])] = np.nan
    return {"players": players, "team": m["team"], "ball": ball}
