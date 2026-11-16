"""Second layer of the learned evidence: features built from the SEQUENCE of layer-1
probabilities.

Layer 1 decides frame by frame; layer 2 looks at what layer 1 said in the seconds around
(means, minima, maxima, past and future values). It helps most when the tracking is poor.
It is trained on out-of-sample layer-1 probabilities (bench/train_model.py).
"""

import numpy as np
import pandas as pd

RAW = ["g_ball_seen", "g_since_seen", "g_until_seen", "x_d1_diff", "g_ball_speed", "g_mspd_all"]
WINDOWS = (5, 15, 51, 151)
OFFSETS = (-50, -25, -10, -4, 4, 10, 25, 50)
SIDES = (10, 25, 75)
BLOCK = 1 + 3 * len(WINDOWS) + len(OFFSETS) + 2 * len(SIDES)


def _block(sig):
    s = pd.Series(sig)
    cols = [sig]
    for w in WINDOWS:
        r = s.rolling(w, center=True, min_periods=1)
        cols += [r.mean().to_numpy(), r.min().to_numpy(), r.max().to_numpy()]
    for o in OFFSETS:
        cols.append(s.shift(o).to_numpy())
    for w in SIDES:
        cols.append(s.rolling(w, min_periods=1).mean().shift(1).to_numpy())
        cols.append(s[::-1].rolling(w, min_periods=1).mean().shift(1).to_numpy()[::-1])
    return cols


def features(proba, X, names):
    """Layer-1 proba (N,3) + a few of the original features -> (N, F2)."""
    lp = np.log(np.clip(proba, 1e-4, 1.0))
    margin = lp[:, 0] - lp[:, 1]                               # home vs away
    dead = lp[:, 2] - np.logaddexp(lp[:, 0], lp[:, 1])         # dead vs alive
    ix = {n: i for i, n in enumerate(names)}
    cols = _block(margin) + _block(dead) + [X[:, ix[r]] for r in RAW]
    return np.stack(cols, axis=1).astype(np.float64)


def swap_teams(F):
    """The same matrix with home and away swapped."""
    G = F.copy()
    G[:, :BLOCK] *= -1
    for k in range(len(WINDOWS)):
        i = 1 + 3 * k
        G[:, i + 1], G[:, i + 2] = -F[:, i + 2], -F[:, i + 1]     # min <-> max
    G[:, 2 * BLOCK + RAW.index("x_d1_diff")] *= -1
    return G
