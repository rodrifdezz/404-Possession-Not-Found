"""EXPERIMENTAL (v4 ablation, bench/v4.py): possession-change features for layer 2.

Not used by the submission. src/stack.py already gives layer 2 the H/A margin (log P(H) -
log P(A)) of layer 1 with centred means / minima / maxima, shifted values and separate past and
future means. What it does not give is any explicit COMPARISON between the past and the future,
which a tree can only approximate with many splits. Only those comparisons are added:

  d12, d50      future mean - past mean of the margin over 12 frames (~0.5 s) and 50 (2 s):
                the size and direction of a change of possession around this frame;
  rev25, rev75  persistent reversal: when the past and future means (25 / 75 frames) favour
                different teams, the weaker of the two, signed by the future team; else 0;
  dip           centred 15-frame mean minus the average of the past and future 50-frame means:
                a short excursion towards the other team that is then undone (quick recovery,
                or a touch without control);
  flips50       number of sign changes of the (5-frame smoothed) margin within +-2 s.

All but flips50 change sign when the teams are swapped (swap_teams()).
"""

import numpy as np
import pandas as pd

NAMES = ["s2_d12", "s2_d50", "s2_rev25", "s2_rev75", "s2_dip", "s2_flips50"]
ANTISYMMETRIC = [True, True, True, True, True, False]


def _past(s, w):
    return s.rolling(w, min_periods=1).mean().shift(1).to_numpy()


def _future(s, w):
    return s[::-1].rolling(w, min_periods=1).mean().shift(1).to_numpy()[::-1]


def _reversal(p, f):
    opposite = np.sign(p) * np.sign(f) < 0
    return np.where(opposite, np.sign(f) * np.minimum(np.abs(p), np.abs(f)), 0.0)


def features(proba):
    """Layer-1 proba (N,3) -> (N, len(NAMES))."""
    lp = np.log(np.clip(proba, 1e-4, 1.0))
    margin = lp[:, 0] - lp[:, 1]
    s = pd.Series(margin)
    p12, f12 = _past(s, 12), _future(s, 12)
    p25, f25 = _past(s, 25), _future(s, 25)
    p50, f50 = _past(s, 50), _future(s, 50)
    p75, f75 = _past(s, 75), _future(s, 75)
    c15 = s.rolling(15, center=True, min_periods=1).mean().to_numpy()
    sm = np.sign(s.rolling(5, center=True, min_periods=1).mean().to_numpy())
    change = np.zeros(len(sm))
    change[1:] = (sm[1:] * sm[:-1] < 0).astype(float)
    flips = pd.Series(change).rolling(101, center=True, min_periods=1).sum().to_numpy()
    cols = [f12 - p12, f50 - p50, _reversal(p25, f25), _reversal(p75, f75),
            c15 - (p50 + f50) / 2.0, flips]
    return np.stack(cols, axis=1).astype(np.float64)


def swap_teams(E):
    G = E.copy()
    for i, anti in enumerate(ANTISYMMETRIC):
        if anti:
            G[:, i] = -E[:, i]
    return G
