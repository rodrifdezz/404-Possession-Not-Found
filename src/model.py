"""Learned evidence: per-frame features -> probability of H / A / D, in two layers.

  layer 1  HistGradientBoostingClassifier on ~90 features of the frame and its surroundings
           (src/frame_features.py)
  layer 2  HistGradientBoostingClassifier on the sequence of layer-1 probabilities (src/stack.py)

Both are fitted by bench/train_model.py ONLY on public matches of other teams (the Metrica Sports
open sample data), with their tracking degraded to resemble the automatic tracking of the
challenge. No labelled data of the challenge match is used. The fitted model is
models/possession_hgb.pkl (it holds no match data) and models/possession_hgb.json describes it.

Both layers are symmetric: the prediction is averaged with the one obtained after swapping the
teams, so the model cannot favour home or away.

The probabilities are handed to the decoder as the same two quantities the rule-based evidence
produces (team log-ratio and dead-ball log-odds), so src/hmm.py is shared by both.
"""

import os
import pickle

import numpy as np

from . import ball as ball_mod
from . import config as C
from . import frame_features, stack

MODEL_PATH = os.path.join(C.ROOT, "models", "possession_hgb.pkl")
DECIMALS = 3      # inputs are rounded: same decisions on macOS, Linux and Windows
P_FLOOR = 1e-4    # no state is ever ruled out by a single frame

_CACHE = {}


def load(path=None):
    path = MODEL_PATH if path is None else path
    if path not in _CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError("Model {} is missing. Fit it with: python -m bench.train_model"
                                    .format(path))
        with open(path, "rb") as fh:
            _CACHE[path] = pickle.load(fh)
    return _CACHE[path]


def prepare(X):
    X = np.round(np.asarray(X, dtype=np.float64), DECIMALS)
    X[~np.isfinite(X)] = np.nan
    return X


def sym_proba(clf, X, X_swapped):
    p1 = clf.predict_proba(prepare(X))
    p2 = clf.predict_proba(prepare(X_swapped))[:, [1, 0, 2]]
    return np.round((p1 + p2) / 2.0, 4)


def stage1(clf, X, names):
    return sym_proba(clf, X, frame_features.swap_teams(X, names))


def stage2(clf, P1, X, names):
    F = stack.features(P1, X, names)
    return sym_proba(clf, F, stack.swap_teams(F))


def predict_proba(X, names, bundle=None):
    """(N,3) probabilities [H, A, D]."""
    bundle = load() if bundle is None else bundle
    if list(bundle["names"]) != list(names):
        raise ValueError("The features do not match the fitted model: refit it "
                         "(python -m bench.train_model).")
    P1 = stage1(bundle["stage1"], X, names)
    if bundle.get("stage2") is None:
        return P1
    return stage2(bundle["stage2"], P1, X, names)


def to_evidence(proba):
    """(team_llr, dead_logit) such that hmm.emissions() gives back exactly `proba`."""
    P = np.clip(np.asarray(proba, dtype=np.float64), P_FLOOR, 1.0)
    P = P / P.sum(axis=1, keepdims=True)
    team_llr = np.log(P[:, 0]) - np.log(P[:, 1])
    dead_logit = np.log(P[:, 2]) - np.log(P[:, 0] + P[:, 1])
    return team_llr, dead_logit


def matrix(atd, p, pitch, halves):
    ball_xy, _ = ball_mod.clean_ball(atd["ball"], p)
    return frame_features.compute(atd, ball_xy, halves, pitch=pitch)


def evidence(atd, p=None, pitch=None, halves=None, bundle=None):
    """Tracking -> (team_llr, dead_logit) per frame. None = challenge match values (config)."""
    p = C.PARAMS if p is None else p
    pitch = C.PITCH if pitch is None else pitch
    halves = C.HALVES if halves is None else halves
    X, names = matrix(atd, p, pitch, halves)
    return to_evidence(predict_proba(X, names, bundle))
