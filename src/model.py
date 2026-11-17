"""Learned evidence: per-frame features -> probability of H / A / D, in two layers.

  layer 1  HistGradientBoostingClassifier on ~90 features of the frame and its surroundings
           (src/frame_features.py)
  layer 2  HistGradientBoostingClassifier on the sequence of layer-1 probabilities (src/stack.py)

Both are fitted ONLY on public matches of other teams, with their tracking degraded to resemble
the automatic tracking of the challenge. No labelled data of the challenge match is used. The
fitted models (config.MODELS) hold no match data; a .json next to each describes it:

  models/possession_v4c.pkl  v4 (submitted): 3 Metrica + 7 IDSSE/DFL matches, layer 1 also gets
                             the touch features (src/touch_features.py); bench/v4_final.py
  models/possession_hgb.pkl  v3: 3 Metrica matches; bench/train_model.py

A model needs the touch features if its feature names end with them (uses_touch()).

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
from . import frame_features, stack, touch_features

DECIMALS = 3      # inputs are rounded: same decisions on macOS, Linux and Windows
P_FLOOR = 1e-4    # no state is ever ruled out by a single frame


def check_version(version=None):
    """A version of config.MODELS (None = config.MODEL_VERSION); ValueError if unknown."""
    version = C.MODEL_VERSION if version is None else version
    if version not in C.MODELS:
        raise ValueError("unknown model version {!r}; choose one of {}".format(
            version, sorted(C.MODELS)))
    return version


def model_path(version=None):
    """models/<file> of a version of config.MODELS (None = config.MODEL_VERSION)."""
    return os.path.join(C.ROOT, "models", C.MODELS[check_version(version)]["file"])


MODEL_PATH = model_path("v3")     # written by bench/train_model.py

_CACHE = {}


def load(path=None):
    path = model_path() if path is None else path
    if path not in _CACHE:
        if not os.path.isfile(path):
            raise FileNotFoundError("Model {} is missing (it is committed; refit with "
                                    "bench/v4_final.py or bench/train_model.py)".format(path))
        with open(path, "rb") as fh:
            _CACHE[path] = pickle.load(fh)
    return _CACHE[path]


def load_version(version=None):
    """The fitted model of a version. A model that stores its decoder costs (v4) must agree with
    config.MODELS, so that editing one without the other cannot silently change the output."""
    version = check_version(version)
    bundle = load(model_path(version))
    stored = bundle.get("costs")
    if stored is not None and dict(stored) != C.MODELS[version]["costs"]:
        raise ValueError("model {} was fitted for decoder costs {} but config.MODELS[{!r}] has {}"
                         .format(model_path(version), stored, version,
                                 C.MODELS[version]["costs"]))
    return bundle


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
        raise ValueError("The features do not match the fitted model ({} computed, {} expected; "
                         "with touch features: {}). Refit it (bench/v4_final.py for v4, "
                         "bench/train_model.py for v3).".format(
                             len(names), len(bundle["names"]), uses_touch(bundle)))
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


def matrix(atd, p, pitch, halves, touch=False):
    """Features of every frame. With touch=True the touch features are appended and the frame
    features are stored as float32 first, exactly as the v4 model saw them when it was fitted."""
    ball_xy, _ = ball_mod.clean_ball(atd["ball"], p)
    X, names = frame_features.compute(atd, ball_xy, halves, pitch=pitch)
    if not touch:
        return X, names
    T, tnames = touch_features.features(atd)
    return np.hstack([np.asarray(X, dtype=np.float32), T]), list(names) + list(tnames)


def uses_touch(bundle):
    k = len(touch_features.NAMES)
    return list(bundle["names"])[-k:] == list(touch_features.NAMES)


def evidence(atd, p=None, pitch=None, halves=None, bundle=None):
    """Tracking -> (team_llr, dead_logit) per frame. None = challenge match values (config) and
    the submitted model."""
    p = C.PARAMS if p is None else p
    pitch = C.PITCH if pitch is None else pitch
    halves = C.HALVES if halves is None else halves
    bundle = load_version() if bundle is None else bundle
    X, names = matrix(atd, p, pitch, halves, touch=uses_touch(bundle))
    return to_evidence(predict_proba(X, names, bundle))
