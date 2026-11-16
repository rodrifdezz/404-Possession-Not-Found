"""Fit the learned evidence (src/model.py) on external matches, and validate it.

    python -m bench.metrica --download     # once: the 3 Metrica sample matches (~250 MB)
    python -m bench.train_model --cv       # leave-one-match-out validation -> docs/BENCH_MODEL_CV.md
    python -m bench.train_model            # fit on the 3 matches -> models/possession_hgb.pkl

Training data: the Metrica Sports open sample matches (bench/metrica.py), OTHER matches than the
challenge one. Each is degraded in seven different ways (VARIANTS) so that the model does not
depend on the quirks of one simulated tracker. Nothing here reads the challenge match.

Validation (--cv): the model is fitted on two matches and scored on the third with exactly the
protocol of bench/run.py (scenarios, seeds, both readings of the truth, official scorer), next to
the rule-based evidence on the same degraded data. The degradation seeds used for scoring are
never used for training.

The fitted model is committed, so make_submission never trains. Everything is deterministic.
"""

import argparse
import json
import os
import pickle
import sys
import time

import numpy as np

from src import config as C
from src import frame_features as F
from src import model as M
from src import stack as S

from . import degrade, metrica, run

CODE = {"H": 0, "A": 1, "D": 2}

# (seed, degradation, argument, share of stoppages in which the ball is kept visible)
VARIANTS = [
    (101, "markov", "normal", 1.0),
    (102, "markov", "normal", 0.5),
    (103, "markov", "easy", 1.0),
    (104, "markov", "hard", 0.5),
    (105, "markov", "normal", 0.0),
    (106, "scenario", "short_gaps", 1.0),
    (107, "scenario", "long_gaps", 0.5),
]

HGB1 = dict(max_iter=250, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=100,
            l2_regularization=1.0, random_state=0)
HGB2 = dict(max_iter=200, learning_rate=0.08, max_leaf_nodes=31, min_samples_leaf=200,
            l2_regularization=1.0, random_state=0)
STEP = 3      # 1 frame in 3 is used for fitting (neighbouring frames are almost identical)


def log(*a):
    print(*a, flush=True)


def hide_stoppage_ball(m, keep, seed):
    """Copy of m where the rebuilt ball (metrica.fill_dead_ball) is kept in a share `keep` of the
    holes of the original file and removed again in the rest."""
    if keep >= 1.0:
        return m
    rng = np.random.default_rng(seed)
    ball = m["ball"].copy()
    nan = np.isnan(m["ball_raw"][:, 0])
    pad = np.diff(np.concatenate(([0], nan.astype(int), [0])))
    for s, e in zip(np.flatnonzero(pad == 1), np.flatnonzero(pad == -1)):
        if rng.random() >= keep:
            ball[s:e] = np.nan
    return dict(m, ball=ball)


def variant(m, seed, kind, arg, keep):
    mm = hide_stoppage_ball(m, keep, seed + 1000)
    return (degrade.degrade_markov if kind == "markov" else degrade.degrade)(mm, arg, seed)


def features_of(m, d):
    X, names = M.matrix(d, C.PARAMS, m["pitch"], m["halves"])
    return {"lab": m["labels"], "X": X, "names": names}


def rows(item):
    return np.flatnonzero(item["lab"] != "X")[::STEP]


def targets_of(item, idx):
    y = np.array([CODE[c] for c in item["lab"][idx]])
    return y, np.where(y == 0, 1, np.where(y == 1, 0, 2))


def fit_stage1(items):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys = [], []
    for it in items:
        idx = rows(it)
        y, y_swapped = targets_of(it, idx)
        Xs += [M.prepare(it["X"][idx]), M.prepare(F.swap_teams(it["X"][idx], it["names"]))]
        ys += [y, y_swapped]
    return HistGradientBoostingClassifier(**HGB1).fit(np.concatenate(Xs), np.concatenate(ys))


def fit_stage2(items, probs):
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xs, ys = [], []
    for it, P1 in zip(items, probs):
        Fm = S.features(P1, it["X"], it["names"])
        idx = rows(it)
        y, y_swapped = targets_of(it, idx)
        Xs += [M.prepare(Fm[idx]), M.prepare(S.swap_teams(Fm[idx]))]
        ys += [y, y_swapped]
    return HistGradientBoostingClassifier(**HGB2).fit(np.concatenate(Xs), np.concatenate(ys))


def fit_two_layers(data, games):
    """Layer 1 on all `games`; layer 2 on out-of-sample layer-1 probabilities (per match)."""
    names = data[(games[0], VARIANTS[0][0])]["names"]
    s1 = fit_stage1([data[(g, v[0])] for g in games for v in VARIANTS])
    items, probs = [], []
    for g in games:
        others = [o for o in games if o != g]
        oof = fit_stage1([data[(o, v[0])] for o in others for v in VARIANTS])
        for v in VARIANTS:
            it = data[(g, v[0])]
            items.append(it)
            probs.append(M.stage1(oof, it["X"], names))
    return {"stage1": s1, "stage2": fit_stage2(items, probs), "names": names}


def cross_validate(matches, data, quick):
    cfgs = run.configs(run.QUICK_GRID if quick else run.GRID)
    seeds = run.SEEDS[:1] if quick else run.SEEDS
    games = sorted(matches)
    per_match, reference = [], []
    t0 = time.time()
    for test in games:
        bundle = fit_two_layers(data, [g for g in games if g != test])

        def evidence(d, m, bundle=bundle):
            return M.evidence(d, C.PARAMS, m["pitch"], m["halves"], bundle)

        scores, ref = run.run_match(matches[test], cfgs, seeds, log=lambda *a: None,
                                    reference=True, evidence=evidence)
        per_match.append(scores)
        reference.append(ref)
        log("  match {} scored ({:.0f} s)".format(test, time.time() - t0))
    suffix = "_QUICK" if quick else ""
    note = ("Leave-one-match-out for the model itself: for every match, both layers are fitted "
            "on the other matches and scored on this one.")
    best = run.report(games, cfgs, per_match,
                      os.path.join(C.ROOT, "docs", "BENCH_MODEL_CV{}.md".format(suffix)),
                      os.path.join(C.OUT_DIR, "bench", "results_model_cv{}.json".format(suffix)),
                      reference, "model", "metrica", note, "python -m bench.train_model --cv")
    log("best decoder costs for the model evidence:", run.fmt(best))
    log("-> copy them into MODEL_COSTS in src/config.py")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cv", action="store_true")
    ap.add_argument("--quick", action="store_true", help="with --cv: small grid, one seed")
    args = ap.parse_args(argv)
    games = metrica.available()
    if len(games) < 2:
        sys.exit("Matches missing (download them with: python -m bench.metrica --download).")

    t0 = time.time()
    matches, data = {}, {}
    for g in games:
        matches[g] = metrica.load(g)
        for seed, kind, arg, keep in VARIANTS:
            data[(g, seed)] = features_of(matches[g], variant(matches[g], seed, kind, arg, keep))
        log("match {}: {} training variants ready ({:.0f} s)".format(g, len(VARIANTS),
                                                                    time.time() - t0))
    if args.cv:
        return cross_validate(matches, data, args.quick)

    bundle = fit_two_layers(data, games)
    os.makedirs(os.path.dirname(M.MODEL_PATH), exist_ok=True)
    with open(M.MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    import sklearn
    meta = {"model": "2 x HistGradientBoostingClassifier (stacked)", "stage1": HGB1,
            "stage2": HGB2, "features": bundle["names"], "classes": ["H", "A", "D"],
            "training_data": "github.com/metrica-sports/sample-data, games {}".format(games),
            "variants": [list(v) for v in VARIANTS], "frame_step": STEP,
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "python": sys.version.split()[0]}
    with open(M.MODEL_PATH.replace(".pkl", ".json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    log("model written to {} ({:.0f} s)".format(os.path.relpath(M.MODEL_PATH, C.ROOT),
                                                time.time() - t0))


if __name__ == "__main__":
    main()
