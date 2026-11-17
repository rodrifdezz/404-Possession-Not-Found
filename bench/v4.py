"""EXPERIMENTAL v4 ablation bench: does training on Metrica + DFL, adding touch features
(src/touch_features.py) or adding possession-change features to layer 2 (bench/v4_stack.py) beat
the stored v3 model? (The touch features now live in src/touch_features.py; the winner, V4-C,
is the submitted model: bench/v4_final.py.)

    python -m bench.v4 --features            # cache features of the 10 matches (outputs/v4/)
    python -m bench.v4 --folds 0             # pilot: one outer fold
    python -m bench.v4 --folds 0 1 2 3 4     # full validation (resumes finished folds)
    python -m bench.v4 --report              # docs/BENCH_V4.md from outputs/v4/results/
    python -m bench.v4 --touch-diag          # how reliable the touch detector is (external data)

Protocol (identical for every candidate, results are paired):
  * 10 external matches (3 Metrica, 7 IDSSE/DFL), 5 outer folds of 2 WHOLE matches. A model
    scored on a match never saw it, in either layer.
  * Every held-out match is degraded exactly as in bench/run.py (2 gap scenarios x seeds 0, 1,
    never used for training) and scored with the official code against both readings of the
    truth (bench/targets.py). The decoder is NOT changed: the v3 costs (config.MODELS["v3"]).
  * V3 = the stored v3 recipe (3 Metrica, step 3): on DFL matches the stored model itself; on a
    Metrica match the same recipe fitted on the two other Metrica matches (as BENCH_MODEL_CV).
  * V4-* = 7 training degradations of the 8 matches of the fold, step 6 (the same number of
    training rows as v3 with more matches). Layer 2 is fitted on out-of-sample layer-1
    probabilities from 3 inner groups of whole matches (no match predicts itself).
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
from src import touch_features as v4_touch

from . import degrade, idsse, metrica, run, targets, train_model, v4_stack

import possession_metrics as pm  # noqa: E402  (path set by bench.run)

OUT = os.path.join(C.OUT_DIR, "v4")
FEAT = os.path.join(OUT, "feat")
RES = os.path.join(OUT, "results")
MOD = os.path.join(OUT, "models")

METRICA = ["M1", "M2", "M3"]
DFL = list(idsse.MATCHES)
ALL = METRICA + DFL
FOLDS = [("M1", "J03WMX"), ("M2", "J03WN1"), ("M3", "J03WOH"),
         ("J03WOY", "J03WPY"), ("J03WQQ", "J03WR9")]
TRAIN_TAGS = ["tr{}".format(v[0]) for v in train_model.VARIANTS]
EVAL = [(sc, sd) for sc in degrade.SCENARIOS for sd in run.SEEDS]
EVAL_TAGS = ["ev_{}_{}".format(sc, sd) for sc, sd in EVAL]
STEP_V4 = 6
INNER = 3
COSTS = dict(C.MODELS["v3"]["costs"])     # the decoder of v3, unchanged in this ablation
CANDIDATES = {"V4-A": (False, False), "V4-B": (False, True),
              "V4-C": (True, False), "V4-D": (True, True)}     # (touch, layer-2 extras)
SHORT_S, VSHORT_S = 5.0, 2.0
SWITCH_TOL = 50


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def provider(mid):
    return "metrica" if mid in METRICA else "dfl"


def load_match(mid):
    return metrica.load(int(mid[1:])) if mid in METRICA else idsse.load(mid)


# ----------------------------------------------------------------------------- feature cache

def _path(mid, tag, kind):
    return os.path.join(FEAT, mid, "{}_{}.npy".format(tag, kind))


def train_variant(m, seed, kind, arg, keep):
    """bench.train_model.variant(); the DFL ball is tracked during stoppages, nothing to hide."""
    if "ball_raw" not in m:
        keep = 1.0
    return train_model.variant(m, seed, kind, arg, keep)


def cache_match(mid):
    folder = os.path.join(FEAT, mid)
    os.makedirs(folder, exist_ok=True)
    jobs = [("tr{}".format(v[0]), v) for v in train_model.VARIANTS] + \
           [("ev_{}_{}".format(sc, sd), (sc, sd)) for sc, sd in EVAL]
    if all(os.path.isfile(_path(mid, t, "touch")) for t, _ in jobs):
        return mid
    t0 = time.time()
    m = load_match(mid)
    np.save(os.path.join(folder, "labels.npy"), m["labels"])
    with open(os.path.join(folder, "meta.json"), "w") as fh:
        json.dump({"halves": [list(map(int, h)) for h in m["halves"]],
                   "pitch": list(map(float, m["pitch"]))}, fh)
    for tag, spec in jobs:
        if os.path.isfile(_path(mid, tag, "touch")):
            continue
        d = train_variant(m, *spec) if tag.startswith("tr") else degrade.degrade(m, *spec)
        X, names = M.matrix(d, C.PARAMS, m["pitch"], m["halves"])
        T, tnames = v4_touch.features(d)
        np.save(_path(mid, tag, "base"), X.astype(np.float32))
        np.save(_path(mid, tag, "touch"), T)
        with open(os.path.join(FEAT, "names.json"), "w") as fh:
            json.dump({"base": names, "touch": tnames}, fh)
    log("features of {} cached ({:.0f} s)".format(mid, time.time() - t0))
    return mid


def build_cache(mids, workers):
    from multiprocessing import Pool
    os.makedirs(FEAT, exist_ok=True)
    if workers <= 1:
        for mid in mids:
            cache_match(mid)
        return
    with Pool(workers) as pool:
        for mid in pool.imap_unordered(cache_match, mids):
            log("done", mid)


def names(touch):
    with open(os.path.join(FEAT, "names.json")) as fh:
        nm = json.load(fh)
    return nm["base"] + (nm["touch"] if touch else [])


def item(mid, tag, touch):
    X = np.load(_path(mid, tag, "base"), mmap_mode="r")
    if touch:
        X = np.concatenate([X, np.load(_path(mid, tag, "touch"))], axis=1)
    return X


def labels(mid):
    return np.load(os.path.join(FEAT, mid, "labels.npy"))


def meta(mid):
    with open(os.path.join(FEAT, mid, "meta.json")) as fh:
        z = json.load(fh)
    return [tuple(h) for h in z["halves"]], tuple(z["pitch"])


# ----------------------------------------------------------------------------- fitting

def _rows(lab, step):
    return np.flatnonzero(lab != "X")[::step]


def _fit(blocks_fn, n_rows, n_feat, hgb):
    """HistGradientBoosting on the rows produced by blocks_fn() (each block added twice: as is
    and with the teams swapped), filled into one preallocated array."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    Xall = np.empty((2 * n_rows, n_feat), dtype=np.float64)
    yall = np.empty(2 * n_rows, dtype=np.int64)
    i = 0
    for Xb, Xs, y in blocks_fn():
        k = len(y)
        Xall[i:i + k] = M.prepare(Xb)
        Xall[i + k:i + 2 * k] = M.prepare(Xs)
        yall[i:i + k] = y
        yall[i + k:i + 2 * k] = np.where(y == 0, 1, np.where(y == 1, 0, 2))
        i += 2 * k
    assert i == 2 * n_rows
    return HistGradientBoostingClassifier(**hgb).fit(Xall, yall)


def _y(lab, idx):
    return np.array([train_model.CODE[c] for c in lab[idx]])


def fit_stage1(mids, touch, step):
    nm = names(touch)
    labs = {m: labels(m) for m in mids}
    n = sum(len(_rows(labs[m], step)) for m in mids) * len(TRAIN_TAGS)

    def blocks():
        for mid in mids:
            idx = _rows(labs[mid], step)
            for tag in TRAIN_TAGS:
                X = np.asarray(item(mid, tag, touch)[idx])
                yield X, F.swap_teams(X, nm), _y(labs[mid], idx)
    return _fit(blocks, n, len(nm), train_model.HGB1)


def stage2_matrix(P1, X, nm, extras):
    Fm = S.features(P1, X, nm)
    if not extras:
        return Fm, S.swap_teams(Fm)
    E = v4_stack.features(P1)
    return np.hstack([Fm, E]), np.hstack([S.swap_teams(Fm), v4_stack.swap_teams(E)])


def fit_v4(train, touch, step=STEP_V4, layer2=(False, True)):
    """Layer 1 on all `train`; layer 2 (without and/or with the extras, `layer2`) on
    out-of-sample layer-1 probabilities: the training matches are split into INNER groups of
    whole matches and each group is predicted by a layer 1 fitted on the other groups."""
    nm = names(touch)
    train = sorted(train)
    groups = [train[i::INNER] for i in range(INNER)]
    t0 = time.time()
    s1 = fit_stage1(train, touch, step)
    log("    layer 1 on {} matches ({:.0f} s)".format(len(train), time.time() - t0))
    blocks = {e: [] for e in layer2}
    for g in groups:
        others = [m for m in train if m not in g]
        assert not set(others) & set(g)
        oof = fit_stage1(others, touch, step)
        for mid in g:
            lab = labels(mid)
            idx = _rows(lab, step)
            for tag in TRAIN_TAGS:
                X = np.asarray(item(mid, tag, touch))
                P1 = M.stage1(oof, X, nm)
                for extras in layer2:
                    Fm, Fs = stage2_matrix(P1, X, nm, extras)
                    blocks[extras].append((Fm[idx], Fs[idx], _y(lab, idx)))
        log("    inner group {} predicted out of sample ({:.0f} s)".format(g, time.time() - t0))
    s2 = {}
    for extras in layer2:
        b = blocks[extras]
        s2[extras] = _fit(lambda b=b: iter(b), sum(len(y) for _, _, y in b), b[0][0].shape[1],
                          train_model.HGB2)
    log("    layer 2 fitted ({:.0f} s)".format(time.time() - t0))
    return {"stage1": s1, "stage2": s2, "names": nm, "train": train, "touch": touch}


def fit_v3_metrica(train):
    """The v3 recipe (bench/train_model.fit_two_layers, step 3) on Metrica matches only."""
    nm = names(False)
    data = {}
    for mid in train:
        for (seed, *_), tag in zip(train_model.VARIANTS, TRAIN_TAGS):
            data[(mid, seed)] = {"lab": labels(mid), "X": item(mid, tag, False), "names": nm}
    return train_model.fit_two_layers(data, sorted(train))


def proba_v4(bundle, X, extras):
    nm = bundle["names"]
    P1 = M.stage1(bundle["stage1"], X, nm)
    Fm, Fs = stage2_matrix(P1, X, nm, extras)
    return M.sym_proba(bundle["stage2"][extras], Fm, Fs)


# ----------------------------------------------------------------------------- scoring

def extra_metrics(gt_labels, pred_labels):
    """Short possessions recovered (IoU >= 0.5) and false H<->A switches (a predicted direct
    change of team with no true change towards that team within +-2 s)."""
    gt, pr = pm.label_codes(gt_labels), pm.label_codes(pred_labels)
    runs_ = pm.valid_runs(gt)
    pr = pm.fill_abstentions(pr, runs_)
    gs, ps = pm.possession_spells(gt, runs_), pm.possession_spells(pr, runs_)
    hit = pm.matched_gt_flags(pm.spell_overlaps(gs, ps), len(gs), 0.5)
    dur = np.array([b - a for a, b, _ in gs]) / 25.0
    tg = [t for t in pm.transitions(gt, runs_) if t[2] in (pm.H, pm.A)]
    tp = [t for t in pm.transitions(pr, runs_) if {t[1], t[2]} == {pm.H, pm.A}]
    _, _, spurious = pm.match_transitions(tg, tp, SWITCH_TOL, typed="to")
    n_true_ha = sum(1 for t in pm.transitions(gt, runs_) if {t[1], t[2]} == {pm.H, pm.A})
    graded_min = float(np.sum(gt != pm.X)) / 25.0 / 60.0
    return {"short_n": int(np.sum(dur < SHORT_S)), "short_hit": int(np.sum(hit & (dur < SHORT_S))),
            "vshort_n": int(np.sum(dur < VSHORT_S)),
            "vshort_hit": int(np.sum(hit & (dur < VSHORT_S))),
            "ha_pred": len(tp), "ha_false": len(spurious), "ha_true": n_true_ha,
            "minutes": graded_min}


def score_labels(mid, pred, rows, cand, tag):
    truth = targets.all_targets(labels(mid))
    for tname, gt in truth.items():
        r = {"candidate": cand, "match": mid, "provider": provider(mid), "eval": tag,
             "target": tname}
        r.update(run.score(gt, pred))
        r.update(extra_metrics(gt, pred))
        rows.append(r)


def decode(team_llr, dead_logit, halves):
    return run.decode_grid(team_llr, dead_logit, halves, [COSTS])[0]


def run_fold(k):
    path = os.path.join(RES, "fold{}.json".format(k))
    if os.path.isfile(path):
        log("fold {} already done".format(k))
        return
    test = list(FOLDS[k])
    train = [m for m in ALL if m not in test]
    assert not set(train) & set(test) and len(train) == 8
    log("fold {}: test {} | train {}".format(k, test, train))
    rows = []
    os.makedirs(MOD, exist_ok=True)
    # V3
    stored = M.load(M.model_path("v3"))
    v3_metrica = {}
    for mid in test:
        if mid in METRICA:
            tr = [m for m in METRICA if m != mid]
            t0 = time.time()
            v3_metrica[mid] = fit_v3_metrica(tr)
            log("  V3 recipe fitted on {} ({:.0f} s)".format(tr, time.time() - t0))
    for mid in test:
        halves, _ = meta(mid)
        bundle = v3_metrica.get(mid, stored)
        for tag in EVAL_TAGS:
            X = np.asarray(item(mid, tag, False))
            P = M.predict_proba(X, names(False), bundle)
            score_labels(mid, decode(*M.to_evidence(P), halves), rows, "V3", tag)
    log("  V3 scored")
    # V4: one layer-1 per feature set, two layer-2 per layer-1
    for touch in (False, True):
        mpath = os.path.join(MOD, "fold{}_{}.pkl".format(k, "touch" if touch else "base"))
        if os.path.isfile(mpath):
            with open(mpath, "rb") as fh:
                bundle = pickle.load(fh)
        else:
            log("  fitting V4 ({} features)".format("touch" if touch else "original"))
            bundle = fit_v4(train, touch)
            with open(mpath, "wb") as fh:
                pickle.dump(bundle, fh, protocol=4)
        assert not set(bundle["train"]) & set(test)
        for cand, (ct, extras) in CANDIDATES.items():
            if ct != touch:
                continue
            for mid in test:
                halves, _ = meta(mid)
                for tag in EVAL_TAGS:
                    P = proba_v4(bundle, np.asarray(item(mid, tag, touch)), extras)
                    score_labels(mid, decode(*M.to_evidence(P), halves), rows, cand, tag)
            log("  {} scored".format(cand))
    os.makedirs(RES, exist_ok=True)
    with open(path, "w") as fh:
        json.dump({"fold": k, "test": test, "train": train, "rows": rows}, fh, indent=0)
    log("fold {} written".format(k))


# ----------------------------------------------------------------------------- touch diagnostics

def touch_diag():
    """Reliability of the detector on external matches (their labels are used ONLY to measure,
    never to detect): touches per minute and agreement of the touch team with the team in
    possession 0.5 s later, on the clean tracking and on the evaluation degradations."""
    out = []
    for mid in ALL:
        m = load_match(mid)
        lab = m["labels"]
        graded_min = np.sum(lab != "X") / 25.0 / 60.0
        versions = [("clean", {"players": m["players"], "team": m["team"],
                               "ball": m.get("ball_raw", m["ball"])})]
        versions += [("ev_{}_{}".format(sc, sd), degrade.degrade(m, sc, sd)) for sc, sd in EVAL]
        for name, d in versions:
            t = v4_touch.detect(d)
            fr, tm = t["frame"], t["team"]
            after = np.clip(fr + 12, 0, len(lab) - 1)
            alive = np.isin(lab[after], ["H", "A"]) & (tm != 0)
            agree = np.mean(np.where(lab[after][alive] == "H", 1, -1) == tm[alive]) \
                if alive.any() else np.nan
            out.append({"match": mid, "version": name, "per_min": len(fr) / graded_min,
                        "team_agree": float(agree), "assigned": float(np.mean(tm != 0))
                        if len(tm) else np.nan})
        log("touch diagnostics for {}".format(mid))
    os.makedirs(RES, exist_ok=True)
    with open(os.path.join(RES, "touch_diag.json"), "w") as fh:
        json.dump(out, fh, indent=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--features", action="store_true")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--touch-diag", action="store_true")
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args(argv)
    if args.features:
        build_cache(ALL, args.workers)
    if args.touch_diag:
        touch_diag()
    for k in args.folds or []:
        run_fold(k)
    if args.report:
        from . import v4_report
        v4_report.write()


if __name__ == "__main__":
    sys.exit(main())
