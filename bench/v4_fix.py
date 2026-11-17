"""EXPERIMENTAL: why V4-C makes more false H<->A switches than V3, and ONE conservative correction
(a higher switch cost in the unchanged Viterbi decoder) chosen by nested internal validation.
Nothing here changes the submission, src/ or models/; historical results (outputs/v4/results)
are only read.

    python -m bench.v4_fix --proba              # outer-test probabilities of V3, V4-A, V4-C
    python -m bench.v4_fix --diagnose           # breakdown of the false switches
    python -m bench.v4_fix --inner 0 1 2 3 4    # inner out-of-sample V4-C on each fold's training matches
    python -m bench.v4_fix --select             # nested choice of the cost + paired outer comparison

Pre-registered (fixed before any inner or corrected outer result was seen):
  * the only change is switch_cost, taken from GRID; dead_cost (48), the features, the models and
    the decoder are unchanged. switch_cost is paid only by a direct H<->A change (H -> D -> A
    costs 2 x 48), so it acts precisely on the measure that failed;
  * for outer fold k the cost is chosen ONLY from out-of-sample V4-C probabilities of its 8
    training matches (v4.INNER groups of whole matches, as in v4.fit_v4: layer 1 fitted without
    the group; layer 2 fitted on the other groups). The outer test pair is never used;
  * constraint: inner false H<->A per 90 min <= MAX_GROWTH x that of the stored V3 on the same
    inner DFL matches (V3 never saw DFL; on Metrica it would be in-sample, so Metrica matches are
    left out of the constraint only). Objective: the highest inner mean spellF1 among the costs
    that meet it; if none does, the largest cost of GRID;
  * the corrected candidate (V4-C*) is then judged on the outer folds with the unchanged rule of
    bench/v4_report.py against V3, and against V4-C for spellF1 and short possessions;
  * production cost, if adopted = median of the five inner choices (outer folds never used).

Known, accepted approximation: the inner layer 2 of group g is fitted on layer-1 probabilities of
the other groups, produced by layer-1 models that saw g. This only affects how well the cost is
chosen, never the outer estimate (the outer pair is out of every model involved).
"""

import argparse
import json
import os
import pickle
import time

import numpy as np

from src import config as C
from src import model as M

from . import run, targets, train_model, v4, v4_report

import possession_metrics as pm  # noqa: E402  (path set by bench.run)

FIX = os.path.join(v4.OUT, "fix")
PROBA = os.path.join(FIX, "proba")
INNER_DIR = os.path.join(FIX, "inner")
GRID = [6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 14.0]
MAX_GROWTH = v4_report.MAX_FALSE_GROWTH + 1.0
DEAD = C.MODELS["v3"]["costs"]["dead_cost"]
SOURCES = {"V3": None, "V4-A": ("base", False), "V4-C": ("touch", False)}


def _p(folder, *parts):
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, "_".join(parts) + ".npy")


def proba_path(cand, mid, tag):
    return _p(os.path.join(PROBA, cand), mid, tag)


def inner_path(k, mid, tag):
    return _p(os.path.join(INNER_DIR, "fold{}".format(k)), mid, tag)


# ----------------------------------------------------------------------------- probabilities

def save_proba():
    """Outer-test probabilities from the saved fold models (V4) and the V3 recipe; Metrica V3
    (recipe refitted on the two other Metrica matches) last because it is the slow part."""
    stored = M.load(M.model_path("v3"))
    nm0 = v4.names(False)
    for k, test in enumerate(v4.FOLDS):
        for cand in ("V4-A", "V4-C"):
            kind, extras = SOURCES[cand]
            with open(os.path.join(v4.MOD, "fold{}_{}.pkl".format(k, kind)), "rb") as fh:
                bundle = pickle.load(fh)
            assert not set(bundle["train"]) & set(test)
            for mid in test:
                for tag in v4.EVAL_TAGS:
                    X = np.asarray(v4.item(mid, tag, kind == "touch"))
                    np.save(proba_path(cand, mid, tag), v4.proba_v4(bundle, X, extras))
        for mid in test:
            if mid in v4.DFL:
                for tag in v4.EVAL_TAGS:
                    X = np.asarray(v4.item(mid, tag, False))
                    np.save(proba_path("V3", mid, tag), M.predict_proba(X, nm0, stored))
        v4.log("outer probabilities of fold {} saved".format(k))
    save_v3_metrica(v4.METRICA)


def save_v3_metrica(mids):
    nm0 = v4.names(False)
    for mid in mids:
        if all(os.path.isfile(proba_path("V3", mid, t)) for t in v4.EVAL_TAGS):
            continue
        bundle = v4.fit_v3_metrica([m for m in v4.METRICA if m != mid])
        for tag in v4.EVAL_TAGS:
            X = np.asarray(v4.item(mid, tag, False))
            np.save(proba_path("V3", mid, tag), M.predict_proba(X, nm0, bundle))
        v4.log("V3 recipe probabilities of {} saved".format(mid))


def decode_costs(P, halves, costs):
    tl, dl = M.to_evidence(P)
    return run.decode_grid(tl, dl, halves, [{"switch_cost": c, "dead_cost": DEAD} for c in costs])


# ----------------------------------------------------------------------------- diagnosis

def _bucket_duration(s):
    return "< 2 s" if s < 2 else "2-5 s" if s < 5 else "5-10 s" if s < 10 else ">= 10 s"


def false_switch_details(gt_labels, pred_labels, X, T, nm, tnm):
    """One dict per false H<->A switch (same definition as v4.extra_metrics)."""
    gt, pr = pm.label_codes(gt_labels), pm.label_codes(pred_labels)
    runs_ = pm.valid_runs(gt)
    pr = pm.fill_abstentions(pr, runs_)
    all_gt = pm.transitions(gt, runs_)
    tg = [t for t in all_gt if t[2] in (pm.H, pm.A)]
    tp_all = pm.transitions(pr, runs_)
    tp = [t for t in tp_all if {t[1], t[2]} == {pm.H, pm.A}]
    _, _, spurious = pm.match_transitions(tg, tp, v4.SWITCH_TOL, typed="to")
    pred_frames = np.array(sorted(t[0] for t in tp_all))
    starts = np.array([start for start, _ in runs_])
    stops = np.array([stop for _, stop in runs_])
    seen, since = X[:, nm.index("g_raw_seen")], X[:, nm.index("g_since_seen")]
    tsince, tuntil = T[:, tnm.index("g_tsince")], T[:, tnm.index("g_tuntil")]
    out = []
    for f, a, b in spurious:
        stop = int(stops[np.searchsorted(starts, f, side="right") - 1])
        nxt = pred_frames[pred_frames > f]
        end = min(int(nxt[0]), stop) if len(nxt) else stop
        same = [abs(t[0] - f) for t in tg if t[2] == b]
        d = min(same) if same else 10 ** 9
        if d <= v4.SWITCH_TOL:
            cat = "true change, already matched (duplicate)"
        elif d <= 125:
            cat = "true change shifted 2-5 s"
        elif d <= 250:
            cat = "true change shifted 5-10 s"
        elif gt[f] == b:
            cat = "spurious: return to the true team"
        elif gt[f] == a:
            cat = "spurious: excursion away from the true team"
        else:
            cat = "spurious: during dead ball"
        vis = "visible" if seen[f] > 0 else ("lost < 1 s" if since[f] <= 25 else "lost >= 1 s")
        out.append({"cat": cat, "dur": _bucket_duration((end - f) / 25.0), "vis": vis,
                    "touch": bool(min(tsince[f], tuntil[f]) <= 1.0)})
    return out


def diagnose():
    nm, tnm = v4.names(False), v4.names(True)[len(v4.names(False)):]
    rows = []
    for mid in v4.ALL:
        halves, _ = v4.meta(mid)
        truth = targets.all_targets(v4.labels(mid))
        for tag in v4.EVAL_TAGS:
            X = np.load(v4._path(mid, tag, "base"), mmap_mode="r")
            T = np.load(v4._path(mid, tag, "touch"))
            for cand in SOURCES:
                pred = decode_costs(np.load(proba_path(cand, mid, tag)), halves, [6.0])[0]
                for tname, gt in truth.items():
                    minutes = float(np.sum(pm.label_codes(gt) != pm.X)) / 25.0 / 60.0
                    det = false_switch_details(gt, pred, X, T, nm, tnm)
                    rows.append({"cand": cand, "match": mid, "provider": v4.provider(mid),
                                 "eval": tag, "target": tname, "minutes": minutes,
                                 "false": det})
        v4.log("diagnosed", mid)
    with open(os.path.join(FIX, "diag.json"), "w") as fh:
        json.dump(rows, fh)
    print_diag(rows)


def print_diag(rows=None):
    if rows is None:
        with open(os.path.join(FIX, "diag.json")) as fh:
            rows = json.load(fh)

    def rate(cand, prov, key=None, val=None):
        sel = [r for r in rows if r["cand"] == cand and (prov is None or r["provider"] == prov)]
        n = sum(1 for r in sel for d in r["false"] if key is None or d[key] == val)
        return 90.0 * n / sum(r["minutes"] for r in sel)

    out = {}
    for prov in (None, "metrica", "dfl"):
        out["all" if prov is None else prov] = {c: rate(c, prov) for c in SOURCES}
    for key in ("cat", "dur", "vis", "touch"):
        vals = sorted({d[key] for r in rows for d in r["false"]}, key=str)
        out[key] = {str(v): {c: rate(c, None, key, v) for c in SOURCES} for v in vals}
        for prov in ("metrica", "dfl"):
            out[key + "_" + prov] = {str(v): {c: rate(c, prov, key, v) for c in SOURCES}
                                     for v in vals}
    with open(os.path.join(FIX, "diag_summary.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    for section, d in out.items():
        print("\n##", section)
        for k, v in d.items():
            print("  {:<48}".format(k) if isinstance(v, dict) else "", end="")
            if isinstance(v, dict):
                print("  ".join("{} {:6.2f}".format(c, x) for c, x in v.items()))
            else:
                print("  {} {:6.2f}".format(k, v))
    return out


# ----------------------------------------------------------------------------- nested selection

def inner(k):
    """Out-of-sample V4-C probabilities of the 8 training matches of outer fold k."""
    test = list(v4.FOLDS[k])
    train = sorted(m for m in v4.ALL if m not in test)
    if all(os.path.isfile(inner_path(k, m, t)) for m in train for t in v4.EVAL_TAGS):
        v4.log("inner fold {} already done".format(k))
        return
    groups = [train[i::v4.INNER] for i in range(v4.INNER)]
    nm = v4.names(True)
    t0 = time.time()
    P1 = {}
    for g in groups:
        others = [m for m in train if m not in g]
        assert not set(others) & set(g) and not set(others) & set(test)
        s1 = v4.fit_stage1(others, True, v4.STEP_V4)
        for mid in g:
            for tag in v4.TRAIN_TAGS + v4.EVAL_TAGS:
                P1[(mid, tag)] = M.stage1(s1, np.asarray(v4.item(mid, tag, True)), nm)
        v4.log("  fold {}: inner layer 1 without {} ({:.0f} s)".format(k, g, time.time() - t0))
    for g in groups:
        blocks = []
        for mid in train:
            if mid in g:
                continue
            lab = v4.labels(mid)
            idx = v4._rows(lab, v4.STEP_V4)
            for tag in v4.TRAIN_TAGS:
                Fm, Fs = v4.stage2_matrix(P1[(mid, tag)], np.asarray(v4.item(mid, tag, True)),
                                          nm, False)
                blocks.append((Fm[idx], Fs[idx], v4._y(lab, idx)))
        s2 = v4._fit(lambda: iter(blocks), sum(len(b[2]) for b in blocks), blocks[0][0].shape[1],
                     train_model.HGB2)
        for mid in g:
            for tag in v4.EVAL_TAGS:
                Fm, Fs = v4.stage2_matrix(P1[(mid, tag)], np.asarray(v4.item(mid, tag, True)),
                                          nm, False)
                np.save(inner_path(k, mid, tag), M.sym_proba(s2, Fm, Fs))
        v4.log("  fold {}: inner layer 2 for {} ({:.0f} s)".format(k, g, time.time() - t0))


def score_rows(mid, pred, cand, tag):
    rows = []
    v4.score_labels(mid, pred, rows, cand, tag)
    return rows


def choose_cost(k):
    test = list(v4.FOLDS[k])
    train = sorted(m for m in v4.ALL if m not in test)
    rows = []
    for mid in train:
        halves, _ = v4.meta(mid)
        for tag in v4.EVAL_TAGS:
            preds = decode_costs(np.load(inner_path(k, mid, tag)), halves, GRID)
            for c, pred in zip(GRID, preds):
                rows += score_rows(mid, pred, c, tag)
            if mid in v4.DFL:
                ref = decode_costs(np.load(proba_path("V3", mid, tag)), halves, [6.0])[0]
                rows += score_rows(mid, ref, "V3", tag)
    dfl = [m for m in train if m in v4.DFL]
    limit = MAX_GROWTH * v4_report.false_rate(rows, "V3", dfl)
    table = []
    for c in GRID:
        spell = float(np.mean([r["spell_f1"] for r in rows if r["candidate"] == c]))
        table.append({"cost": c, "spell_f1": spell,
                      "false_dfl": v4_report.false_rate(rows, c, dfl),
                      "false_all": v4_report.false_rate(rows, c, train)})
    ok = [t for t in table if t["false_dfl"] <= limit]
    best = max(ok, key=lambda t: t["spell_f1"])["cost"] if ok else GRID[-1]
    return {"fold": k, "train": train, "v3_false_dfl": limit / MAX_GROWTH, "limit": limit,
            "table": table, "cost": best}


def select():
    choices = [choose_cost(k) for k in range(len(v4.FOLDS))]
    hist, _ = v4_report.load_rows()
    rows = [r for r in hist if r["candidate"] in ("V3", "V4-C")]
    check = []
    for ch in choices:
        for mid in v4.FOLDS[ch["fold"]]:
            halves, _ = v4.meta(mid)
            for tag in v4.EVAL_TAGS:
                pred, base = decode_costs(np.load(proba_path("V4-C", mid, tag)), halves,
                                          [ch["cost"], 6.0])
                rows += score_rows(mid, pred, "V4-C*", tag)
                check += score_rows(mid, base, "V4-C", tag)
    # the saved probabilities must reproduce the historical V4-C exactly
    old = {(r["match"], r["eval"], r["target"]): r["spell_f1"] for r in hist
           if r["candidate"] == "V4-C"}
    worst = max(abs(old[(r["match"], r["eval"], r["target"])] - r["spell_f1"]) for r in check)
    assert worst < 1e-9, "saved probabilities do not reproduce the historical V4-C ({})".format(
        worst)
    matches = list(v4.ALL)
    spell = v4_report.per_match(rows, "spell_f1")
    res = {"choices": choices, "production_cost": float(np.median([c["cost"] for c in choices])),
           "vs_V3": v4_report.decide(rows, spell, "V4-C*", "V3", matches),
           "vs_V4C": v4_report.decide(rows, spell, "V4-C*", "V4-C", matches),
           "spell": spell, "macro": v4_report.per_match(rows, "macro_f1")}
    for c in ("V3", "V4-C", "V4-C*"):
        s = v4_report.summed(rows, c, ["short_n", "short_hit", "vshort_n", "vshort_hit",
                                       "ha_pred", "ha_false", "minutes", "n_pred"])
        res[c] = dict(s, false_90=v4_report.false_rate(rows, c),
                      false_90_metrica=v4_report.false_rate(rows, c, v4.METRICA),
                      false_90_dfl=v4_report.false_rate(rows, c, v4.DFL),
                      spell_all=float(np.mean([spell[c][m] for m in matches])))
    for d in (res["vs_V3"], res["vs_V4C"]):
        d["checks"] = {k: bool(v) for k, v in d["checks"].items()}
        d["adopted"] = bool(d["adopted"])
    with open(os.path.join(FIX, "select.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    with open(os.path.join(FIX, "rows.json"), "w") as fh:
        json.dump(rows, fh)
    for ch in choices:
        print("fold", ch["fold"], "chosen cost", ch["cost"], "limit", round(ch["limit"], 2),
              [(t["cost"], round(t["spell_f1"], 4), round(t["false_dfl"], 1))
               for t in ch["table"]])
    print(json.dumps({k: res[k] for k in ("production_cost", "vs_V3", "vs_V4C")}, indent=1))
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--proba", action="store_true")
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--inner", type=int, nargs="*", default=None)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--v3-metrica", nargs="*", default=None)
    args = ap.parse_args(argv)
    if args.proba:
        save_proba()
    if args.v3_metrica:
        save_v3_metrica(args.v3_metrica)
    if args.diagnose:
        diagnose()
    for k in args.inner or []:
        inner(k)
    if args.select:
        select()


if __name__ == "__main__":
    main()
