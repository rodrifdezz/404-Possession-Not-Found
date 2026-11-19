"""Interactive demo of the possession model: python -m bench.demo

    python -m bench.metrica --download     # once: the open Metrica sample matches
    python -m bench.demo                   # -> outputs/demo/possession_demo.html (~10 min first time)
    python -m bench.demo --game 3 --start 50 --minutes 5   # game 3, from minute 50, 5 minutes

One self-contained HTML page (open it in any browser, no server) with two scenes:

1. REPLAY on a public match (Metrica Sports open sample data, game 2 by default). The tracking is
   degraded exactly as in the external bench (bench/degrade.py: ball lost in bursts, noise,
   dropouts), i.e. what the model sees. The model is the v4 recipe (frame + touch features, two
   layers, decoder costs of config.MODELS["v4c"]) fitted ONLY on the other Metrica matches, so
   the replayed match is unseen. The submitted model was fitted on all ten external matches,
   this one included, so it is not used here: its scores on this match would be optimistic.
   Shown live: players and ball, detected touches, P(H) / P(A) / P(D), the decoded possession
   next to the truth and the official baseline, and spellF1 / macroF1 computed by the official
   scorer on the elapsed part.

2. CHALLENGE, public window. Only LABELS: the public ground truth of the first 5 minutes, the
   official baselines and our submission file, scored with the official code. No tracking, no
   video and nothing else from the licensed match data is written to the page.

Everything is deterministic: the same command gives the same page.
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
from src import hmm
from src import model as M
from src import stack as S
from src import touch_features

from . import degrade, metrica, train_model

OUT = os.path.join(C.OUT_DIR, "demo")
TEMPLATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "demo_template.html")
FPS = 25
STEP = 3                     # training rows: 1 frame in 3
PLAYER_FPS = 25              # frames written to the page (every frame: smooth animation)
CHECK_EVERY_S = 5            # running metrics every 5 s of match time
SCENARIO, SEED = "long_gaps", 0   # degradation of the replayed match (an evaluation setting)


def log(*a):
    print(*a, flush=True)


def _official():
    sys.path.insert(0, os.path.join(C.OFFICIAL_REPO, "evaluation"))
    import possession_metrics as pm
    return pm


# ----------------------------------------------------------------------------- held-out model

def _items(m, touch=True):
    out = []
    for seed, kind, arg, keep in train_model.VARIANTS:
        d = train_model.variant(m, seed, kind, arg, keep)
        X, names = M.matrix(d, C.PARAMS, m["pitch"], m["halves"], touch=touch)
        out.append({"lab": m["labels"], "X": X, "names": names})
    return out


def fit_heldout(train_games):
    """v4 recipe (touch features, two layers) fitted on `train_games` only. Layer 2 is fitted on
    out-of-sample layer-1 probabilities: each training match predicted by a layer 1 fitted on the
    other training matches."""
    path = os.path.join(OUT, "model_games_{}.pkl".format("_".join(map(str, train_games))))
    if os.path.isfile(path):
        with open(path, "rb") as fh:
            return pickle.load(fh)
    t0 = time.time()
    items = {g: _items(metrica.load(g)) for g in train_games}
    log("  training features ready ({:.0f} s)".format(time.time() - t0))
    s1 = train_model.fit_stage1([it for g in train_games for it in items[g]])
    oof_items, probs = [], []
    for g in train_games:
        others = [o for o in train_games if o != g]
        oof = train_model.fit_stage1([it for o in others for it in items[o]])
        for it in items[g]:
            oof_items.append(it)
            probs.append(M.stage1(oof, it["X"], it["names"]))
    bundle = {"stage1": s1, "stage2": train_model.fit_stage2(oof_items, probs),
              "names": items[train_games[0]][0]["names"]}
    os.makedirs(OUT, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    log("  model fitted on games {} ({:.0f} s)".format(train_games, time.time() - t0))
    return bundle


# ----------------------------------------------------------------------------- scene 1: replay

def nearest_smooth(atd, halves, window=50):
    """The official baseline: nearest player to the ball + majority vote over +-window frames."""
    d = np.linalg.norm(atd["players"] - atd["ball"][:, None, :], axis=2)
    d = np.where(np.isnan(d), np.inf, d)
    code = np.where(np.isfinite(d.min(axis=1)), atd["team"][np.argmin(d, axis=1)], 0)
    n = len(code)
    hc = np.concatenate(([0], np.cumsum(code > 0)))
    ac = np.concatenate(([0], np.cumsum(code < 0)))
    i = np.arange(n)
    lo, hi = np.maximum(0, i - window), np.minimum(n, i + window + 1)
    h, a = hc[hi] - hc[lo], ac[hi] - ac[lo]
    lab = np.where(h + a == 0, "D", np.where(h >= a, "H", "A"))
    out = np.full(n, "X", dtype="<U1")
    for s, e in halves:
        out[s:e + 1] = lab[s:e + 1]
    return out


def spans(labels, offset=0):
    """Runs of equal labels as [start, end_exclusive, label], frames relative to `offset`."""
    labels = np.asarray(labels)
    ch = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    st = np.concatenate(([0], ch))
    en = np.concatenate((ch, [len(labels)]))
    return [[int(s - offset), int(e - offset), str(labels[s])] for s, e in zip(st, en)]


def running_metrics(pm, truth, preds, a, b):
    """Official spellF1 / macroF1 on [a, t) for t every CHECK_EVERY_S seconds."""
    out = {k: [] for k in preds}
    out["t"] = []
    step = CHECK_EVERY_S * FPS
    for t in range(a + step, b + 1, step):
        gt = np.full(len(truth), "X", dtype="<U1")
        gt[a:t] = truth[a:t]
        out["t"].append(int(t - a))
        for name, pred in preds.items():
            r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
            out[name].append([round(r["spell_f1_mean"], 4), round(r["macro_f1"], 4),
                              int(r["n_pred_spells"]), int(r["n_gt_spells"])])
    return out


def replay(game, start_min, minutes, train_games):
    pm = _official()
    m = metrica.load(game)
    # --start counts match minutes from the kick-off: past the first half it moves to the second
    start = int(round(start_min * 60 * FPS))
    a = None
    for h0, h1 in m["halves"]:
        if start <= h1 - h0:
            a, end = h0 + start, h1
            break
        start -= h1 - h0 + 1
    if a is None:
        sys.exit("--start {} is beyond the end of the match".format(start_min))
    b = min(a + int(round(minutes * 60 * FPS)), end + 1)
    log("replay: game {}, frames {}..{} ({:.1f} min)".format(game, a, b, (b - a) / FPS / 60))

    bundle = fit_heldout(train_games)
    d = degrade.degrade(m, SCENARIO, SEED)
    X, names = M.matrix(d, C.PARAMS, m["pitch"], m["halves"], touch=True)
    P = M.predict_proba(X, names, bundle)
    costs = C.MODELS["v4c"]["costs"]
    team_llr, dead_logit = M.to_evidence(P)
    ours = hmm.decode(team_llr, dead_logit, dict(C.PARAMS, **costs), m["halves"])
    base = nearest_smooth(d, m["halves"])
    truth = m["labels"]
    t = touch_features.detect(d)

    rm = running_metrics(pm, truth, {"ours": ours, "baseline": base}, a, b)
    sl = slice(a, b, max(1, FPS // PLAYER_FPS))
    pl = d["players"][sl]
    ball = d["ball"][sl]
    keep = np.flatnonzero(np.any(~np.isnan(d["players"][a:b, :, 0]), axis=0))

    def r1(x):
        return [None if not np.isfinite(v) else round(float(v), 1) for v in x]

    in_win = (t["frame"] >= a) & (t["frame"] < b)
    return {
        "source": "Metrica Sports open sample data, game {} (anonymised). "
                  "github.com/metrica-sports/sample-data".format(game),
        "fps": FPS, "step": max(1, FPS // PLAYER_FPS), "n": int(b - a),
        "pitch": list(m["pitch"]),
        "team": [int(x) for x in d["team"][keep]],
        "players": [[r1(pl[:, k, 0]), r1(pl[:, k, 1])] for k in keep],
        "ball": [r1(ball[:, 0]), r1(ball[:, 1])],
        "proba": [[round(float(x), 3) for x in P[a:b:FPS // 5, j]] for j in range(3)],
        "proba_step": FPS // 5,
        "truth": spans(truth[a:b]), "ours": spans(ours[a:b]), "baseline": spans(base[a:b]),
        "touches": [[int(f - a), int(s)] for f, s in zip(t["frame"][in_win], t["team"][in_win])],
        "metrics": rm,
        "model": "v4 recipe fitted on Metrica games {} only (this match unseen); decoder "
                 "switch_cost {:g}, dead_cost {:g}".format(train_games, costs["switch_cost"],
                                                          costs["dead_cost"]),
        "degradation": "bench/degrade.py, scenario {}, seed {}".format(SCENARIO, SEED),
    }


# ----------------------------------------------------------------------------- scene 2: labels only

def challenge(csv_path):
    """Public 5-minute window of the challenge: labels and official scores only."""
    if not os.path.isfile(csv_path):
        log("challenge scene skipped: {} not found (run python -m src.make_submission)"
            .format(csv_path))
        return None
    import pandas as pd
    pm = _official()
    ev = os.path.join(C.OFFICIAL_REPO, "evaluation")
    gt = pd.read_csv(os.path.join(ev, "gt", "BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
    a, b = 11185, 18685
    rows = []
    files = [("Our model (v4)", csv_path)] + [
        (n.replace(".csv", ""), os.path.join(ev, "results", n))
        for n in ("baseline_smooth125.csv", "baseline_smooth50.csv", "baseline_smooth25.csv",
                  "baseline_raw.csv")]
    tracks = [{"name": "Ground truth (public)", "spans": spans(gt[a:b])}]
    for name, path in files:
        if not os.path.isfile(path):
            continue
        pred = pd.read_csv(path)["label"].astype(str).str.strip().str.upper().to_numpy()
        r = pm.evaluate(pm.label_codes(gt), pm.label_codes(pred), bootstrap_n=0)
        rows.append({"name": name, "spell": round(r["spell_f1_mean"], 3),
                     "macro": round(r["macro_f1"], 3), "poss": int(r["n_pred_spells"]),
                     "dur": round(float(r["mean_spell_duration_s"]), 1)})
        if name.startswith("Our") or name == "baseline_smooth50":
            tracks.append({"name": name, "spans": spans(pred[a:b])})
    return {"n": b - a, "rows": rows, "tracks": tracks,
            "note": "Public window only (first 5 minutes, 17 possessions): a wiring check, not "
                    "the ranking. Labels only: no tracking or video of the match is shown."}


# ----------------------------------------------------------------------------- page

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--game", type=int, default=2)
    ap.add_argument("--start", type=float, default=0.0,
                    help="match minute to start at, counted from the kick-off (45 = second half)")
    ap.add_argument("--minutes", type=float, default=10.0)
    ap.add_argument("--csv", default=os.path.join(C.OUT_DIR, C.TEAM_NAME + ".csv"))
    ap.add_argument("--out", default=os.path.join(OUT, "possession_demo.html"))
    args = ap.parse_args(argv)
    if args.game not in metrica.available():
        sys.exit("Metrica game {} missing (python -m bench.metrica --download).".format(args.game))
    train = [g for g in metrica.available() if g != args.game]
    data = {"team_name": "404 Possession Not Found",
            "replay": replay(args.game, args.start, args.minutes, train),
            "challenge": challenge(args.csv)}
    with open(TEMPLATE, encoding="utf-8") as fh:
        page = fh.read().replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":")))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(page)
    log("written {} ({:.1f} MB)".format(os.path.relpath(args.out, C.ROOT),
                                        os.path.getsize(args.out) / 1e6))


if __name__ == "__main__":
    main()
