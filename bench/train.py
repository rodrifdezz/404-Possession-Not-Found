"""Train and validate the learned evidence (v3) on the external matches only.

    python -m bench.train        # leave-one-match-out validation + final model (~35 min)

Protocol, fixed before looking at any result:
  - data: the 7 IDSSE matches, each degraded in the three scenarios of bench/degrade.py
    (seed 0), never the challenge match;
  - for each match m: fit both logistic models (src/learned.py) on the other 6, choose the decoder
    costs on the other 6, and score m. v2 (rules, costs from config) is scored on exactly the
    same degraded copies;
  - adoption rule: v3 replaces v2 only if
      (1) its mean leave-one-out spellF1 is at least v2 + 0.01,
      (2) it is better on at least 5 of the 7 matches, and
      (3) it is also better on average in the stoppage_gaps scenario alone, so that it does not
          depend on our guess of WHEN the real tracking loses the ball;
  - final model: fitted on all 7, costs chosen on all 7 -> models/evidence_v3.json.

Labels: the dead model learns DFL's BallStatus; the team model learns the "control_2s" reading
(bench/targets.py), i.e. the challenge definition that a touch without control does not change
possession. Scores average both readings of the truth, as in bench/run.py.

Outputs: docs/BENCH_V3.md and models/evidence_v3.json.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from src import config as C
from src import learned, pipeline

from . import degrade, idsse, run, targets

SEED = 0
SUBSAMPLE = 10              # training rows: one graded frame in 10 (neighbours are near copies)
GRID = {"switch_cost": [2, 4, 8, 16, 32], "dead_cost": [4, 8, 16, 32, 64, 128, 256, 512]}
MIN_GAIN, MIN_WINS, ROBUST_SCENARIO = 0.01, 5, "stoppage_gaps"
MODEL_PATH = os.path.join(C.ROOT, C.EVIDENCE_MODEL)
REPORT_PATH = os.path.join(C.ROOT, "docs", "BENCH_V3.md")


def prepare(match_ids, log=print):
    """One entry per (match, scenario): features, truths and the v2 evidence."""
    versions = []
    for mid in match_ids:
        m = idsse.load(mid, use_cache=False)
        truth = targets.all_targets(m["labels"])
        for scenario in degrade.SCENARIOS:
            t = time.time()
            d = degrade.degrade(m, scenario, SEED)
            dead_X, team_X = learned.features(d, m["pitch"], m["halves"])
            v2_team, v2_dead = pipeline.evidence(d, [], C.PARAMS, m["pitch"], m["halves"],
                                                 source="rules")
            versions.append({"match": mid, "scenario": scenario, "halves": m["halves"],
                             "dead_X": dead_X, "team_X": team_X, "truth": truth,
                             "v2": (v2_team.astype(np.float32), v2_dead.astype(np.float32))})
            log("  {} {}: features in {:.0f} s".format(mid, scenario, time.time() - t))
    return versions


def training_rows(versions):
    """Subsampled training rows of the dead model and of the team model."""
    dX, dy, tX, ty = [], [], [], []
    for v in versions:
        dfl, ctrl = v["truth"]["dfl"], v["truth"]["control_2s"]
        rows = np.flatnonzero(dfl != "X")[::SUBSAMPLE]
        dX.append(v["dead_X"][rows])
        dy.append(dfl[rows] == "D")
        alive = rows[np.isin(ctrl[rows], ["H", "A"])]
        tX.append(v["team_X"][alive])
        ty.append(ctrl[alive] == "H")
    return (np.concatenate(dX), np.concatenate(dy).astype(float),
            np.concatenate(tX), np.concatenate(ty).astype(float))


def fit(versions):
    dX, dy, tX, ty = training_rows(versions)
    return {"dead": learned.fit_logistic(dX, dy, intercept=True),
            "team": learned.fit_logistic(tX, ty, intercept=False)}


def scores(v, team_llr, dead_logit, cfgs):
    """Per configuration: spellF1 / macroF1 / predicted spells, averaged over both truths."""
    preds = run.decode_grid(team_llr, dead_logit, v["halves"], cfgs)
    out = []
    for p in preds:
        s = [run.score(gt, p) for gt in v["truth"].values()]
        out.append({k: float(np.mean([x[k] for x in s])) for k in s[0]})
    return out


def v3_scores(model, versions, cfgs):
    return [scores(v, learned.predict(model["team"], v["team_X"]),
                   learned.predict(model["dead"], v["dead_X"]), cfgs) for v in versions]


def mean_over(score_lists, key="spell_f1"):
    """(K,) mean of key over a list of per-version score lists."""
    return np.mean([[s[key] for s in sl] for sl in score_lists], axis=0)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--matches", nargs="*", default=None, help="subset, for a quick check")
    match_ids = ap.parse_args(argv).matches or idsse.available()
    if len(match_ids) < 3:
        sys.exit("Matches missing in {} (download them with: python -m bench.download)."
                 .format(idsse.IDSSE_DIR))
    cfgs = run.configs(GRID)
    v2_cfg = {"switch_cost": C.PARAMS["switch_cost"], "dead_cost": C.PARAMS["dead_cost"]}
    print("{} matches x {} scenarios, {} cost configurations".format(
        len(match_ids), len(degrade.SCENARIOS), len(cfgs)))
    versions = prepare(match_ids)

    v2 = [scores(v, *v["v2"], [v2_cfg])[0] for v in versions]

    rows = []      # one per held-out version: match, scenario, v2 and v3 scores
    for mid in match_ids:
        t = time.time()
        held = [i for i, v in enumerate(versions) if v["match"] == mid]
        train = [i for i, v in enumerate(versions) if v["match"] != mid]
        model = fit([versions[i] for i in train])
        sc = v3_scores(model, versions, cfgs)
        k = int(np.argmax(mean_over([sc[i] for i in train])))
        for i in held:
            rows.append({"match": mid, "scenario": versions[i]["scenario"], "cfg": cfgs[k],
                         "v2": v2[i]["spell_f1"], "v3": sc[i][k]["spell_f1"],
                         "v2_macro": v2[i]["macro_f1"], "v3_macro": sc[i][k]["macro_f1"]})
        print("  fold {}: v2 {:.3f}  v3 {:.3f}  ({}, {:.0f} s)".format(
            mid, np.mean([v2[i]["spell_f1"] for i in held]),
            np.mean([sc[i][k]["spell_f1"] for i in held]), run.fmt(cfgs[k]), time.time() - t))

    per_match = summarise(rows, "match", match_ids)
    per_scenario = summarise(rows, "scenario", list(degrade.SCENARIOS))
    gain = np.mean([r["v3"] - r["v2"] for r in per_match])
    wins = sum(r["v3"] > r["v2"] for r in per_match)
    robust = next(r for r in per_scenario if r["scenario"] == ROBUST_SCENARIO)
    adopted = bool(gain >= MIN_GAIN and wins >= MIN_WINS and robust["v3"] > robust["v2"])

    final = fit(versions)
    sc = v3_scores(final, versions, cfgs)
    k = int(np.argmax(mean_over(sc)))
    counts = (float(mean_over(sc, "n_pred")[k]),
              float(np.mean([s["n_pred"] for s in v2])), float(np.mean([s["n_gt"] for s in v2])))

    model = {"version": "v3", "dead_features": learned.DEAD_FEATURES,
             "team_features": learned.TEAM_FEATURES, "dead": final["dead"], "team": final["team"],
             "costs": cfgs[k],
             "trained_on": "IDSSE/DFL matches {} degraded by bench/degrade.py (scenarios {}, "
                           "seed {}), every {}th graded frame".format(
                               ", ".join(match_ids), ", ".join(degrade.SCENARIOS), SEED,
                               SUBSAMPLE)}
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    with open(MODEL_PATH, "w", encoding="utf-8") as fh:
        json.dump(model, fh, indent=1)
    write_report(per_match, per_scenario, gain, wins, adopted, final, cfgs[k], counts)
    print("mean gain {:+.3f}, better on {}/{}, {}: v2 {:.3f} v3 {:.3f} -> {}".format(
        gain, wins, len(per_match), ROBUST_SCENARIO, robust["v2"], robust["v3"],
        "ADOPT v3" if adopted else "keep v2"))
    print("final costs:", run.fmt(cfgs[k]))


def summarise(rows, key, order):
    """Mean v2 / v3 scores grouped by key ("match" or "scenario"), plus the chosen costs."""
    out = []
    for value in order:
        g = [r for r in rows if r[key] == value]
        out.append({key: value, "cfg": g[0]["cfg"],
                    **{s: float(np.mean([r[s] for r in g]))
                       for s in ("v2", "v3", "v2_macro", "v3_macro")}})
    return out


def write_report(per_match, per_scenario, gain, wins, adopted, final, cfg, counts):
    n = len(per_match)
    lines = [
        "# Learned evidence v3: external validation", "",
        "Generated by `python -m bench.train`. Data: {} IDSSE/DFL matches degraded as in "
        "`bench/degrade.py` ({} scenarios, seed {}). Scoring: the official challenge code, mean "
        "of both readings of the truth. Nothing here uses the challenge match.".format(
            n, len(per_scenario), SEED), "",
        "## Leave-one-match-out", "",
        "For each match, v3 is fitted and its costs are chosen on the other matches only. v2 is "
        "the rules of `dead.py` / `team.py` with the costs in `config.py`. Each row is the mean "
        "over the degradation scenarios.", "",
        "| match left out | spellF1 v2 | spellF1 v3 | difference | v3 costs (chosen on the others) "
        "| macroF1 v2 | macroF1 v3 |",
        "|---|---|---|---|---|---|---|"]
    for r in per_match:
        lines.append("| {} | {:.3f} | **{:.3f}** | {:+.3f} | {} | {:.3f} | {:.3f} |".format(
            r["match"], r["v2"], r["v3"], r["v3"] - r["v2"], run.fmt(r["cfg"]),
            r["v2_macro"], r["v3_macro"]))
    mean = lambda key: np.mean([r[key] for r in per_match])
    lines += [
        "| **mean** | {:.3f} | **{:.3f}** | **{:+.3f}** | | {:.3f} | {:.3f} |".format(
            mean("v2"), mean("v3"), gain, mean("v2_macro"), mean("v3_macro")), "",
        "## By degradation scenario (same held-out scores)", "",
        "| scenario | spellF1 v2 | spellF1 v3 | difference | macroF1 v2 | macroF1 v3 |",
        "|---|---|---|---|---|---|"]
    for r in per_scenario:
        lines.append("| {} | {:.3f} | **{:.3f}** | {:+.3f} | {:.3f} | {:.3f} |".format(
            r["scenario"], r["v2"], r["v3"], r["v3"] - r["v2"], r["v2_macro"], r["v3_macro"]))
    robust = next(r for r in per_scenario if r["scenario"] == ROBUST_SCENARIO)
    lines += [
        "", "Adoption rule (fixed beforehand): mean gain >= {}, better on >= {} of {} matches, and "
        "better in the `{}` scenario. Result: gain {:+.3f}, better on {}/{}, `{}` {:+.3f} -> "
        "**{}**.".format(MIN_GAIN, MIN_WINS, n, ROBUST_SCENARIO, gain, wins, n, ROBUST_SCENARIO,
                         robust["v3"] - robust["v2"],
                         "v3 adopted" if adopted else "v3 rejected, v2 kept"), "",
        "## Final model (fitted on all {} matches)".format(n), "",
        "Decoder costs chosen on all matches: `{}`. Predicted possessions per match: v3 {:.0f}, "
        "v2 {:.0f}, truth {:.0f}.".format(run.fmt(cfg), *counts), "",
        "Standardised coefficients (log-odds per standard deviation), largest first:", ""]
    for title, part, names in (("Dead ball", final["dead"], learned.DEAD_FEATURES),
                               ("Home in possession", final["team"], learned.TEAM_FEATURES)):
        coef = np.asarray(part["coef"])
        order = np.argsort(-np.abs(coef))[:10]
        lines += ["**{}**: {}".format(title, ", ".join(
            "`{}` {:+.2f}".format(names[i], coef[i]) for i in order)), ""]
    with open(REPORT_PATH, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


if __name__ == "__main__":
    main()
