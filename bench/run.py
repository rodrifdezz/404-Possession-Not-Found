"""External bench: choose the decoder parameters on matches that are NOT the challenge match.

    python -m bench.run                  # full grid, 7 matches x 2 scenarios x 2 seeds (~20 min)
    python -m bench.run --quick          # quick check (1 seed, small grid)
    python -m bench.run --report-only    # rewrite docs/BENCH.md from the saved raw results
    python -m bench.run --no-cache       # do not read or write external/idsse/cache/
    python -m bench.run --evidence model # score the learned evidence (src/model.py) against the
                                         # rules, on matches it was NOT fitted on -> BENCH_MODEL.md
    python -m bench.run --data metrica   # the 3 Metrica sample matches instead of IDSSE/DFL

For every external match: it is degraded (bench/degrade.py), run through the SAME pipeline as the
submission (src/pipeline.py, without Smart Tagging because DFL has none) and scored with the
official challenge code (evaluation/possession_metrics.py) against both readings of the truth
(bench/targets.py).

Validation: leave one match out. For each match the parameters are chosen on the other six and
scored on the one left out, so the estimate is not fitted to the match it measures. The final
parameters are chosen on all seven.

Outputs: docs/BENCH.md (summary for the technical document) and outputs/bench/ (raw, ignored).
"""

import argparse
import itertools
import json
import os
import pickle
import sys
import time

import numpy as np

from src import config as C
from src import hmm, pipeline

from . import degrade, idsse, metrica, targets

sys.path.insert(0, os.path.join(C.OFFICIAL_REPO, "evaluation"))
import possession_metrics as pm  # noqa: E402  (official scorer, MIT)

GRID = {
    "switch_cost": [4, 6, 8, 12, 16, 24],
    "dead_cost": [6, 12, 24, 48, 96, 192, 384],
}
QUICK_GRID = {"switch_cost": [8, 16], "dead_cost": [6, 48]}
V1 = {"switch_cost": 8.0, "dead_cost": 6.0}
SEEDS = [0, 1]


def configs(grid):
    keys = list(grid)
    return [dict(zip(keys, v)) for v in itertools.product(*(grid[k] for k in keys))]


def decode_grid(team_llr, dead_logit, halves, cfgs):
    """(K, N) labels for K configurations, decoded together in one Viterbi pass per half."""
    em = hmm.emissions(team_llr, dead_logit)
    trans = np.stack([hmm.transition_matrix(c) for c in cfgs])
    out = np.full((len(cfgs), len(team_llr)), "X", dtype="<U1")
    names = np.array(["H", "A", "D"])
    for a, b in halves:
        seg = np.broadcast_to(em[a:b + 1], (len(cfgs),) + em[a:b + 1].shape)
        out[:, a:b + 1] = names[hmm.viterbi_batch(seg, trans)]
    return out


def score(gt_labels, pred_labels):
    gt, pred = pm.label_codes(gt_labels), pm.label_codes(pred_labels)
    runs = pm.valid_runs(gt)
    filled = pm.fill_abstentions(pred, runs)
    spells = pm.spell_scores(gt, filled, runs)
    return {"spell_f1": spells["mean_f1"],
            "macro_f1": pm.macro_f1(pm.scored_confusion(pm.confusion(gt, pred))),
            "n_pred": spells["n_pred"], "n_gt": spells["n_gt"]}


def run_match(m, cfgs, seeds, log=print, source="rules", reference=False, evidence=None):
    """dict[(scenario, seed, target)] -> list of scores, one per configuration.

    source     which evidence is scored ("rules" or "model").
    reference  also score the rules with their own costs (config.PARAMS) on the same degraded
               data, and return (scores, reference) with reference[(scenario, seed, target)].
    evidence   optional function(degraded_match, m) -> (team_llr, dead_logit) replacing `source`
               (bench/train_model.py uses it to score a model that is not the stored one).
    """
    truth = targets.all_targets(m["labels"])
    out, ref = {}, {}
    for scenario, seed in itertools.product(degrade.SCENARIOS, seeds):
        d = degrade.degrade(m, scenario, seed)
        if evidence is not None:
            team_llr, dead_logit = evidence(d, m)
        else:
            team_llr, dead_logit = pipeline.evidence(d, [], None, m["pitch"], m["halves"], source)
        preds = decode_grid(team_llr, dead_logit, m["halves"], cfgs)
        for name, gt in truth.items():
            out[(scenario, seed, name)] = [score(gt, p) for p in preds]
        if reference:
            rules = pipeline.predict(d, [], None, m["pitch"], m["halves"], "rules")
            for name, gt in truth.items():
                ref[(scenario, seed, name)] = score(gt, rules)
        log("    {} seed {}: done".format(scenario, seed))
    return (out, ref) if reference else out


def objective(per_match):
    """(matches, K): mean spellF1 over scenarios, seeds and readings of the truth."""
    return np.array([np.mean([[s["spell_f1"] for s in v] for v in r.values()], axis=0)
                     for r in per_match])


def lomo(obj):
    """For each match: the configuration chosen on the others and its spellF1 on this one."""
    rows = []
    for i in range(len(obj)):
        others = np.delete(obj, i, axis=0).mean(axis=0)
        k = int(np.argmax(others))
        rows.append((k, float(obj[i, k])))
    return rows


def fmt(c):
    return "switch {switch_cost:g} · dead {dead_cost:g}".format(**c)


DATA_NOTE = {
    "idsse": "{} Bundesliga matches (IDSSE / DFL: Bassek et al., Sci Data 2025, CC BY 4.0)",
    "metrica": "{} Metrica Sports open sample matches (labels derived from their events, "
               "bench/metrica.py)",
}


def report(match_ids, cfgs, per_match, path_md, path_json, reference=None, source="rules",
           data="idsse", note="", command="python -m bench.run"):
    """Write the report. reference = per match, the rules scored with their own costs (then the
    comparison column is the rules; without it, it is the v1 costs with the same evidence)."""
    obj = objective(per_match)
    rows = lomo(obj)
    best = int(np.argmax(obj.mean(axis=0)))
    n = len(match_ids)

    def mean_of(i, k, key):
        return float(np.mean([s[k][key] for s in per_match[i].values()]))

    def ref_of(i, key):
        return float(np.mean([s[key] for s in reference[i].values()]))

    if reference is not None:
        ref_name = "rules v2"
        ref_spell = np.array([ref_of(i, "spell_f1") for i in range(n)])
    else:
        ref_name = "v1"
        v1 = next(i for i, c in enumerate(cfgs) if all(c[k] == V1[k] for k in V1))
        ref_spell = obj[:, v1]

    title = "# External bench ({}){}".format(
        "IDSSE / DFL" if data == "idsse" else "Metrica sample data",
        "" if source == "rules" else " — learned evidence")
    lines = [title, "",
             "Generated by `{}`{}. Data: {}, degraded to resemble the challenge "
             "tracking (bench/degrade.py). Scoring: the official challenge code. Nothing in this "
             "bench uses the challenge match.{}".format(
                 command, "" if source == "rules" else " (evidence: {})".format(source),
                 DATA_NOTE[data].format(n), " " + note if note else ""), "",
             "## Leave-one-match-out validation", "",
             "| match left out | spellF1 {} | config chosen on the others | spellF1 |".format(
                 ref_name),
             "|---|---|---|---|"]
    for i, mid in enumerate(match_ids):
        k, s = rows[i]
        lines.append("| {} | {:.3f} | {} | **{:.3f}** |".format(mid, ref_spell[i], fmt(cfgs[k]), s))
    lines += ["| **mean** | {:.3f} | | **{:.3f}** |".format(
        ref_spell.mean(), np.mean([s for _, s in rows])), "",
        "## Final parameters (chosen on all {} matches)".format(len(match_ids)), "",
        "`{}`".format(fmt(cfgs[best])), "",
        "| | spellF1 | macroF1 | predicted / true possessions |", "|---|---|---|---|"]
    if reference is not None:
        lines.append("| {} | {:.3f} | {:.3f} | {:.0f} / {:.0f} per match |".format(
            ref_name, ref_spell.mean(), np.mean([ref_of(i, "macro_f1") for i in range(n)]),
            np.mean([ref_of(i, "n_pred") for i in range(n)]),
            np.mean([ref_of(i, "n_gt") for i in range(n)])))
    for label, k in ((() if reference is not None else (("v1", v1),)) + (("final", best),)):
        n_pred = np.mean([mean_of(i, k, "n_pred") for i in range(n)])
        n_gt = np.mean([mean_of(i, k, "n_gt") for i in range(n)])
        macro = np.mean([mean_of(i, k, "macro_f1") for i in range(n)])
        lines.append("| {} | {:.3f} | {:.3f} | {:.0f} / {:.0f} per match |".format(
            label, obj[:, k].mean(), macro, n_pred, n_gt))
    order = np.argsort(-obj.mean(axis=0))[:8]
    lines += ["", "## Top 8 configurations (mean over matches)", "",
              "| config | spellF1 | worst match |", "|---|---|---|"]
    for k in order:
        lines.append("| {} | {:.3f} | {:.3f} |".format(fmt(cfgs[k]), obj[:, k].mean(),
                                                         obj[:, k].min()))
    os.makedirs(os.path.dirname(path_md), exist_ok=True)
    with open(path_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    os.makedirs(os.path.dirname(path_json), exist_ok=True)
    with open(path_json, "w", encoding="utf-8") as fh:
        json.dump({"matches": [str(m) for m in match_ids], "configs": cfgs,
                   "objective": obj.tolist(), "lomo": rows, "best": cfgs[best],
                   "reference": None if reference is None else ref_spell.tolist()}, fh, indent=1)
    return cfgs[best]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="parse the XML every time; write nothing to external/idsse/cache/")
    ap.add_argument("--matches", nargs="*", default=None)
    ap.add_argument("--evidence", choices=pipeline.SOURCES, default="rules",
                    help="evidence to score; 'model' also scores the rules as the reference")
    ap.add_argument("--data", choices=("idsse", "metrica"), default="idsse")
    args = ap.parse_args(argv)

    source, with_ref = args.evidence, args.evidence != "rules"
    if source == "model" and args.data == "metrica":
        sys.exit("The stored model was fitted on the Metrica matches: scoring it on them proves "
                 "nothing. Use --data idsse, or python -m bench.train_model --cv.")
    reader = idsse if args.data == "idsse" else metrica
    suffix = ("" if source == "rules" else "_" + source) + \
             ("" if args.data == "idsse" else "_" + args.data) + ("_quick" if args.quick else "")
    raw_path = os.path.join(C.OUT_DIR, "bench", "raw{}.pkl".format(suffix))
    reference = None
    if args.report_only:
        with open(raw_path, "rb") as fh:
            match_ids, cfgs, per_match, reference = pickle.load(fh)
    else:
        match_ids = args.matches or reader.available()
        if args.data == "metrica":
            match_ids = [int(m) for m in match_ids]
        if len(match_ids) < 2:
            sys.exit("Matches missing (download them with: python -m bench.{})."
                     .format("download" if args.data == "idsse" else "metrica --download"))
        cfgs = configs(QUICK_GRID if args.quick else GRID)
        seeds = SEEDS[:1] if args.quick else SEEDS
        print("{} matches, {} configurations, {} scenarios x {} seeds".format(
            len(match_ids), len(cfgs), len(degrade.SCENARIOS), len(seeds)))
        per_match = []
        for mid in match_ids:
            t = time.time()
            print("  {}: loading".format(mid))
            res = run_match(reader.load(mid, use_cache=not args.no_cache), cfgs, seeds,
                            source=source, reference=with_ref)
            per_match.append(res[0] if with_ref else res)
            if with_ref:
                reference = (reference or []) + [res[1]]
            print("  {}: {:.0f} s".format(mid, time.time() - t))
        os.makedirs(os.path.dirname(raw_path), exist_ok=True)
        with open(raw_path, "wb") as fh:
            pickle.dump((match_ids, cfgs, per_match, reference), fh)
    note = ""
    if source == "model":
        note = ("The model was fitted on other matches (Metrica sample data), so every match "
                "here is unseen by it; only the two decoder costs are chosen leave-one-match-out.")
    best = report(match_ids, cfgs, per_match,
                  os.path.join(C.ROOT, "docs", "BENCH{}.md".format(suffix.upper())),
                  os.path.join(C.OUT_DIR, "bench", "results{}.json".format(suffix)),
                  reference, source, args.data, note)
    print("best configuration:", fmt(best))


if __name__ == "__main__":
    main()
