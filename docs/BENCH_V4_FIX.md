# V4-C: false H<->A switches, one correction, and the v4 model

Follow-up of [BENCH_V4.md](BENCH_V4.md). There, V4-C (layer 1 with the touch features) beat V3 on
spellF1 in all 10 held-out matches but was rejected by the rule fixed beforehand: false H<->A
switches rose by 23 % (limit 20 %). This page explains why, tests a single correction chosen
without looking at the outer folds, and describes the model fitted on the 10 matches. That model
is now the default of `python -m src.make_submission` (v4); v3 stays available with `--model v3`.
The rule, the outer folds and the historical results of BENCH_V4.md are unchanged.

Code: `bench/v4_fix.py` (diagnosis, nested selection), `bench/v4_final.py` (final fit),
`src/touch_features.py` and `src/model.py` (inference), tests in `tests/test_v4.py`.

## 1. Where the extra false switches come from

Outer held-out predictions of BENCH_V4.md (10 matches x 2 gap scenarios x 2 seeds x 2 readings of
the truth). A false switch is a predicted direct H<->A change with no true change towards that team
within +-2 s (as in BENCH_V4.md). Per 90 minutes:

| | V3 | V4-A | V4-C |
|---|---|---|---|
| all | 33.3 | 40.0 | 41.0 |
| Metrica | 28.6 | 37.3 | 41.3 |
| DFL | 35.3 | 41.1 | 40.8 |

* Most of the increase comes from training on 10 matches (V4-A, +6.7), not from the touch
  features (+1.0 more). On DFL the touch features slightly reduce it; on Metrica they add 4.0.

| kind (per 90 min) | V3 | V4-C |
|---|---|---|
| true change towards that team, 2-5 s away | 18.8 | 20.6 |
| true change towards that team, 5-10 s away | 6.1 | 9.3 |
| spurious: excursion away from the true team | 6.3 | 7.6 |
| spurious: return to the true team | 1.9 | 3.1 |
| spurious: during dead ball | 0.2 | 0.3 |

* About three quarters are real changes placed outside the +-2 s tolerance (V3 75 %, V4-C 73 %).
  Of the +7.7 of V4-C, +5.0 are shifted real changes and +2.6 are truly spurious.
* By duration of the predicted possession that starts: the increase is in possessions under 2 s
  (6.6 -> 10.6); those of 10 s or more do not change (10.5 -> 11.3).
* By ball visibility at the switch: the increase happens with the ball visible (20.1 -> 25.9);
  with the ball lost it hardly changes (13.2 -> 15.1). It is not an artefact of interpolation.
* With a detected touch within 1 s: 19.0 -> 25.1; without: 14.3 -> 15.9.

Reading: V4-C reacts to real contacts that the truth does not count as a change of possession
(clearances, touches without control), producing very short possessions, most of them next to a
real change. The share of predicted H<->A changes that are false is the same for V3, V4-A and V4-C
(about 22 %). V4-C predicts more changes (179 vs 150 per 90 min) and gets about 22 more right per
90 min. The rule counts false switches in absolute terms; it was kept as it is.

## 2. The correction: switch_cost of the decoder, chosen by nested validation

* Only change: `switch_cost` of the unchanged Viterbi decoder, from {6, 7, 8, 9, 10, 12, 14}.
  It is paid only by a direct H<->A change (H -> D -> A costs 2 x 48), so it acts precisely on the
  failed measure. Features, models, `dead_cost` and the decoder code are unchanged.
* Pre-registered in `bench/v4_fix.py` before any inner or corrected outer result was seen.
  For each outer fold, the cost is chosen ONLY from out-of-sample V4-C probabilities of its 8
  training matches. The outer test pair is never used.
  * Constraint: inner false switches <= 1.2 x those of the stored V3 on the same inner DFL matches.
  * Objective: the highest inner spellF1 among the costs that meet the constraint.
* **Limitation of the inner stacking.** The inner out-of-sample predictions come from 3 groups of
  whole matches. Layer 1 of group g is fitted without g. The inner layer 2 of g is fitted on the
  layer-1 probabilities of the other groups, and those were produced by layer-1 models that did
  see g. This mild leak can only make the choice of the cost worse; it cannot bias the outer
  estimate, because the outer pair is out of every model involved.
* **Limitation of the reference.** Metrica matches are left out of the constraint, because there
  the stored V3 would be in-sample.
* The outer evaluation used, for each fold, the cost chosen by that fold: 9, 9, 8, 9, 8. The
  production cost is the median of these five choices: 9.
* In the inner validation, spellF1 changes by less than 0.002 between costs 7 and 10, so the
  choice is not a fine tuning to the threshold.
* The saved probabilities reproduce the historical V4-C exactly (checked in `--select`).

## 3. Paired outer comparison (unchanged rule of `bench/v4_report.py`)

| | V3 | V4-C (cost 6) | **V4-C, cost of its fold** |
|---|---|---|---|
| spellF1, mean of 10 matches | 0.608 | 0.667 | **0.667** |
| macroF1, mean | 0.888 | 0.911 | **0.911** |
| short possessions < 5 s recovered | 27.8 % | 37.6 % | **36.2 %** |
| very short < 2 s recovered | 12.6 % | 21.2 % | **19.4 %** |
| false H<->A per 90 min | 33.3 | 41.0 (+23 %) | **36.1 (+8.6 %)** |
| false H<->A per 90 min, Metrica | 28.6 | 41.3 | 36.3 (**+27 %**) |
| false H<->A per 90 min, DFL | 35.3 | 40.8 | 36.1 (+2 %) |
| predicted H<->A per 90 min | 149.7 | 179.0 | 169.9 |

spellF1 per held-out match:

| match | V3 | V4-C | V4-C corrected |
|---|---|---|---|
| M1 | 0.690 | 0.726 | 0.733 |
| M2 | 0.727 | 0.760 | 0.759 |
| M3 | 0.667 | 0.716 | 0.715 |
| J03WMX | 0.558 | 0.629 | 0.631 |
| J03WN1 | 0.502 | 0.598 | 0.600 |
| J03WOH | 0.596 | 0.665 | 0.668 |
| J03WOY | 0.557 | 0.608 | 0.603 |
| J03WPY | 0.591 | 0.648 | 0.644 |
| J03WQQ | 0.606 | 0.659 | 0.660 |
| J03WR9 | 0.586 | 0.658 | 0.659 |

Results of the rule:

* Against V3: mean difference +0.059 (Metrica +0.041, DFL +0.067), 10/10 wins, worst match
  +0.033, false switches +8.6 %. **All four conditions pass.**
* Against uncorrected V4-C:
  * spellF1 is the same (+0.0006), with 12 % fewer false switches;
  * short-possession recovery drops by 1.4 points; most of the gain over V3 is kept.

**Limitation to keep in mind.** The rule applies to all matches, as fixed beforehand, and it
passes. Per provider, however, Metrica false switches are still +27 % over V3 (DFL +2 %). On
Metrica-like tracking the candidate makes noticeably more false team changes than V3, while
still scoring higher spellF1 on every Metrica match.

## 4. The v4 model

### Fit

`python -m bench.v4_final` fits the V4-C recipe of BENCH_V4.md on the 3 Metrica and 7 DFL
matches:

* the 7 training degradations, 1 frame in 6, with the touch features;
* layer 2 fitted on out-of-sample layer-1 probabilities of 3 groups of whole matches, with no
  extras;
* decoder Viterbi with `switch_cost` = 9 and `dead_cost` = 48.

The fit took 34 min on a laptop. The model is `models/possession_v4c.pkl`
(md5 `4ec3a3c1c401564439cfc7c545b02c5d`, 5.1 MB, 107 features: 89 frame features and then the
18 touch features), described by `possession_v4c.json`, which lists the features, the training
matches, the hyper-parameters, the decoder costs and the scikit-learn version. It holds no match
data. The v3 model `models/possession_hgb.pkl` is unchanged.

### Inference

`python -m src.make_submission` uses `config.MODEL_VERSION = "v4c"` and the costs of
`config.MODELS["v4c"]`. It computes the same features as the bench: the frame features stored as
float32, then the touch features. It applies layer 1 and layer 2, both symmetric, and decodes
with `src/hmm.py`. Inference needs only `src/`, the committed model and the official data: no
external match, no feature cache and no `bench/` module (a test checks this in a fresh
interpreter).

### Checks

* 199,319 rows, header `frame,label`; the official validator says `Valid.`
* Generated twice: md5 `ad30bb9305e408ce2cde485fea822e17` both times. It was first generated
  with the experimental script, then twice through `make_submission`.
* `--model v3` still gives md5 `d5484211b83141ac50ae7d101de57fae`.
* 75/75 tests pass. Those of the v4 model check:
  * the versions and the decoder costs Viterbi receives (9 / 48 for v4, 6 / 48 for v3);
  * the 107 features in order: the 89 of v3, then the 18 touch features, in float32;
  * the bundle and its metadata (md5, features, costs, scikit-learn version);
  * valid labels;
  * determinism;
  * that swapping the teams swaps H and A;
  * that the probabilities are valid;
  * that v3 and v4 do not share state;
  * that inference does not import `bench/`;
  * that the following are rejected with a clear error:
    * an unknown version;
    * a missing model file;
    * incompatible features;
    * decoder costs that disagree between the model and `config`.

### Aggregate statistics on the challenge match

These are a sanity check only; nothing was tuned to them and no label of the challenge match
was used.

| | v3 (`--model v3`) | v4 (default) |
|---|---|---|
| H / A / D | 31.6 % / 19.0 % / 49.4 % | 31.6 % / 19.0 % / 49.4 % |
| possessions (per min) | 296 (2.79) | 277 (2.62) |
| mean / median duration | 10.9 s / 6.3 s | 11.6 s / 7.1 s |
| direct H<->A changes (per 90 min) | 159 (135) | 142 (121) |
| graded frames with a different label | | 5.3 % |

Both models give the same overall shares. The candidate makes slightly fewer and longer
possessions, with fewer direct team changes, as expected from the higher switch cost. Frames
outside the halves are X in both.

### Remaining limitations

* All the evidence comes from 10 external matches with simulated degradations. The challenge
  tracking may behave differently.
* The touch detector agrees with the team in possession about 90 % of the time on degraded
  tracking (BENCH_V4.md). It is less reliable when the ball is lost; ball-loss periods
  contribute little to the extra false switches (section 1).
* Metrica-like tracking: false switches +27 % over V3 (section 3).
* Inner stacking approximation of the nested selection (section 2).
* The four V4 variants of BENCH_V4.md were compared on the same outer folds, so the 0.667 of the
  one we kept is slightly optimistic.
* All figures on this page come from external matches. None is a score on the challenge match,
  either its public window or the private result.
* Which file is finally submitted (v4 or v3) is the team's decision; both are regenerated by
  the same command.
