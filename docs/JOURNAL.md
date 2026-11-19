# Experiment journal

One entry per change. Date · who · what · why · effect on the full match
(possessions, mean duration, % D) · decision (kept / dropped).

## 2026-10-25 · v0
- Skeleton: ball cleaning + rule-based dead ball + team by distance + 3-state Viterbi.
- Initial parameters from physical reasoning (see `src/config.py`).
- To be logged: result on the real match (`make_submission` summary).

## 2026-11-01 · JAN · v1 (audit fixes)
- Removed `KNOWN_DEAD` (a fixed frame range before the 2nd-half kick-off): the rules forbid
  constants tied to specific frames of the match.
- Normalised `hmm.emissions`: p(D) = sigmoid(dead_logit), p(H) = p(alive)·sigmoid(team_llr),
  p(A) = p(alive)·sigmoid(-team_llr). In v0 the team evidence added up to +1.5 nats to H/A over D
  and hid stoppages with a player near the ball. H − A is still team_llr.
- `make_submission` fails if the official validator rejects the CSV.
- Thresholds, costs and architecture unchanged.
- Effect on the full match: D 32.3% → 47.4%; possessions 571 → 516, mean duration 7.5 → 6.5 s.
  Stretch before the 2nd-half kick-off: 99.9% D with KNOWN_DEAD; without it, 34% with the v0
  emissions and 63% with the new ones.
- Sanity check on the public 5 min (nothing was tuned on them):
  macro F1 0.7047 → 0.7814 (D 0.55 → 0.74); spell F1 0.2824 → 0.2588 (within the ~0.04 noise).
- Decision: kept (fixes a model error and a rule violation). Watch the 47% D on the full match:
  it looks high. (Update 2026-11-13: it is not; real matches have 41-52 % D.)

## 2026-11-13 · JAN · external bench + temperature (branch JAN_banco_externo, PR 1)
- Diagnosis: with a player < 2 m from the ball, `team_llr` gives up to 3 nats per frame, while
  switching possession and back costs 2 × 8 = 16 nats. ~6 frames (0.24 s) of an opponent near the
  ball are enough to create a fake possession. That is why the full match has 516 possessions
  (reference: ~200-250). Cause: at 25 fps the evidence of consecutive frames is not independent
  and gets counted 25 times per second.
- Change: `emission_temper` in `hmm.emissions` (log emissions / τ = one independent observation
  every τ frames). With τ = 1 the model is exactly v1.
- External bench (`bench/`): 7 IDSSE/DFL matches (CC BY 4.0), whose BallStatus / BallPossession
  flags are the same kind of flags the challenge GT is derived from (CHALLENGE.md). They are
  degraded according to the figures in DATA.md (41 % without ball, mostly in the air, in crowds
  or off the pitch; two gap-length scenarios). They are scored with the official code against two
  readings of the truth (the DFL flag as it is, and touches without control < 2 s given back to
  the opponent) and validated leaving one match out.
- Ingestion: `python -m bench.download` (figshare API, md5-checked). The 7 matches have 21-22
  players per frame, 41-52 % dead ball and 6.5-11 possessions per live minute on the raw DFL flag
  (5.2-7.9 with the 2 s control rule).
- `src/pipeline.py`: the submission and the bench run the same code. Refactor checked: the
  full-match CSV is identical to v1 (md5 977d0b9a…).
- The repo was normalised to English (code, comments, tests, docs).
- Note: `outputs/404-possession-not-found.csv` was v0 (md5 05a2ecc7…). It must be regenerated
  before submitting.

## 2026-11-13 · JAN · v2: decoder costs chosen on the external bench
- First bench run (125 configs: temperature × switch × dead). Two lessons:
  1. The temperature adds nothing: dividing the emissions by τ is exactly the same as multiplying
     both costs by τ (the two best configs, τ 1 / 8 / 48 and τ 3 / 4 / 12, are one model). The
     diagnosis above was half right: the evidence is over-counted, but the cure is just the costs.
     `emission_temper` was removed.
  2. What matters is `dead_cost`: the fragmentation came mostly from short fake stoppages (live →
     D → live) splitting possessions, not from H↔A flicker. The best `dead_cost` (48) was at the
     edge of that grid.
- Second run with a wider grid (switch 4-24 × dead 6-384, 42 configs): the optimum is interior
  (dead 48 beats 96). Chosen on all 7 matches: **switch 12, dead 48** (tied at 3 decimals with
  switch 8 · dead 48; the bench's choice was applied as is, not picked by hand).
- Leave-one-match-out spellF1: v1 0.261 → **0.294**, better on every one of the 7 held-out
  matches (+0.011 to +0.050). Predicted possessions per external match: 511 → 273 (truth: 380).
  Details: `docs/BENCH.md`.
- Effect on the full challenge match: possessions 516 → **333** (3.1/min), mean duration
  6.5 → **10.0 s**, median 6.5 s, D 47.4 % → 47.7 %. Valid; md5 98fdf1aa… identical on two runs.
- Not looked at: the public 5 minutes (not used to choose or confirm anything).
- Decision: kept.
- Limit: the absolute bench level is low (~0.29). Tuning the decoder is done; the remaining margin
  is in the evidence (`dead.py`, `team.py`), especially dead-ball detection. Any new rule must be
  validated on the bench before being kept.

## 2026-11-16 · v3: learned evidence (gradient boosting) replaces the hand-weighted rules
- Why: v2 plateaued at 0.29 on the IDSSE bench, close to the official baseline. The decoder was
  tuned; the limit was the evidence (hand-set weights in `dead.py` / `team.py`).
- What: `src/frame_features.py` (~90 tracking features per frame, one block per team plus a global
  one, with past / future / centred windows; no player identities, no Smart Tagging),
  `src/model.py` + `src/stack.py` (two stacked HistGradientBoosting layers -> P(H, A, D)), handed
  to the same decoder as before through `model.to_evidence` (exactly the same probabilities).
  `EVIDENCE = "model"` in `src/config.py`; `"rules"` still reproduces v2.
- Training data: the 3 Metrica open sample matches (`bench/metrica.py`; labels derived by code
  from their events, touches < 0.5 s given back). Seven degradations per match
  (`bench/train_model.py`: Markov presets normal / easy / hard, the two bench scenarios, ball
  visible or not during stoppages). Nothing from the challenge match.
- Validation (`python -m bench.train_model --cv`, `docs/BENCH_MODEL_CV.md`): both layers fitted on
  two matches and scored on the third with the protocol of `bench/run.py` (2 scenarios x 2 seeds
  x 2 readings of the truth, official scorer). spellF1: official baseline 0.238, rules v2 0.362,
  **model 0.689** (0.689 / 0.715 / 0.663; better than the rules on every match by 0.26-0.37).
  macroF1 0.50 -> 0.90.
- Decoder costs for the model, chosen by that bench and applied as they came: **switch 6,
  dead 48** (flat optimum: top 8 configs within 0.004).
- Smart Tagging dropped from the submitted system: it cannot be validated on external matches,
  and with it the bench would not be scoring what we submit.
- Fixes found on the way: `make_submission` now writes `\n` line endings on every OS (on Windows
  the CSV had `\r\n`, so the md5 differed between machines) and prints the md5; dependencies fully
  pinned (scikit-learn added); `frame_features.orientation` made symmetric outside the halves
  (caught by the new team-swap test).
- Reproducibility: the full-size synthetic match gives the same md5 on Python 3.9, 3.12 and
  3.13 (Linux), twice each.
- Honest caveat: on the toy simulator (`tools/make_fake_data.py`) the rules score higher than the
  model (spellF1 0.81 vs 0.55). The toy is built on the rules' own assumptions (ball glued 0.6 m
  from its carrier, ball parked off the pitch when dead, players jittering in place), so it says
  nothing about football; it stays a plumbing check. It does show that the model is not immune to
  a change of domain, which is why the IDSSE check below matters.
- Not looked at: the public 5 minutes.
- To be logged by whoever runs it on the challenge match: the `make_submission` summary and md5
  on two machines (macOS + Windows), and `python -m bench.run --evidence model` on the 7 IDSSE
  matches (independent check on another league; the model has never seen them).
- Decision: adopted (`EVIDENCE = "model"`), subject to the two checks above.

## 2026-11-17 · JAN · independent DFL validation of v3

- Ran `python -m bench.run --evidence model` on seven external IDSSE/DFL matches, using two
  ball-gap scenarios and two degradation seeds. Full results: `docs/BENCH_MODEL.md`.
- The stored v3 model was trained only on the three Metrica sample matches; none of the DFL
  matches were used to fit its classifiers.
- Leave-one-match-out Spell F1: rules v2 **0.296** → model v3 **0.571**. The model outperformed
  the rules on all seven matches.
- Macro F1: **0.496** → **0.883**.
- Decoder costs selected by the external benchmark: `switch_cost = 6`, `dead_cost = 48`,
  matching the current v3 configuration. No parameter changes needed.
- Full challenge-match CSV validated successfully: 199,319 frames, 296 possessions,
  10.9 s mean duration, 49.4% D on graded frames.
- Local reproducibility verified across repeated Windows runs.
  MD5: `d5484211b83141ac50ae7d101de57fae`.
- Cross-machine MD5 verification remains pending.
- Decision: retain v3 without changing its model or parameters.

## 2026-11-17 · JAN · v4: touch detector, 10 external matches (branch JAN_v4_ablaciones)

- Ablation bench `bench/v4.py`, over 10 external matches (3 Metrica, 7 DFL). The matches are
  split into 5 folds of 2 whole matches; each fold is scored by models fitted on the other 8.
  Every candidate gets the same degradations and seeds, so the results are paired. An adoption
  rule was fixed in `bench/v4_report.py` before any result. Full results: `docs/BENCH_V4.md`.
  - V4-A, the v3 recipe on the 10 matches: +0.040 spellF1 over v3.
  - V4-B, adding possession-change features to layer 2: +0.001 over V4-A. Rejected.
  - V4-C, adding the touch features of `src/touch_features.py` to layer 1: +0.019 over V4-A,
    better on 10 / 10 matches.
  - V4-D, both: no better than V4-C.
- Every candidate failed the false-switch condition: false direct H↔A changes may rise by at
  most 20 % over v3, and V4-C raised them by 23 %.
- Diagnosis (`bench/v4_fix.py`): about 3/4 of the false changes are real changes placed 2-10 s
  away. The increase comes in possessions under 2 s, with the ball visible and a touch nearby,
  i.e. touches without control. Most of the increase comes from training on 10 matches, not from
  the touch features.
- One correction tested: the decoder's `switch_cost`. It was chosen by nested validation: for each
  outer fold, only out-of-sample predictions on its 8 training matches were used. The choices
  were 9, 9, 8, 9 and 8. Applied per fold, V4-C passes the unchanged rule against v3:
  - spellF1 0.608 → 0.667, 10 / 10 wins, worst match +0.033;
  - false changes +8.6 %;
  - short possessions recovered 27.8 % → 36.2 %.
- Limitations (`docs/BENCH_V4_FIX.md`, `docs/TECHNICAL.md` §7):
  - false changes +27 % on the Metrica matches, against +2 % on DFL;
  - inner-stacking approximation of the nested choice;
  - the same folds were used to compare the four candidates, so 0.667 is slightly optimistic;
  - all figures are on external matches: none is a score on the challenge match (public window
    or private result).
- Final model: V4-C fitted on all 10 matches (`python -m bench.v4_final`, 34 min). It is
  `models/possession_v4c.pkl` (md5 `4ec3a3c1c401564439cfc7c545b02c5d`, 5.1 MB, 107 features),
  with decoder costs `switch_cost = 9` (median of the inner choices) and `dead_cost = 48`.
- Integration:
  - `config.MODEL_VERSION = "v4c"`, with `config.MODELS` holding the file and the costs of each
    model;
  - `python -m src.make_submission` generates v4;
  - `--model v3` regenerates v3, still md5 `d5484211b83141ac50ae7d101de57fae`;
  - inference needs only `src/`, the committed model and the official data; no external match
    and no `bench/` module.
- Challenge match, checked for gross errors only (nothing was tuned on it, and the public
  5 minutes were not looked at):
  - `Valid.`, 199,319 rows;
  - 277 possessions (v3: 296), mean 11.6 s (v3: 10.9 s);
  - H / A / D 31.6 / 19.0 / 49.4 %, as with v3;
  - 142 direct H↔A changes (v3: 159);
  - 5.3 % of the graded frames get a different label from v3.
- MD5 `ad30bb9305e408ce2cde485fea822e17`, identical on two Windows runs. Cross-machine check
  pending.
- Decision: v4 prepared as the default. Submitting it instead of v3 is pending the team's
  approval.

## 2026-11-17 · submission

- Submitted v4, generated by `python -m src.make_submission` with the default model.
- md5 `ad30bb9305e408ce2cde485fea822e17`, identical on Windows and on macOS. This closes the
  cross-machine check and the v3 / v4 decision left open above.
- `main` is left as submitted until the results are published.

## 2026-11-19 · demo

- `bench/demo.py` and `bench/demo_template.html`: a web page that replays a Metrica sample match
  with the model fitted on the other two Metrica matches, scored live with the official code, plus
  the labels and official scores of the public window of the challenge. See `docs/DEMO.md`.
- Nothing in the submission (`src/`, `models/`) changes.