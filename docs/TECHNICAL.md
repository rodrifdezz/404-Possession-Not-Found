---
title: "404 Possession Not Found — Technical document"
subtitle: "More than a Hack 2027 · Phase 1 · Ball possession"
geometry: margin=2.2cm
fontsize: 10.5pt
colorlinks: true
---

Repository: <https://github.com/rodrifdezz/404-Possession-Not-Found> ·
Command that regenerates the file: `python -m src.make_submission`

## 1. Approach

We treat possession as **sequence labelling with a strong temporal prior**, in two parts:

1. **Per-frame evidence.** A gradient-boosting model, in two stacked layers, turns about 90
   features of the automatic tracking into the probability that a frame is `H` (home in
   possession), `A` (away in possession) or `D` (dead ball).
2. **Decoding.** A three-state Viterbi decoder with a cost for every change of state turns those
   probabilities into whole possessions, one half at a time.

The model is fitted **only on public matches of other teams**, whose clean tracking we damage on
purpose until it looks like the automatic tracking of the challenge. Nothing in the pipeline is
fitted, tuned or selected on the challenge match: no hand label, no correction, no constant tied
to what happens in it, and the public five minutes were used only to check that the pipeline is
wired correctly (frames aligned, file well-formed).

### Why this and not something else

* **The ranking metric counts whole possessions** (spellF1, temporal-IoU matching). Per-frame
  decisions flicker, and each flicker invents possessions. The decoder is what converts evidence
  into few, clean intervals and puts each change where the evidence is strongest. We do not
  abstain inside the graded halves: the decoder already holds the previous state through
  occlusions, which is what the scorer's forward-fill of `X` would do.
* **The ball is missing on 41 % of the graded frames.** A method that only looks at "who is
  nearest to the ball" is blind there. Most of our features describe the *collective* behaviour of
  the two teams, which is visible without the ball: depth and width of each block, how many
  players are behind the ball, where each team is moving, pressure around the last known ball
  position, how long the ball has been unseen and how long until it is seen again.
* **We started with hand-written rules and measured their ceiling.** Our first system (v0–v2,
  still in the repository as `EVIDENCE = "rules"`) scored distance-to-ball and stoppage cues with
  hand-set weights. On external matches it stayed close to the official baseline however we tuned
  the decoder (section 5), because the weights of a dozen interacting cues cannot be set by
  reasoning alone. Learning those weights from labelled matches of other teams more than doubled
  the score on matches the model had never seen.
* **We cannot tune on the challenge match**, so every choice is made on external matches with the
  official scoring code, always on matches left out of the fitting.

## 2. Pipeline

| step | module | what it does |
|---|---|---|
| read | `src/load.py` | ATD text + metadata → positions in metres; `video_frame = frame_id − 1` |
| ball cleaning | `src/ball.py` | linear interpolation of ball gaps of at most 0.6 s |
| features | `src/frame_features.py` | about 90 per frame: one block per team (each in its own attacking direction), one global block, and their past / future / centred windows up to 10 s |
| layer 1 | `src/model.py` | histogram gradient boosting → P(H), P(A), P(D) per frame |
| layer 2 | `src/stack.py` | histogram gradient boosting on the *sequence* of layer-1 probabilities (rolling mean / min / max over 0.2–6 s, values up to ±2 s away) |
| decoding | `src/hmm.py` | Viterbi over H / A / D with `switch_cost` (H↔A) and `dead_cost` (alive↔dead), each half on its own |
| output | `src/make_submission.py` | `frame,label` CSV, `X` outside the graded halves, checked by the official validator |

**Features.** No player identity is used (track slots swap in the provided data): only positions
and the team of each slot. Per team: distance of its nearest and second-nearest player to the
ball, ball speed relative to that player (a ball moving *with* a player is controlled; a pass
flying past a defender is not), players within 3 m and 8 m of the ball, centroid, depth, width,
most advanced player, players behind the ball, mean speed and mean velocity along its attacking
direction. Global: ball speed and acceleration (a touch signature), distance of the ball to the
lines, how long the ball has been still, time since / until the ball is seen, mean speed of all
players, number of tracked objects.

**Symmetry.** Home and away are interchangeable: every team feature exists once per side, the
training set is augmented by swapping the teams, and both layers average each prediction with the
prediction for the swapped match. The model cannot learn a home or away bias; a test checks that
swapping the teams swaps `H` and `A` in the output exactly.

**Determinism.** No randomness at inference. Model inputs and probabilities are rounded
(10⁻³ and 10⁻⁴) so that floating-point differences between platforms cannot change a decision, the
Viterbi recursion runs on plain Python floats, the CSV is written with fixed line endings and
every dependency is pinned. `make_submission` prints the md5 of the file.

## 3. Assets used and dropped

* **Used: tactical ATD** (players and ball) and its metadata (team of each slot), plus the graded
  frame ranges published in DATA.md (needed to write `X` outside the halves and to know each
  team's attacking direction per half).
* **Dropped: Smart Tagging.** Our rule-based system used it (set-piece tags as stoppage evidence,
  phase tags as a weak team prior). We have no labelled data of other matches that comes with
  Smart Tagging, so we could never measure whether it helps, and the public window may not be used
  for that. The submitted system therefore does not read it: what we validated is exactly what we
  submit.
* **Dropped: homography and tactical video.** Not needed by the algorithm; we did not build our own
  detection or tracking.
* **Dropped: broadcast recording.** Not synchronised with the reference timeline and a third of
  its frames have no geometry.
* **Dropped: the "known dead" stretch before the second-half kick-off.** An early version forced
  `D` on that published frame range; we removed it so that no decision depends on a constant tied
  to specific frames of the match. The model labels that stretch from the tracking like any other.

## 4. Training and validation data

None of it comes from the provided match.

| dataset | used for | matches | labels | who produced the labels |
|---|---|---|---|---|
| Metrica Sports open sample data (`github.com/metrica-sports/sample-data`, published 2020–21, anonymised) | fitting the model; validating it leave-one-match-out | 3 (283 graded minutes, 1,034 possessions) | per-frame H / A / D derived **by code** from the event files (7,300 events: passes, recoveries, balls out, fouls, set pieces…) | events annotated by Metrica Sports; conversion to per-frame labels by us (`bench/metrica.py`) |
| IDSSE / DFL (Bassek et al., *Scientific Data* 12, 195, 2025, CC BY 4.0) | choosing the decoder costs of the rule-based system; independent check of the model on another league | 7 Bundesliga matches | the per-frame `BallStatus` / `BallPossession` flags of the official DFL tracking | DFL / Sportec Solutions (shipped with the data) |

We labelled nothing by hand, on any match.

**From events to labels (Metrica).** Possession is the team of the last control event (pass, set
piece, shot, recovery, ball lost, carry). The ball is dead from the event that stops play (ball
out, foul, offside, goal, shot out) to the next restart (set piece, or goal kick), provided nobody
plays the ball in between.

**Degradation.** The external tracking is clean; the challenge tracking is not. Before fitting or
scoring, each external match is damaged using only what DATA.md states about the provided data:
the ball removed in bursts until 41 % of the graded frames have none, more often in crowds and
off the pitch (and in the air, where the data has ball height); position noise; players lost for seconds at a time (20.5 tracked
objects on average); tracks fragmented over 46 slots. Since we do not know how long the real ball
gaps are, we use several degradations (two independent simulators, short and long gaps, mild and
harsh noise) and **fit on a mixture of seven of them**, so the model cannot rely on the quirks of
one simulated tracker. In the Metrica files the ball is barely recorded during stoppages, which
would let a model learn "no ball = dead ball"; we rebuild a plausible ball position there and
remove it again in part of the stoppages, so both situations are seen.

## 5. Results on external matches

All figures are spellF1 computed with the official `possession_metrics.py`, on tracking degraded
as above, averaged over two gap-length scenarios, two random seeds and two readings of the truth
(the labels as they are, and with touches shorter than 2 s between two spells of the opponent
given back to it).

**Model, on matches it has never seen** (Metrica sample matches, leave-one-match-out: both layers
fitted on two matches, scored on the third; degradation seeds different from the training ones):

| match left out | official baseline | our rules (v2) | **submitted system** |
|---|---|---|---|
| Metrica 1 | 0.235 | 0.336 | **0.689** |
| Metrica 2 | 0.239 | 0.350 | **0.715** |
| Metrica 3 | 0.241 | 0.402 | **0.663** |
| **mean** | **0.238** | **0.362** | **0.689** |

Frame-level macroF1 goes from 0.50 (rules) to 0.90, and the predicted number of possessions is
256 per match against 317 true ones (the rules predict 304, but in the wrong places).

**Decoder costs.** `switch_cost = 6`, `dead_cost = 48` (nats), chosen on the same leave-one-out
predictions. The optimum is flat: the eight best settings are within 0.004 spellF1 of each other
(`docs/BENCH_MODEL_CV.md`), so the result does not hinge on fine tuning.

**Rule-based system, for reference** (IDSSE / DFL matches, `docs/BENCH.md`): 0.261 with the
initial costs, 0.294 after choosing them leave-one-match-out on the seven matches. That plateau,
close to the baseline, is what made us replace the hand-set weights with a fitted model.

**Adoption criterion.** The learned evidence replaces the rules only if it beats them on every
left-out external match. It does, by 0.26 to 0.37.

The official baseline (nearest player plus a ±2 s majority vote) scores 0.243 on the full
challenge match (EVALUATION.md) and 0.238 on our degraded external matches, which suggests the two
are of comparable difficulty. Even so, these numbers describe other matches: the score on the
challenge match will differ.

**Independent validation on a second provider (IDSSE / DFL).** The stored v3 classifier, trained on three Metrica sample matches, was evaluated on seven unseen Bundesliga matches with artificially degraded tracking. Mean spellF1 improved from **0.296 (rules v2)** to **0.571 (model v3)**, outperforming the rules on all seven matches. MacroF1 improved from 0.496 to 0.883. Leave-one-match-out decoder selection returned `switch_cost = 6` and `dead_cost = 48`, matching the existing configuration. No model or parameter changes were made. Full results: `docs/BENCH_MODEL.md`. These are external validation results, not a score on the challenge match.

## 6. Possession rules implemented

* **Control, not proximity.** Possession changes only when the other team controls the ball. The
  training labels encode it: a spell shorter than 0.5 s that directly follows a spell of the
  opponent (a touch, a deflection, a lost duel annotated as recovery-and-loss) goes back to the
  opponent, following CHALLENGE.md ("a simple touch of the ball without control" does not count).
* **Loose and contested balls** keep the previous owner: the labels only change at a control
  event, and the decoder needs sustained evidence to pay `switch_cost`.
* **Dead ball** runs from the stoppage to the restart. Dead-ball ownership is not modelled.
* **Restarts.** A stoppage always separates two possessions (`H D H` is two possessions), as in
  the ground truth. The new possession starts at the restart touch.
* **Short stoppage cues do not split a possession.** `dead_cost` makes the decoder ask for
  sustained evidence before it opens or closes a stoppage.
* **Occlusions.** With the ball unseen the decoder keeps the current state unless the collective
  features say otherwise; we never abstain inside a half.

## 7. Limitations and what we would do with more time

* **Domain gap.** The model learns from clean tracking that we damaged ourselves. The real
  automatic tracking may fail in ways no simulation reproduces (false ball detections, duplicated
  players, systematic errors near the stands). Mixing degradations reduces the risk; it does not
  remove it.
* **Event-derived labels.** The Metrica labels time a change of possession at the annotated
  event, which can differ by a few frames from a tracking-derived ground truth, and three matches
  are few. Fitting also on the seven DFL matches, whose flags are of the same kind as the
  challenge ground truth, is the first thing we would do next.
* **Short possessions** (under 2 s) are recovered much less often than long ones; most of them
  happen while the ball is occluded.
* **Smart Tagging is unused** because we could not validate it; with labelled matches that include
  it, set-piece tags would probably sharpen the restarts.
* **No explicit touch detector.** Snapping each change of possession to the nearest detected touch
  would improve the timing of the boundaries, which the metric rewards.

## 8. Reproduction

Python 3.9–3.12. Clone the official repository next to ours and unpack the data bundle in its
`data/` folder (or set `MTAH_REPO` / `MTAH_DATA_DIR`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.make_submission     # -> outputs/404-possession-not-found.csv (validated, md5 printed)
python -m unittest discover -s tests -v
```

The fitted model is in `models/`. Refitting and re-running the validation are optional and need
only public data:

```bash
python -m bench.metrica --download     # Metrica sample matches
python -m bench.train_model --cv       # leave-one-match-out validation (section 5)
python -m bench.train_model            # refit the stored model
python -m bench.download               # IDSSE / DFL matches
python -m bench.run --evidence model   # the stored model on the 7 DFL matches
```
