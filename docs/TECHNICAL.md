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
   features of the automatic tracking, plus 18 features from an explicit ball-touch detector, into
   the probability that a frame is `H` (home in possession), `A` (away in possession) or `D`
   (dead ball).
2. **Decoding.** A three-state Viterbi decoder with a cost for every change of state turns those
   probabilities into whole possessions, one half at a time.

The model (v4) is fitted **only on public matches of other teams**: 3 Metrica Sports sample
matches and 7 IDSSE / DFL Bundesliga matches. Their clean tracking is damaged on purpose until it
looks like the automatic tracking of the challenge. Nothing in the pipeline is
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
| touches | `src/touch_features.py` | touch detector on the raw ball → 18 features: time since / until a touch, team of the last / next touch, balance of touches per team over 2 s and 5 s, change of team between touches, ball visibility around the frame |
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

**Touch detector.** A touch is a change of the ball's velocity of at least 4 m/s (about 3 times
the tracking noise), measured over 4 frames on each side of the raw, lightly smoothed ball. It
also needs a player within 2.5 m, and it goes to the team of the nearest player. Touches closer
than 8 frames are merged. A touch is only detected where every frame of its window is observed,
so the edge of a ball gap, where an interpolation would show a fake kink, can never produce one.
The thresholds come from physics and the tracking noise and were fixed before any experiment.
Measured on the external matches only, the team of a touch agrees with the team in possession
0.5 s later 95–97 % of the time on clean tracking and 90 % on degraded tracking. Without the
ball the detector sees nothing: the touch features then only say how long ago the last touch
was.

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
| Metrica Sports open sample data (`github.com/metrica-sports/sample-data`, published 2020–21, anonymised) | fitting the model; validating it with whole matches held out | 3 (283 graded minutes, 1,034 possessions) | per-frame H / A / D derived **by code** from the event files (7,300 events: passes, recoveries, balls out, fouls, set pieces…) | events annotated by Metrica Sports; conversion to per-frame labels by us (`bench/metrica.py`) |
| IDSSE / DFL (Bassek et al., *Scientific Data* 12, 195, 2025, CC BY 4.0) | fitting the model (v4); validating it with whole matches held out; decoder costs of the rule-based system | 7 Bundesliga matches | the per-frame `BallStatus` / `BallPossession` flags of the official DFL tracking | DFL / Sportec Solutions (shipped with the data) |

We labelled nothing by hand, on any match. The external data is downloaded by our scripts from
its original sources and is not redistributed in the repository.

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

**Submitted system (v4), on matches it has never seen.** The 10 external matches are split into
5 folds of 2 whole matches. Each fold is scored by models fitted on the other 8 matches, in both
layers. Every candidate sees the same degraded inputs, so the comparisons are paired
(`docs/BENCH_V4.md`, `docs/BENCH_V4_FIX.md`).

| | v3 (previous) | **v4 (submitted)** |
|---|---|---|
| spellF1, mean of 10 held-out matches | 0.608 | **0.667** (better on 10 / 10) |
| Metrica (3) / DFL (7) | 0.695 / 0.571 | 0.736 / 0.638 |
| macroF1 | 0.888 | 0.911 |
| possessions under 5 s recovered | 27.8 % | 36.2 % |
| false direct H↔A changes per 90 min | 33.3 | 36.1 (+8.6 %) |

v3 is the stored v3 model on the DFL matches. On a Metrica match it is the same recipe fitted on
the two other Metrica matches.

**How v4 was chosen.** The two improvements were tested as ablations on the same folds, and an
adoption rule was fixed before seeing any result. A candidate had to:

* win on average, overall and for each provider;
* win on at least 70 % of the matches;
* lose no more than 0.03 on any match;
* raise false H↔A changes by at most 20 %.

Training on all 10 matches gave +0.040. The touch features added +0.019 on top of that, winning
on all 10 matches. Possession-change features in layer 2 added nothing (+0.001), so they were
dropped.

With the v3 decoder costs, the touch model raised false changes by 23 % and was rejected. Three
quarters of those false changes are real changes placed 2–10 s away from the true one. The rest
are reactions to touches without control.

The single correction tested was the cost of a direct H↔A change in the decoder. It was chosen
inside each fold from the 8 training matches only, never from the held-out pair. The five folds
chose 9, 9, 8, 9 and 8. Applied that way, the model passes all four conditions. The submitted
costs are the median of the five choices: **`switch_cost = 9`, `dead_cost = 48`** (nats).

**History.** v3, fitted on the 3 Metrica matches only, scored 0.689 leave-one-match-out on them
and 0.571 on the 7 DFL matches it never saw (`docs/BENCH_MODEL_CV.md`, `docs/BENCH_MODEL.md`). Our
hand-written rules (v2) scored 0.362 and 0.296; that plateau, close to the baseline, is why we
moved to a fitted model. The official baseline (nearest player plus a ±2 s majority vote) scores
0.243 on the full challenge match (EVALUATION.md) and 0.238 on our degraded external matches. All
these numbers describe other matches. None of them is a score on the challenge match, either its
public window or the private result; that score will differ.

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
  remove it. The touch detector depends on the ball being tracked, and it is the part most
  exposed to that gap.
* **Limits of the evaluation.** Ten external matches are few. The same five folds were used to
  compare the four v4 candidates, so the 0.667 of the candidate we kept is slightly optimistic.
  The decoder cost was chosen inside each fold. That inner selection used an approximation: the
  inner layer 2 is fitted on layer-1 probabilities of models that had seen the match being
  predicted. This can only make the chosen cost worse, never the held-out figures better.
* **More false changes on Metrica-like tracking.** Overall, false direct H↔A changes rise by
  8.6 % over v3. On the three Metrica matches they rise by **27 %** (28.6 → 36.3 per 90 min),
  against 2 % on DFL; spellF1 still improves on each of those matches. Most of these false
  changes are real changes placed a few seconds away, or reactions to touches without control.
* **Event-derived labels.** The Metrica labels time a change of possession at the annotated
  event, which can differ by a few frames from a tracking-derived ground truth.
* **Short possessions** (under 2 s) are still recovered far less often than long ones (19 %),
  mostly because they happen while the ball is occluded, where the touch detector is blind.
* **Smart Tagging is unused** because we could not validate it; with labelled matches that include
  it, set-piece tags would probably sharpen the restarts.

## 8. Reproduction

Python 3.9–3.12. Clone the official repository next to ours and unpack the data bundle in its
`data/` folder (or set `MTAH_REPO` / `MTAH_DATA_DIR`).

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.make_submission     # -> outputs/404-possession-not-found.csv (validated, md5 printed)
python -m unittest discover -s tests -v
```

Expected md5: `ad30bb9305e408ce2cde485fea822e17` (v4). The fitted models are in `models/`
(`possession_v4c.pkl`, submitted; `possession_hgb.pkl`, v3), so nothing is trained and no
external match is needed to generate the file. `python -m src.make_submission --model v3 --name
v3` regenerates the previous v3 file. Refitting and re-running the validation are optional and
need only public data:

```bash
python -m bench.metrica --download     # Metrica sample matches
python -m bench.download               # IDSSE / DFL matches
python -m bench.v4 --features          # degraded copies and their features (local cache)
python -m bench.v4 --folds 0 1 2 3 4 --touch-diag --report   # held-out comparison (section 5)
python -m bench.v4_final               # refit the submitted model on the 10 matches
```
