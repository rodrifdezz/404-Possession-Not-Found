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
