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

## 2026-11-13 · JAN · external bench + temperature (branch JAN_banco_externo)
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
- To do: run `python -m bench.run`, copy the chosen parameters into `config.py` and log here the
  effect on the full match.
