# 404 Possession Not Found

Team submission for **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports · FC Barcelona Innovation Hub · Mobile World Capital Barcelona).

For every frame of the tactical video we output `H` (home in possession), `A` (away in possession),
`D` (dead ball) or `X` (not graded), from the provided automatic tracking.

Technical document: [`docs/TECHNICAL.md`](docs/TECHNICAL.md).

## Method

| step | file | what it does |
|---|---|---|
| 1. ball cleaning | `src/ball.py` | linear interpolation of short ball gaps |
| 2. features | `src/frame_features.py` | ~90 per frame, from the tracking only: one block per team (distance and relative speed of its nearest players to the ball, shape and movement of the team), one global block (ball speed, ball near the lines, time since / until the ball is seen), and their past / future windows |
| 3. touches | `src/touch_features.py` | explicit ball-touch detector (sharp change of the raw ball's velocity with a player within 2.5 m) → 18 features: time since / until a touch, team of the last / next touch, touch balance per team |
| 4. evidence | `src/model.py`, `src/stack.py` | two stacked gradient-boosting layers → P(H), P(A), P(D) per frame |
| 5. temporal model | `src/hmm.py` | 3-state (H/A/D) Viterbi with switching costs (switch 9, dead 48), decoded per half |

The model (v4) is fitted on **public matches of other teams only**: 3 Metrica Sports sample
matches and 7 IDSSE / DFL Bundesliga matches, with their tracking degraded to resemble the
automatic tracking of the challenge. Nothing is fitted, tuned or selected on the challenge match:
no hand annotation, no constant tied to specific frames, and the public ground truth is used only
to check that the pipeline is wired correctly.

On the 10 external matches, each scored by models fitted on the other matches, with the official
code ([`docs/BENCH_V4.md`](docs/BENCH_V4.md), [`docs/BENCH_V4_FIX.md`](docs/BENCH_V4_FIX.md)):

| | spellF1 |
|---|---|
| previous submission (v3: 3 Metrica matches, no touch features) | 0.608 |
| **submitted system (v4: `python -m src.make_submission`)** | **0.667** (better on 10 / 10 matches) |

* Decoder costs of v4: `switch_cost = 9`, `dead_cost = 48`. The switch cost was chosen inside
  each fold from its training matches only.
* Cost of the gain: false direct H↔A changes rise by 8.6 % overall, and by 27 % on the Metrica
  matches.
* The four v4 variants were compared on the same folds, so the 0.667 is slightly optimistic.
* None of these figures is a score on the challenge match (public window or private result).

The official baseline scores 0.238 on our degraded Metrica matches and our first rule-based
system 0.362 ([`docs/BENCH_MODEL_CV.md`](docs/BENCH_MODEL_CV.md)). v3 and the rule-based system
(`src/dead.py`, `src/team.py`, with Smart Tagging) are kept: `--model v3` and `--evidence rules`.

## Reproduce

Requires Python 3.9–3.12, the match data bundle, and a clone of the official repository next to
this one:

```
parent/
├── more-than-a-hack-2027/        official repo (scorer, validator)
│   └── data/tactical/...         match data bundle
└── 404-Possession-Not-Found/     this repo
```

```bash
python3 -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m src.make_submission        # -> outputs/404-possession-not-found.csv (validated, md5 printed)
python -m unittest discover -s tests -v
```

Expected md5: `ad30bb9305e408ce2cde485fea822e17`. `make_submission` exits with an error if the
official validator rejects the CSV. The fitted models are in `models/`, so nothing is trained and
no external match is needed when the submission is generated.
`python -m src.make_submission --model v3 --name v3` regenerates the previous (v3) file,
md5 `d5484211b83141ac50ae7d101de57fae`.

Different locations: `MTAH_REPO=/path/to/official/repo` and/or `MTAH_DATA_DIR=/path/to/data`.

`python -m src.evaluate` runs the official scorer on the public 5-minute window (a wiring check,
not a tuning set).

## External data: fitting and validation

Everything below uses public matches of other teams and is optional (the fitted model and the
reports are in the repository).

```bash
python -m bench.metrica --download     # 3 Metrica sample matches into external/metrica/ (~250 MB)
python -m bench.download               # 7 IDSSE / DFL matches into external/idsse/ (~2.6 GB)

# v4 (submitted): held-out comparison on the 10 matches, correction, refit (several hours)
python -m bench.v4 --features          # degraded copies + features, cached in outputs/v4/
python -m bench.v4 --folds 0 1 2 3 4 --touch-diag --report   # -> docs/BENCH_V4.md
python -m bench.v4_fix --proba --diagnose --inner 0 1 2 3 4 --select   # -> docs/BENCH_V4_FIX.md
python -m bench.v4_final               # refit models/possession_v4c.pkl

# v3 and the rule-based system
python -m bench.train_model --cv       # leave-one-match-out validation -> docs/BENCH_MODEL_CV.md
python -m bench.train_model            # refit models/possession_hgb.pkl
python -m bench.run                    # decoder costs of the rule-based system -> docs/BENCH.md
python -m bench.run --evidence model   # v3 on the 7 DFL matches -> docs/BENCH_MODEL.md
```

* **Metrica Sports sample data** (`github.com/metrica-sports/sample-data`): tracking plus
  human-annotated events; per-frame labels are derived from the events by `bench/metrica.py`.
* **IDSSE / DFL** (Bassek et al., *Scientific Data* 12, 195, 2025, CC BY 4.0): seven Bundesliga
  matches whose per-frame `BallStatus` / `BallPossession` flags are the same kind of tracking
  flags the challenge ground truth is derived from.

Each external match is degraded to resemble the provided automatic tracking, using only the
figures published in DATA.md (`bench/degrade.py`), run through the same pipeline as the
submission (`src/pipeline.py`) and scored with the official scoring code.

## Demo

`python -m bench.demo` writes `outputs/demo/possession_demo.html`, a self-contained page that
replays a public match (Metrica sample data) with the model running live and scored by the
official code, plus the public window of the challenge (labels and scores only). It is safe to
record and publish: see [`docs/DEMO.md`](docs/DEMO.md).

## Synthetic data

`tools/make_fake_data.py` writes a simulated match in the same file formats plus its ground truth,
so the pipeline can be developed and discussed without exposing the licensed data. It is a
plumbing check (file formats, alignment, determinism), not a measure of accuracy: its players do
not move like footballers.

```bash
python tools/make_fake_data.py
MTAH_DATA_DIR=fake_data python -m src.make_submission --name fake
python -m src.evaluate --gt fake_data/fake_gt.csv
```

## Repository

| path | content |
|---|---|
| `src/` | the pipeline (`load` → `ball` → `frame_features` + `touch_features` → `model` + `stack` → `hmm`), `make_submission`, `evaluate`; `dead`, `team`, `features` are the rule-based evidence |
| `models/` | fitted models (`.pkl`) and their descriptions (`.json`): `possession_v4c` (submitted), `possession_hgb` (v3) |
| `bench/` | external data readers, degradation, validation and fitting |
| `tests/` | unit and integration tests on synthetic data |
| `docs/` | technical document, bench reports, experiment journal |

## Data

The match data is licensed for the competition only and is **not** in this repository
(`.gitignore` excludes data, caches, outputs and the external downloads).
