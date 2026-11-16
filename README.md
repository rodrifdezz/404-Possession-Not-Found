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
| 3. evidence | `src/model.py`, `src/stack.py` | two stacked gradient-boosting layers → P(H), P(A), P(D) per frame |
| 4. temporal model | `src/hmm.py` | 3-state (H/A/D) Viterbi with switching costs, decoded per half |

The model is fitted on **public matches of other teams only** (Metrica Sports open sample data),
with their tracking degraded to resemble the automatic tracking of the challenge. Nothing is
fitted, tuned or selected on the challenge match: no hand annotation, no constant tied to specific
frames, and the public ground truth is used only to check that the pipeline is wired correctly.

On external matches the model has never seen, scored with the official code
([`docs/BENCH_MODEL_CV.md`](docs/BENCH_MODEL_CV.md)):

| | spellF1 |
|---|---|
| official baseline (nearest player + majority vote) | 0.238 |
| our rule-based system (v2, `EVIDENCE = "rules"`) | 0.362 |
| **submitted system (`EVIDENCE = "model"`)** | **0.689** |

The earlier rule-based system (`src/dead.py`, `src/team.py`, with Smart Tagging) is kept and can
be run with `--evidence rules`.

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

`make_submission` exits with an error if the official validator rejects the CSV. The fitted model
is in `models/`, so nothing is trained when the submission is generated.

Different locations: `MTAH_REPO=/path/to/official/repo` and/or `MTAH_DATA_DIR=/path/to/data`.

`python -m src.evaluate` runs the official scorer on the public 5-minute window (a wiring check,
not a tuning set).

## External data: fitting and validation

Everything below uses public matches of other teams and is optional (the fitted model and the
reports are in the repository).

```bash
python -m bench.metrica --download     # 3 Metrica sample matches into external/metrica/ (~250 MB)
python -m bench.train_model --cv       # leave-one-match-out validation -> docs/BENCH_MODEL_CV.md
python -m bench.train_model            # refit models/possession_hgb.pkl

python -m bench.download               # 7 IDSSE / DFL matches into external/idsse/ (~2.6 GB)
python -m bench.run                    # decoder costs of the rule-based system -> docs/BENCH.md
python -m bench.run --evidence model   # the stored model on those 7 matches -> docs/BENCH_MODEL.md
```

* **Metrica Sports sample data** (`github.com/metrica-sports/sample-data`): tracking plus
  human-annotated events; per-frame labels are derived from the events by `bench/metrica.py`.
* **IDSSE / DFL** (Bassek et al., *Scientific Data* 12, 195, 2025, CC BY 4.0): seven Bundesliga
  matches whose per-frame `BallStatus` / `BallPossession` flags are the same kind of tracking
  flags the challenge ground truth is derived from.

Each external match is degraded to resemble the provided automatic tracking, using only the
figures published in DATA.md (`bench/degrade.py`), run through the same pipeline as the
submission (`src/pipeline.py`) and scored with the official scoring code.

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
| `src/` | the pipeline (`load` → `ball` → `frame_features` → `model` + `stack` → `hmm`), `make_submission`, `evaluate`; `dead`, `team`, `features` are the rule-based evidence |
| `models/` | fitted model (`.pkl`) and its description (`.json`) |
| `bench/` | external data readers, degradation, validation and fitting |
| `tests/` | unit and integration tests on synthetic data |
| `docs/` | technical document, bench reports, experiment journal, team guide |

## Data

The match data is licensed for the competition only and is **not** in this repository
(`.gitignore` excludes data, caches, outputs and the external downloads).
