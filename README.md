# 404 Possession Not Found

Team submission for **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports · FC Barcelona Innovation Hub · Mobile World Capital Barcelona).

For every frame of the tactical video we output `H` (home in possession), `A` (away in possession),
`D` (dead ball) or `X` (no answer), from the provided automatic tracking and Smart Tagging.

## Method

| step | file | what it does |
|---|---|---|
| 1. ball cleaning | `src/ball.py` | linear interpolation of short ball gaps, ball speed |
| 2. dead ball | `src/dead.py` | per-frame log-odds of a stoppage: ball off the pitch, players walking, ball still, long ball loss, Smart Tagging set pieces |
| 3. team in control | `src/team.py` | per-frame log-ratio home vs away: nearest-player distance advantage, damped while the ball is in flight, weak Smart Tagging prior |
| 4. temporal model | `src/hmm.py` | 3-state (H/A/D) Viterbi with switching costs, decoded per half; emissions are normalised probabilities, p(H) = p(alive)·p(H\|alive), tempered so that consecutive frames are not counted as independent evidence |

Parameters live in `src/config.py`. They are set from physical reasoning or chosen on an external
bench of other matches (below), never tuned on the public ground truth. No hand annotation of the provided match is used anywhere, and no rule or constant
is tied to specific frames of the match.

## Reproduce

Requires Python 3.9+, the match data bundle, and a clone of the official repository next to this one:

```
parent/
├── more-than-a-hack-2027/        official repo (scorer, validator)
│   └── data/tactical/...         match data bundle
└── 404-Possession-Not-Found/     this repo
```

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m src.make_submission        # -> outputs/404-possession-not-found.csv (validated)
python -m src.evaluate               # official scorer, public 5-minute window
python -m unittest discover -s tests -v
```

`make_submission` exits with an error if the official validator rejects the CSV.

Different locations: `MTAH_REPO=/path/to/official/repo` and/or `MTAH_DATA_DIR=/path/to/data`.

## External bench (how decoder parameters are chosen)

The emission temperature and the switching costs are chosen on **other matches only**: the seven
open IDSSE / DFL Bundesliga games (Bassek et al., *Scientific Data* 12, 195, 2025, CC BY 4.0),
whose per-frame `BallStatus` / `BallPossession` flags are the same kind of tracking flags the
challenge ground truth is derived from. Each game is degraded to resemble the provided automatic
tracking, using only the figures published in DATA.md (`bench/degrade.py`), run through the same
pipeline as the submission (`src/pipeline.py`) and scored with the official scoring code,
leaving one match out at a time. Results: `docs/BENCH.md`.

```bash
python -m bench.download     # 7 matches from figshare into external/idsse/ (~2.6 GB, md5-checked)
python -m bench.run          # ~20 min -> docs/BENCH.md
```

The bench needs only the Python standard library on top of `requirements.txt`.

## Synthetic data

`tools/make_fake_data.py` writes a simulated match in the same file formats plus its ground truth,
so the pipeline can be developed and discussed without exposing the licensed data.

```bash
python tools/make_fake_data.py
MTAH_DATA_DIR=fake_data python -m src.make_submission --name fake_v0
python -m src.evaluate --gt fake_data/fake_gt.csv
```

## Data

The match data is licensed for the competition only and is **not** in this repository.
