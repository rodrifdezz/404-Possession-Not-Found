# 404 Possession Not Found

Team submission for **More than a Hack 2027 — Phase 1: Ball Possession**
(Metrica Sports · FC Barcelona Innovation Hub · Mobile World Capital Barcelona).

For every frame of the tactical video we output `H` (home in possession), `A` (away in possession),
`D` (dead ball) or `X` (no answer), from the provided automatic tracking and Smart Tagging.

## Method (v0)

| step | file | what it does |
|---|---|---|
| 1. ball cleaning | `src/ball.py` | linear interpolation of short ball gaps, ball speed |
| 2. dead ball | `src/dead.py` | per-frame log-odds of a stoppage: ball off the pitch, players walking, ball still, long ball loss, Smart Tagging set pieces, known pre-kickoff stretch |
| 3. team in control | `src/team.py` | per-frame log-ratio home vs away: nearest-player distance advantage, damped while the ball is in flight, weak Smart Tagging prior |
| 4. temporal model | `src/hmm.py` | 3-state (H/A/D) Viterbi with switching costs, decoded per half |

Parameters live in `src/config.py` and are set from physical reasoning, not tuned on the public
ground truth. No hand annotation of the provided match is used anywhere.

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
```

Different locations: `MTAH_REPO=/path/to/official/repo` and/or `MTAH_DATA_DIR=/path/to/data`.

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
