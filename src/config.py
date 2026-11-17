"""Paths, match constants and algorithm parameters. Everything configurable lives here.

Paths (can be changed with environment variables, no code changes needed):
  MTAH_REPO      folder of the official Metrica repository   (default ../more-than-a-hack-2027)
  MTAH_DATA_DIR  folder containing tactical/                  (default <MTAH_REPO>/data)
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL_REPO = os.path.abspath(
    os.environ.get("MTAH_REPO") or os.path.join(ROOT, "..", "more-than-a-hack-2027"))
DATA_DIR = os.path.abspath(
    os.environ.get("MTAH_DATA_DIR") or os.path.join(OFFICIAL_REPO, "data"))

OUT_DIR = os.path.join(ROOT, "outputs")   # gitignored
CACHE_DIR = os.path.join(ROOT, "cache")   # gitignored: holds data derived from the match, NEVER committed

# Team name: names the CSV and the document that are submitted.
TEAM_NAME = "404-possession-not-found"

# ---------------------------------------------------------------- match (from DATA.md)
STEM = "BAR-ATH_tactical"
N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)                         # metres (length, width)
HALVES = [(11185, 90651), (110808, 190197)]   # graded frames, inclusive


def asset(name):
    suffix = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt",
              "smart_tagging": "_smart_tagging.xml"}[name]
    return os.path.join(DATA_DIR, "tactical", STEM + suffix)


# ---------------------------------------------------------------- evidence
# Where the per-frame evidence handed to the decoder comes from:
#   "model"  gradient-boosting model fitted on external matches (src/model.py,
#            bench/train_model.py). It uses the tracking only (no Smart Tagging).
#   "rules"  hand-weighted rules (src/dead.py, src/team.py) plus Smart Tagging: the v2 system.
# Chosen on external matches only (docs/BENCH_MODEL.md), never on the challenge match.
EVIDENCE = "model"

# Fitted models of the "model" evidence (models/<file>) and the decoder costs that go with each
# (the rules keep the ones in PARAMS). All chosen on external matches, never on the challenge match.
#   v4c  3 Metrica + 7 DFL matches, with the touch features (src/touch_features.py); fitted by
#        python -m bench.v4_final; costs from nested validation (docs/BENCH_V4_FIX.md)
#   v3   3 Metrica matches, fitted by python -m bench.train_model; costs from
#        docs/BENCH_MODEL_CV.md. Regenerates the previous submission.
MODELS = {
    "v4c": {"file": "possession_v4c.pkl", "costs": {"switch_cost": 9.0, "dead_cost": 48.0}},
    "v3": {"file": "possession_hgb.pkl", "costs": {"switch_cost": 6.0, "dead_cost": 48.0}},
}
MODEL_VERSION = "v4c"      # the submitted one

# ---------------------------------------------------------------- parameters
# Set from physical reasoning or on the external bench, NEVER tuned against the public ground
# truth (forbidden). Every change must be justified and logged in docs/JOURNAL.md.
PARAMS = {
    # ball cleaning
    "ball_max_gap_s": 0.6,          # shorter gaps are linearly interpolated
    "ball_speed_smooth": 5,         # frames of rolling median for the ball speed
    # team evidence
    "control_radius_m": 2.0,        # ball "at the feet" of a player
    "near_radius_m": 4.0,           # ball nearby, weak evidence
    "team_gain_per_m": 1.0,         # log-ratio per metre of nearest-player advantage
    "team_llr_clip": 3.0,
    "ball_flight_speed": 15.0,      # m/s: above this the ball is in flight -> damped evidence
    "flight_factor": 0.2,
    "smart_tag_team_llr": 0.3,      # small push for a phase tagged with a team
    # dead ball
    "dead_prior_logit": -1.5,
    "out_margin_m": 0.5,            # ball beyond the lines by more than this
    "slow_players_mps": 1.0,        # mean player speed of "everybody walking"
    "fast_players_mps": 2.0,
    "still_ball_mps": 0.5,
    "still_ball_s": 2.0,
    "long_gap_s": 2.0,
    "set_piece_tail_s": 1.5,        # a set-piece tag ends ~1.5 s after the actual restart
    # temporal model (in nats; cost of changing state)
    # Both are chosen on the external bench (python -m bench.run, see docs/BENCH.md), never on
    # the challenge match.
    "switch_cost": 12.0,            # H <-> A          (v1: 8)
    "dead_cost": 48.0,              # alive <-> D      (v1: 6)
}

# Smart Tagging phases whose 'Team' is the team IN possession (hypothesis to check visually on
# the video; the "DEFENDING ..." ones are left out until their meaning is confirmed).
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                   "ATTACKING TRANSITION", "LONG BALL"}
TEAM_TAG = {"barcelona": +1, "atletic club": -1, "athletic club": -1}
