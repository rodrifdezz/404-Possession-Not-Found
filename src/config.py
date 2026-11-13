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
    # These three are chosen on the external bench (python -m bench.run, see docs/BENCH.md),
    # never on the challenge match. Until they are updated they keep their v1 values.
    "emission_temper": 1.0,         # frames per independent observation (see hmm.py)
    "switch_cost": 8.0,             # H <-> A
    "dead_cost": 6.0,               # alive <-> D
}

# Smart Tagging phases whose 'Team' is the team IN possession (hypothesis to check visually on
# the video; the "DEFENDING ..." ones are left out until their meaning is confirmed).
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                   "ATTACKING TRANSITION", "LONG BALL"}
TEAM_TAG = {"barcelona": +1, "atletic club": -1, "athletic club": -1}
