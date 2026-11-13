"""The whole algorithm as one function, with no file reading: tracking -> labels.

Used by the submission (make_submission, the challenge match) and by the external bench
(bench/, other matches), so that both run exactly the same code.

    atd   = {"players": (N, slots, 2) metres, "team": (slots,) +1 home / -1 away,
             "ball": (N, 2) metres, NaN where there is no ball}
    tags  = list of Smart Tagging phases (empty if there are none)
    pitch = (length, width) in metres, origin at a corner
"""

from . import config as C
from . import ball, dead, features, hmm, team


def evidence(atd, tags, p=None, pitch=None):
    """(team_llr, dead_logit) per frame: what the decoder receives."""
    p = C.PARAMS if p is None else p
    ball_xy, ball_speed = ball.clean_ball(atd["ball"], p)
    f = features.compute(atd, ball_xy, ball_speed)
    return team.team_llr(f, tags, p), dead.dead_logit(f, tags, p, pitch)


def predict(atd, tags, p=None, pitch=None, halves=None):
    """H/A/D labels inside the halves and X outside. None = challenge match values (config)."""
    p = C.PARAMS if p is None else p
    team_llr, dead_logit = evidence(atd, tags, p, pitch)
    return hmm.decode(team_llr, dead_logit, p, halves)
