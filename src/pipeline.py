"""The whole algorithm as one function, with no file reading: tracking -> labels.

Used by the submission (make_submission, the challenge match) and by the external bench
(bench/, other matches), so that both run exactly the same code.

    atd    = {"players": (N, slots, 2) metres, "team": (slots,) +1 home / -1 away,
              "ball": (N, 2) metres, NaN where there is no ball}
    tags   = list of Smart Tagging phases (empty if there are none)
    pitch  = (length, width) in metres, origin at a corner
    halves = [(start, inclusive_end), ...]
"""

from . import config as C
from . import ball, dead, features, hmm, learned, team


def evidence(atd, tags, p=None, pitch=None, halves=None, source=None):
    """(team_llr, dead_logit) per frame: what the decoder receives.

    source = "rules" or "learned"; None = config.EVIDENCE.
    """
    p = C.PARAMS if p is None else p
    source = C.EVIDENCE if source is None else source
    if source == "learned":
        team_llr, dead_logit = learned.evidence(atd, learned.load_model(), pitch, halves, p)
        return team.add_tag_llr(team_llr, tags, p), dead.add_tag_logit(dead_logit, tags, p)
    if source != "rules":
        raise ValueError("unknown evidence source: {}".format(source))
    ball_xy, ball_speed = ball.clean_ball(atd["ball"], p)
    f = features.compute(atd, ball_xy, ball_speed)
    return team.team_llr(f, tags, p), dead.dead_logit(f, tags, p, pitch)


def predict(atd, tags, p=None, pitch=None, halves=None):
    """H/A/D labels inside the halves and X outside. None = challenge match values (config)."""
    p = C.PARAMS if p is None else p
    team_llr, dead_logit = evidence(atd, tags, p, pitch, halves)
    return hmm.decode(team_llr, dead_logit, p, halves)
