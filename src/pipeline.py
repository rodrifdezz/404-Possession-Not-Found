"""The whole algorithm as one function, with no file reading: tracking -> labels.

Used by the submission (make_submission, the challenge match) and by the external bench
(bench/, other matches), so that both run exactly the same code.

    atd    = {"players": (N, slots, 2) metres, "team": (slots,) +1 home / -1 away,
              "ball": (N, 2) metres, NaN where there is no ball}
    tags   = list of Smart Tagging phases (empty if there are none; only the rules use them)
    pitch  = (length, width) in metres, origin at a corner
    halves = [(start, inclusive_end), ...]
    source = "model" or "rules"; None = config.EVIDENCE
"""

from . import config as C
from . import ball, dead, features, hmm, team

SOURCES = ("model", "rules")


def params(source=None, p=None):
    """Parameters for an evidence source: PARAMS, with the decoder costs of that source."""
    source = C.EVIDENCE if source is None else source
    if source not in SOURCES:
        raise ValueError("unknown evidence source: {}".format(source))
    if p is not None:
        return p
    return dict(C.PARAMS, **C.MODEL_COSTS) if source == "model" else C.PARAMS


def evidence(atd, tags, p=None, pitch=None, halves=None, source=None):
    """(team_llr, dead_logit) per frame: what the decoder receives."""
    source = C.EVIDENCE if source is None else source
    p = params(source, p)
    if source == "model":
        from . import model            # scikit-learn is only needed for this source
        return model.evidence(atd, p, pitch, halves)
    ball_xy, ball_speed = ball.clean_ball(atd["ball"], p)
    f = features.compute(atd, ball_xy, ball_speed)
    return team.team_llr(f, tags, p), dead.dead_logit(f, tags, p, pitch)


def predict(atd, tags, p=None, pitch=None, halves=None, source=None):
    """H/A/D labels inside the halves and X outside. None = challenge match values (config)."""
    source = C.EVIDENCE if source is None else source
    p = params(source, p)
    team_llr, dead_logit = evidence(atd, tags, p, pitch, halves, source)
    return hmm.decode(team_llr, dead_logit, p, halves)
