"""Step 3 - who is in control: per-frame log-ratio (positive = Barça/H, negative = Athletic/A).

Distance advantage of each team's nearest player, damped while the ball is in flight, plus a weak
push from Smart Tagging. Natural next step: detect touches (sudden changes of ball speed or
direction) and require sustained control before possession changes.
"""

import numpy as np

from . import config as C


def team_llr(f, tags, p=C.PARAMS):
    dh = np.nan_to_num(f["d_home"], nan=99.0)
    da = np.nan_to_num(f["d_away"], nan=99.0)
    closest = np.minimum(dh, da)
    adv = np.clip(p["team_gain_per_m"] * (da - dh), -p["team_llr_clip"], p["team_llr_clip"])

    llr = np.zeros(len(dh))
    control = f["ball_seen"] & (closest < p["control_radius_m"])
    near = f["ball_seen"] & ~control & (closest < p["near_radius_m"])
    llr[control] = adv[control]
    llr[near] = 0.3 * adv[near]

    with np.errstate(invalid="ignore"):
        flying = f["ball_speed"] > p["ball_flight_speed"]
    llr[flying] *= p["flight_factor"]
    return add_tag_llr(llr, tags, p)


def add_tag_llr(llr, tags, p=C.PARAMS):
    """Add, in place, the Smart Tagging evidence of the team in possession: a weak push during
    attacking phases. Returns llr."""
    for t in tags:
        sign = C.TEAM_TAG.get(t["tags"].get("team", "").lower(), 0)
        if sign and t["code"] in C.ATTACKING_CODES:
            llr[max(t["start"], 0):max(t["end"], 0)] += sign * p["smart_tag_team_llr"]
    return llr
