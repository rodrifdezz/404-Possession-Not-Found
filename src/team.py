"""Pieza 3 - quién controla: log-ratio por fotograma (positivo = Barça/H, negativo = Athletic/A).

v0: ventaja de distancia del jugador más cercano de cada equipo, atenuada si el balón vuela,
más un empujón débil del Smart Tagging. Siguiente paso natural: detectar toques (cambios bruscos
de velocidad/dirección del balón) y exigir control sostenido para cambiar de posesión.
"""

import numpy as np

from . import config as C


def team_llr(f, tags, p=C.PARAMS):
    dh = np.nan_to_num(f["d_home"], nan=99.0)
    da = np.nan_to_num(f["d_away"], nan=99.0)
    closest = np.minimum(dh, da)
    adv = np.clip(p["team_gain_per_m"] * (da - dh), -p["team_llr_clip"], p["team_llr_clip"])

    llr = np.zeros(C.N_FRAMES)
    control = f["ball_seen"] & (closest < p["control_radius_m"])
    near = f["ball_seen"] & ~control & (closest < p["near_radius_m"])
    llr[control] = adv[control]
    llr[near] = 0.3 * adv[near]

    with np.errstate(invalid="ignore"):
        flying = f["ball_speed"] > p["ball_flight_speed"]
    llr[flying] *= p["flight_factor"]

    for t in tags:
        sign = C.TEAM_TAG.get(t["tags"].get("team", "").lower(), 0)
        if sign and t["code"] in C.ATTACKING_CODES:
            llr[max(t["start"], 0):max(t["end"], 0)] += sign * p["smart_tag_team_llr"]
    return llr
