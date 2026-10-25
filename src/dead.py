"""Pieza 2 - balón muerto: devuelve log-odds de "D" por fotograma (positivo = muerto).

Cada regla suma o resta evidencia. Para probar una regla nueva: añádela aquí con su parámetro en
config.PARAMS y comprueba su efecto en el partido completo (nº de tramos D, duración), no en los
5 minutos públicos.
"""

import numpy as np

from . import config as C
from .ball import nan_runs


def _run_length_at_least(mask, n):
    """True en los frames que pertenecen a una racha True de longitud >= n."""
    out = np.zeros_like(mask)
    s, e = nan_runs(mask)
    for a, b in zip(s, e):
        if b - a >= n:
            out[a:b] = True
    return out


def dead_logit(f, tags, p=C.PARAMS):
    n = C.N_FRAMES
    L, W = C.PITCH
    z = np.full(n, p["dead_prior_logit"], dtype=np.float64)

    # 1) balón fuera del campo
    x, y = f["ball_xy"][:, 0], f["ball_xy"][:, 1]
    m = p["out_margin_m"]
    with np.errstate(invalid="ignore"):
        outside = (x < -m) | (x > L + m) | (y < -m) | (y > W + m)
    z[outside] += 4.0

    # 2) ritmo de los jugadores
    ps = np.nan_to_num(f["player_speed"], nan=p["fast_players_mps"])
    z[ps < p["slow_players_mps"]] += 2.0
    z[(ps >= p["slow_players_mps"]) & (ps < 1.4)] += 1.0
    z[ps > p["fast_players_mps"]] -= 1.0

    # 3) balón quieto durante un rato
    with np.errstate(invalid="ignore"):
        still = f["ball_seen"] & (f["ball_speed"] < p["still_ball_mps"])
    z[_run_length_at_least(still, int(p["still_ball_s"] * C.FPS))] += 1.5

    # 4) balón perdido mucho rato
    z[_run_length_at_least(~f["ball_raw_seen"], int(p["long_gap_s"] * C.FPS))] += 0.5

    # 5) Smart Tagging: la parte inicial de un SET PIECES es la preparación del saque
    tail = int(p["set_piece_tail_s"] * C.FPS)
    for t in tags:
        if t["code"] == "SET PIECES":
            z[max(t["start"], 0):max(t["end"] - tail, 0)] += 1.0

    # 6) tramos conocidos
    for a, b in C.KNOWN_DEAD:
        z[a:b + 1] += 20.0
    return z
