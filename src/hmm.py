"""Pieza 4 - convertir evidencias por fotograma en posesiones limpias (Viterbi, 3 estados).

Estados 0=H, 1=A, 2=D. Cada fotograma aporta una puntuación por estado (emisiones, en log) y
cambiar de estado cuesta (switch_cost entre H y A, dead_cost entre vivo y muerto). Viterbi
encuentra la secuencia con mejor puntuación total: pocas posesiones y cambios en el sitio justo.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2


def emissions(team_llr, dead_logit):
    """(N,3) log-puntuaciones. log p(D) / log p(vivo) a partir del logit de muerto."""
    log_dead = -np.logaddexp(0.0, -dead_logit)
    log_alive = -np.logaddexp(0.0, dead_logit)
    e = np.empty((len(team_llr), 3))
    e[:, H] = log_alive + team_llr / 2.0
    e[:, A] = log_alive - team_llr / 2.0
    e[:, D] = log_dead
    return e


def transition_matrix(p=C.PARAMS):
    s, d = p["switch_cost"], p["dead_cost"]
    return np.array([[0.0, -s, -d],
                     [-s, 0.0, -d],
                     [-d, -d, 0.0]])


def viterbi(em, trans):
    n = len(em)
    back = np.zeros((n, 3), dtype=np.int8)
    score = em[0].copy()
    t = trans.tolist()
    e = em.tolist()
    s0, s1, s2 = score
    for i in range(1, n):
        row = (s0, s1, s2)
        new = []
        for j in range(3):
            c0, c1, c2 = row[0] + t[0][j], row[1] + t[1][j], row[2] + t[2][j]
            if c0 >= c1 and c0 >= c2:
                best, arg = c0, 0
            elif c1 >= c2:
                best, arg = c1, 1
            else:
                best, arg = c2, 2
            back[i, j] = arg
            new.append(best + e[i][j])
        s0, s1, s2 = new
    path = np.empty(n, dtype=np.int8)
    path[-1] = int(np.argmax([s0, s1, s2]))
    for i in range(n - 1, 0, -1):
        path[i - 1] = back[i, path[i]]
    return path


def decode(team_llr, dead_logit, p=C.PARAMS):
    """Etiquetas 'H'/'A'/'D' en las partes, 'X' fuera. Cada parte se decodifica por separado."""
    labels = np.full(C.N_FRAMES, "X", dtype="<U1")
    em = emissions(team_llr, dead_logit)
    trans = transition_matrix(p)
    for a, b in C.HALVES:
        path = viterbi(em[a:b + 1], trans)
        labels[a:b + 1] = np.array(["H", "A", "D"])[path]
    return labels
