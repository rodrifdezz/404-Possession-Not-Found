"""Step 4 - turn per-frame evidence into clean possessions (Viterbi, 3 states).

States 0=H, 1=A, 2=D. Each frame contributes a score per state (emissions, in log) and changing
state has a cost (switch_cost between H and A, dead_cost between alive and dead). Viterbi finds
the sequence with the best total score: few possessions, with changes in the right place.

Temperature (emission_temper, in frames). At 25 fps consecutive frames are not independent
observations: a player next to the ball for 0.2 s is ONE clue, not five. Without correcting for
this the evidence is counted ~25 times per second, and a few frames of an opponent near the ball
are enough to pay for a change of possession and its way back. Dividing the log emissions by the
temperature amounts to counting one independent observation every `emission_temper` frames.
With temperature 1 the model is exactly v1.
"""

import numpy as np

from . import config as C

H, A, D = 0, 1, 2


def log_sigmoid(z):
    """log(1 / (1 + e^-z)) without overflow for large |z|."""
    return -np.logaddexp(0.0, -z)


def emissions(team_llr, dead_logit, temper=1.0):
    """(N,3) log-probabilities divided by `temper`.

    Before dividing, p(H) + p(A) + p(D) = 1 on every frame:
        p(D) = sigmoid(dead_logit)
        p(H) = p(alive) * sigmoid(team_llr)      p(A) = p(alive) * sigmoid(-team_llr)

    Team evidence only splits p(alive) between H and A; it cannot take weight away from D.
    """
    log_alive = log_sigmoid(-dead_logit)
    e = np.empty((len(team_llr), 3))
    e[:, H] = log_alive + log_sigmoid(team_llr)
    e[:, A] = log_alive + log_sigmoid(-team_llr)
    e[:, D] = log_sigmoid(dead_logit)
    return e / temper


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


def viterbi_batch(em, trans):
    """K decodings at once: em (K,N,3), trans (K,3,3) -> paths (K,N).

    Same sums and same tie-breaking as viterbi() (the lowest state index wins), so each path is
    identical to viterbi(em[k], trans[k]). Lets the external bench try many parameter
    combinations without repeating the Python loop.
    """
    k, n, _ = em.shape
    back = np.zeros((n, k, 3), dtype=np.int8)
    score = em[:, 0].copy()
    rows = np.arange(k)[:, None]
    cols = np.arange(3)[None, :]
    for i in range(1, n):
        cand = score[:, :, None] + trans          # (K, from, to)
        arg = cand.argmax(axis=1)                 # (K, to); first maximum = lowest index
        back[i] = arg
        score = cand[rows, arg, cols] + em[:, i]
    path = np.empty((k, n), dtype=np.int8)
    path[:, -1] = score.argmax(axis=1)
    for i in range(n - 1, 0, -1):
        path[:, i - 1] = back[i, np.arange(k), path[:, i]]
    return path


def decode(team_llr, dead_logit, p=C.PARAMS, halves=None):
    """'H'/'A'/'D' labels inside the halves, 'X' outside. Each half is decoded on its own.

    halves = [(start, inclusive_end), ...]; None = the challenge match (config.HALVES).
    """
    halves = C.HALVES if halves is None else halves
    labels = np.full(len(team_llr), "X", dtype="<U1")
    em = emissions(team_llr, dead_logit, p.get("emission_temper", 1.0))
    trans = transition_matrix(p)
    for a, b in halves:
        path = viterbi(em[a:b + 1], trans)
        labels[a:b + 1] = np.array(["H", "A", "D"])[path]
    return labels
