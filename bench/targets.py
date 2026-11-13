"""Bench target labels derived from the DFL flags.

DFL's possession flag can change with any touch. CHALLENGE.md, by contrast, says that a change of
possession requires control: "a simple touch of the ball without control" or "a deflection of a
pass that does not intercept it" do not count, and it suggests as a proxy that "a player keeps
the ball for a few seconds". We do not know which reading is closer to the challenge ground
truth, so the bench scores against both:

  dfl          the flag as it is;
  control_2s   a spell of one team shorter than 2 s BETWEEN two spells of the opponent (with no
               stoppage in between) is taken as a touch without control and given back.
"""

import numpy as np

CONTROL_S = 2.0
FPS = 25


def runs(labels):
    """[(start, exclusive_end, label), ...] of constant stretches."""
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate(([0], change))
    ends = np.concatenate((change, [len(labels)]))
    return [(int(s), int(e), labels[s]) for s, e in zip(starts, ends)]


def absorb_uncontrolled(labels, min_s=CONTROL_S):
    """Copy of labels where touches without control (see above) are given back to the opponent."""
    out = labels.copy()
    min_len = int(round(min_s * FPS))
    changed = True
    while changed:
        changed = False
        r = runs(out)
        i = 1
        while i < len(r) - 1:
            prev, (s, e, lab), nxt = r[i - 1][2], r[i], r[i + 1][2]
            if lab in "HA" and e - s < min_len and prev == nxt and prev in "HA" and prev != lab:
                out[s:e] = prev
                changed = True
                i += 2          # the next run no longer has the neighbour we saw: next pass
            else:
                i += 1
    return out


def all_targets(labels):
    return {"dfl": labels, "control_2s": absorb_uncontrolled(labels)}
