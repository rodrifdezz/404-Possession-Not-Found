"""One command: data -> validated submission CSV.

    python -m src.make_submission              # writes outputs/<TEAM_NAME>.csv
    python -m src.make_submission --name my_try   # another name, to compare versions
"""

import argparse
import os
import subprocess
import sys

import numpy as np

from . import config as C
from . import load, pipeline


def summary(labels):
    graded = np.zeros(C.N_FRAMES, dtype=bool)
    for a, b in C.HALVES:
        graded[a:b + 1] = True
    lab = labels[graded]
    n_poss, durs = 0, []
    for a, b in C.HALVES:
        seg = labels[a:b + 1]
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts = np.concatenate(([0], change))
        ends = np.concatenate((change, [len(seg)]))
        for s, e in zip(starts, ends):
            if seg[s] in "HA":
                n_poss += 1
                durs.append((e - s) / C.FPS)
    minutes = graded.sum() / C.FPS / 60
    print("summary (full match, {:.0f} graded min)".format(minutes))
    for k in "HAD":
        print("  {}  {:6.1%}".format(k, np.mean(lab == k)))
    print("  possessions {}  ({:.1f}/min)  mean duration {:.1f} s  median {:.1f} s".format(
        n_poss, n_poss / minutes, np.mean(durs), np.median(durs)))


def write_csv(labels, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))


class ValidationError(Exception):
    """The official validator rejected the CSV."""


def validate(path):
    script = os.path.join(C.OFFICIAL_REPO, "evaluation", "validate_submission.py")
    if not os.path.isfile(script):
        print("(official validator not found at {}; check MTAH_REPO)".format(script))
        return
    if subprocess.run([sys.executable, script, path]).returncode != 0:
        raise ValidationError("the official validator rejects {}".format(path))


def run(name):
    labels = pipeline.predict(load.load_atd(), load.load_smart_tagging())
    path = os.path.join(C.OUT_DIR, name + ".csv")
    write_csv(labels, path)
    print("written", os.path.relpath(path, C.ROOT))
    summary(labels)
    validate(path)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=C.TEAM_NAME)
    try:
        run(ap.parse_args().name)
    except ValidationError as error:
        sys.exit("ERROR: {}".format(error))
