"""One command: data -> validated submission CSV.

    python -m src.make_submission                    # writes outputs/<TEAM_NAME>.csv (v4 model)
    python -m src.make_submission --name my_try      # another name, to compare versions
    python -m src.make_submission --model v3 --name v3   # the previous submission (v3 model)
    python -m src.make_submission --evidence rules   # the rule-based system (v2) instead

The evidence source is config.EVIDENCE and the model config.MODEL_VERSION unless --evidence /
--model are given. The file is fully determined by the data bundle and this repository: the md5
printed at the end must be the same on every machine.
"""

import argparse
import hashlib
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
    with open(path, "w", newline="\n") as fh:      # same bytes on Windows, macOS and Linux
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


def md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def run(name, source=None, version=None):
    source = C.EVIDENCE if source is None else source
    version = C.MODEL_VERSION if version is None else version
    labels = pipeline.predict(load.load_atd(), load.load_smart_tagging(), source=source,
                              version=version)
    path = os.path.join(C.OUT_DIR, name + ".csv")
    write_csv(labels, path)
    print("written {} (evidence: {}{})".format(
        os.path.relpath(path, C.ROOT), source,
        ", model {} {}".format(version, C.MODELS[version]["costs"]) if source == "model" else ""))
    summary(labels)
    validate(path)
    print("md5 {}  (must be identical on every machine)".format(md5(path)))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--name", default=C.TEAM_NAME)
    ap.add_argument("--evidence", choices=pipeline.SOURCES, default=None)
    ap.add_argument("--model", choices=sorted(C.MODELS), default=None)
    args = ap.parse_args()
    try:
        run(args.name, args.evidence, args.model)
    except ValidationError as error:
        sys.exit("ERROR: {}".format(error))
