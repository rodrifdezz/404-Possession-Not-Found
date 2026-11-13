"""Score every CSV in outputs/ with the OFFICIAL scorer.

    python -m src.evaluate                     # against the public 5 minutes
    python -m src.evaluate --gt path/gt.csv    # against another ground truth (e.g. the fake data's)
    python -m src.evaluate --bootstrap 1000    # confidence intervals

Reminder of the rules: the public 5 minutes are for checking that the pipeline is correct, NOT
for choosing parameters or training.
"""

import argparse
import os
import subprocess
import sys

from . import config as C

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gt", default="public")
    ap.add_argument("--bootstrap", type=int, default=None)
    args = ap.parse_args()
    cmd = [sys.executable, os.path.join(C.OFFICIAL_REPO, "evaluation", "run_evaluation.py"),
           "--results", C.OUT_DIR, "--out", os.path.join(C.OUT_DIR, "eval"), "--gt", args.gt]
    if args.bootstrap:
        cmd += ["--bootstrap", str(args.bootstrap)]
    sys.exit(subprocess.run(cmd).returncode)
