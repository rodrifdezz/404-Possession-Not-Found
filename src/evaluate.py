"""Puntúa con el scorer OFICIAL todos los CSV de outputs/.

    python -m src.evaluate                     # contra los 5 minutos públicos
    python -m src.evaluate --gt ruta/gt.csv    # contra otra ground truth (p. ej. la de datos falsos)
    python -m src.evaluate --bootstrap 1000    # intervalos de confianza

Recordatorio de las reglas: los 5 minutos públicos sirven para comprobar que el pipeline es
correcto, NO para elegir parámetros ni entrenar.
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
