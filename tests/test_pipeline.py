"""Prueba de integración: datos falsos -> make_submission.run() -> CSV, con el pipeline real.

    python -m unittest discover -s tests -v

Nada de datos reales. El partido falso lo genera tools/make_fake_data.py, reducido a N_FRAMES
fotogramas cambiando solo las constantes de config (N_FRAMES, HALVES, KNOWN_DEAD) mientras dura
la prueba; PARAMS y el algoritmo no se tocan. Datos, caché y salidas van a una carpeta temporal,
así que data/, cache/ y outputs/ del repo no se usan.

El validador oficial exige 199319 filas por defecto, así que con el CSV reducido se le pasa
--rows N_FRAMES: mismas comprobaciones de formato, otra longitud. No es la validación de entrega.
"""

import contextlib
import csv
import importlib.util
import io
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import config as C
from src import make_submission

N_FRAMES = 4000
HALVES = [(250, 1850), (2150, 3750)]
KNOWN_DEAD = [(2150, 2399)]
VALIDATOR = os.path.join(C.OFFICIAL_REPO, "evaluation", "validate_submission.py")


def load_fake_data_tool():
    path = os.path.join(C.ROOT, "tools", "make_fake_data.py")
    spec = importlib.util.spec_from_file_location("make_fake_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PipelineTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp()
        data_dir = os.path.join(cls.tmp, "fake_data")
        patches = [mock.patch.object(C, "N_FRAMES", N_FRAMES),
                   mock.patch.object(C, "HALVES", HALVES),
                   mock.patch.object(C, "KNOWN_DEAD", KNOWN_DEAD),
                   mock.patch.object(C, "DATA_DIR", data_dir),
                   mock.patch.object(C, "CACHE_DIR", os.path.join(cls.tmp, "cache")),
                   mock.patch.object(C, "OUT_DIR", os.path.join(cls.tmp, "outputs")),
                   # sin validador dentro de run(): pediría 199319 filas (test_validador_oficial)
                   mock.patch.object(C, "OFFICIAL_REPO", os.path.join(cls.tmp, "sin_repo"))]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            fake = load_fake_data_tool()
            fake.write(data_dir, *fake.simulate(np.random.default_rng(0)))
            cls.path = make_submission.run("pipeline_test")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp)

    def test_csv_completo_y_bien_formado(self):
        self.assertTrue(os.path.isfile(self.path))
        with open(self.path, newline="") as fh:
            rows = list(csv.reader(fh))

        self.assertEqual(rows[0], ["frame", "label"])
        frames = [int(f) for f, _ in rows[1:]]
        labels = [l for _, l in rows[1:]]
        # una sola comparación cubre: nº de filas, 0..N-1, en orden, sin huecos ni duplicados
        self.assertEqual(frames, list(range(N_FRAMES)))
        self.assertLessEqual(set(labels), set("HADX"))

        graded = np.zeros(N_FRAMES, dtype=bool)
        for a, b in HALVES:
            graded[a:b + 1] = True
        labels = np.array(labels)
        self.assertEqual(set(labels[~graded]), {"X"})

    @unittest.skipUnless(os.path.isfile(VALIDATOR), "no está el repo oficial (MTAH_REPO)")
    def test_validador_oficial(self):
        out = subprocess.run([sys.executable, VALIDATOR, self.path, "--rows", str(N_FRAMES)],
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                             universal_newlines=True)
        self.assertEqual(out.returncode, 0, out.stdout)
        self.assertIn("Valid.", out.stdout)


if __name__ == "__main__":
    unittest.main()
