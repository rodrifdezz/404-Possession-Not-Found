"""Pruebas de make_submission.validate(): si el validador oficial rechaza el CSV, se falla.

    python -m unittest discover -s tests -v

Con validadores falsos (siempre disponibles) y, si está el repo oficial, con el de verdad.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

from src import config as C
from src import make_submission

VALIDATOR = os.path.join(C.OFFICIAL_REPO, "evaluation", "validate_submission.py")


class ValidadorFalsoTest(unittest.TestCase):
    """Un repo oficial de mentira cuyo validador termina con el código que se le pida."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)
        self.csv = os.path.join(self.tmp, "sub.csv")
        open(self.csv, "w").close()

    def run_validate(self, exit_code=None):
        if exit_code is not None:
            os.makedirs(os.path.join(self.tmp, "evaluation"))
            with open(os.path.join(self.tmp, "evaluation", "validate_submission.py"), "w") as fh:
                fh.write("import sys\nsys.exit({})\n".format(exit_code))
        with mock.patch.object(C, "OFFICIAL_REPO", self.tmp), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            make_submission.validate(self.csv)
        return out.getvalue()

    def test_validador_rechaza_falla(self):
        with self.assertRaises(make_submission.ValidationError):
            self.run_validate(exit_code=1)

    def test_validador_acepta_no_falla(self):
        self.run_validate(exit_code=0)

    def test_sin_validador_avisa(self):
        self.assertIn("no encuentro el validador oficial", self.run_validate())


@unittest.skipUnless(os.path.isfile(VALIDATOR), "no está el repo oficial (MTAH_REPO)")
class ValidadorOficialTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, n_rows):
        path = os.path.join(self.tmp, "sub.csv")
        with open(path, "w") as fh:
            fh.write("frame,label\n")
            fh.writelines("{},X\n".format(i) for i in range(n_rows))
        return path

    def test_csv_correcto_pasa(self):
        with contextlib.redirect_stdout(io.StringIO()):
            make_submission.validate(self.write(C.N_FRAMES))

    def test_csv_con_filas_de_menos_falla(self):
        with self.assertRaises(make_submission.ValidationError):
            make_submission.validate(self.write(C.N_FRAMES - 1))


if __name__ == "__main__":
    unittest.main()
