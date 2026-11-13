"""Tests of make_submission.validate(): if the official validator rejects the CSV, it fails.

    python -m unittest discover -s tests -v

With fake validators (always available) and, if the official repo is present, the real one.
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


class FakeValidatorTest(unittest.TestCase):
    """A dummy official repo whose validator exits with whatever code we ask for."""

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

    def test_validator_rejects_fails(self):
        with self.assertRaises(make_submission.ValidationError):
            self.run_validate(exit_code=1)

    def test_validator_accepts_passes(self):
        self.run_validate(exit_code=0)

    def test_missing_validator_warns(self):
        self.assertIn("official validator not found", self.run_validate())


@unittest.skipUnless(os.path.isfile(VALIDATOR), "official repo not available (MTAH_REPO)")
class OfficialValidatorTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, self.tmp)

    def write(self, n_rows):
        path = os.path.join(self.tmp, "sub.csv")
        with open(path, "w") as fh:
            fh.write("frame,label\n")
            fh.writelines("{},X\n".format(i) for i in range(n_rows))
        return path

    def test_correct_csv_passes(self):
        with contextlib.redirect_stdout(io.StringIO()):
            make_submission.validate(self.write(C.N_FRAMES))

    def test_csv_with_missing_rows_fails(self):
        with self.assertRaises(make_submission.ValidationError):
            make_submission.validate(self.write(C.N_FRAMES - 1))


if __name__ == "__main__":
    unittest.main()
