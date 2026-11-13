"""Tests of the ATD cache (src/load.py) on tiny synthetic datasets.

    python -m unittest discover -s tests -v

No real data: each test writes its own few-frame ATD in a temporary folder and points DATA_DIR
and CACHE_DIR there, so the repo's cache/ is never touched.
"""

import contextlib
import io
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import config as C
from src import load

N_FRAMES = 5
N_SLOTS = 4


def write_dataset(data_dir, ball_x):
    """Fake ATD: 4 slots (2 per team) and the ball at x=ball_x (0-1) on every frame."""
    folder = os.path.join(data_dir, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, C.STEM)
    players = "".join('<Player id="P{0}" teamId="{1}"/>'.format(i, "T1" if i < 2 else "T2")
                      for i in range(N_SLOTS))
    with open(stem + "_atd.xml", "w") as fh:
        fh.write('<Main><Teams><Team id="T1"/><Team id="T2"/></Teams>'
                 '<Players>{}</Players></Main>'.format(players))
    raw = stem + "_atd.txt"
    with open(raw, "w") as fh:
        for f in range(N_FRAMES):
            slots = ";".join("0.{0},0.{0},0".format(s + 1) for s in range(N_SLOTS))
            fh.write("{}:{}:{},0.5,0\n".format(f + 1, slots, ball_x))
    return raw


class CacheTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.cache_dir = os.path.join(self.tmp, "cache")
        self.patches = [mock.patch.object(C, "N_FRAMES", N_FRAMES),
                        mock.patch.object(C, "CACHE_DIR", self.cache_dir)]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        shutil.rmtree(self.tmp)

    def load_from(self, data_dir):
        """load_atd() on data_dir; returns (atd, number of times the txt was parsed)."""
        with mock.patch.object(C, "DATA_DIR", data_dir), \
                mock.patch.object(load, "_parse_raw", wraps=load._parse_raw) as parse, \
                contextlib.redirect_stdout(io.StringIO()):
            atd = load.load_atd()
        return atd, parse.call_count

    def test_different_datasets_do_not_share_cache(self):
        real, fake = os.path.join(self.tmp, "real"), os.path.join(self.tmp, "fake")
        raw_real = write_dataset(real, 0.25)
        write_dataset(fake, 0.75)
        # the dangerous case: the dataset processed second is OLDER than the cache
        old = os.path.getmtime(raw_real) - 3600
        os.utime(raw_real, (old, old))

        atd_fake, _ = self.load_from(fake)
        atd_real, parsed = self.load_from(real)

        self.assertEqual(parsed, 1)
        np.testing.assert_allclose(atd_fake["ball"][:, 0], 0.75 * C.PITCH[0], rtol=1e-6)
        np.testing.assert_allclose(atd_real["ball"][:, 0], 0.25 * C.PITCH[0], rtol=1e-6)
        self.assertEqual(len(os.listdir(self.cache_dir)), 2)

    def test_same_dataset_reuses_cache(self):
        data = os.path.join(self.tmp, "real")
        write_dataset(data, 0.25)

        first, parsed_first = self.load_from(data)
        second, parsed_second = self.load_from(data)

        self.assertEqual((parsed_first, parsed_second), (1, 0))
        for k in ("players", "ball", "team"):
            np.testing.assert_array_equal(first[k], second[k])

    def test_editing_the_txt_invalidates_the_cache(self):
        data = os.path.join(self.tmp, "real")
        raw = write_dataset(data, 0.25)
        meta = raw.replace("_atd.txt", "_atd.xml")
        self.load_from(data)

        # only the TXT changes (content and mtime); the XML stays untouched
        meta_stat = os.stat(meta)
        mtime = os.path.getmtime(raw)
        with open(raw) as fh:
            text = fh.read()
        with open(raw, "w") as fh:
            fh.write(text.replace(":0.25,0.5,0", ":0.75,0.5,0"))
        os.utime(raw, (mtime + 10, mtime + 10))   # different mtime even with a coarse clock
        self.assertEqual(os.stat(meta).st_mtime_ns, meta_stat.st_mtime_ns)
        atd, parsed = self.load_from(data)

        self.assertEqual(parsed, 1)
        np.testing.assert_allclose(atd["ball"][:, 0], 0.75 * C.PITCH[0], rtol=1e-6)


if __name__ == "__main__":
    unittest.main()
