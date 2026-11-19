"""Tests of the demo helpers (bench/demo.py). Synthetic data only.

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from bench import demo


class SpansTest(unittest.TestCase):

    def test_runs(self):
        self.assertEqual(demo.spans(np.array(list("HHADDD"))),
                         [[0, 2, "H"], [2, 3, "A"], [3, 6, "D"]])

    def test_offset(self):
        self.assertEqual(demo.spans(np.array(list("AAH")), offset=0)[1], [2, 3, "H"])


class BaselineTest(unittest.TestCase):

    def test_nearest_team_wins_and_x_outside_halves(self):
        n = 200
        players = np.zeros((n, 2, 2), dtype=np.float32)
        players[:, 0] = (10.0, 10.0)            # home player
        players[:, 1] = (90.0, 50.0)            # away player
        ball = np.tile(np.array([12.0, 10.0], dtype=np.float32), (n, 1))
        ball[120:] = (88.0, 50.0)
        atd = {"players": players, "team": np.array([1, -1]), "ball": ball}
        lab = demo.nearest_smooth(atd, [(10, 189)], window=5)
        self.assertTrue((lab[:10] == "X").all() and (lab[190:] == "X").all())
        self.assertTrue((lab[10:110] == "H").all())
        self.assertTrue((lab[130:190] == "A").all())


if __name__ == "__main__":
    unittest.main()
