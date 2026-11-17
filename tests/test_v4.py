"""Tests of the v4 model: touch features (src/touch_features.py), the submitted v4 model through
the submission pipeline, and the experimental layer-2 extras (bench/v4_stack.py). Synthetic data.

    python -m unittest discover -s tests -v
"""

import json
import os
import unittest

import numpy as np

from bench import v4_stack
from src import config as C
from src import frame_features, hmm, model, pipeline
from src import touch_features as v4_touch
from tests.test_model import PITCH, fake_tracking, swapped

V4C_PATH = model.model_path("v4c")
HAS_V4C = os.path.isfile(V4C_PATH)


def kick(n=200, turn=100, gap=None, player_near=True):
    """Ball at 10 m/s along x that turns to 10 m/s along y at `turn`; one home player follows
    the ball (or stays far away), one away player is far."""
    t = np.arange(n, dtype=np.float64)
    x = 30 + np.minimum(t, turn) * 0.4
    y = 34 + np.maximum(t - turn, 0) * 0.4
    ball = np.c_[x, y]
    home = ball.copy() if player_near else ball + 20.0
    away = ball + np.array([0.0, 25.0])
    players = np.stack([home, away], axis=1)
    if gap is not None:
        ball[gap[0]:gap[1]] = np.nan
    return {"players": players.astype(np.float32), "team": np.array([1, -1], dtype=np.int8),
            "ball": ball.astype(np.float32)}


class TouchDetectionTest(unittest.TestCase):

    def test_kick_next_to_a_player_is_one_touch_of_his_team(self):
        t = v4_touch.detect(kick())
        self.assertEqual(len(t["frame"]), 1)          # duplicates are suppressed
        self.assertLessEqual(abs(int(t["frame"][0]) - 100), 3)
        self.assertEqual(int(t["team"][0]), 1)

    def test_kink_without_a_player_is_not_a_touch(self):
        self.assertEqual(len(v4_touch.detect(kick(player_near=False))["frame"]), 0)

    def test_gap_edges_are_never_touches(self):
        """The turn happens while the ball is unseen: an interpolation would show a kink at the
        edges of the gap, the detector (raw ball, fully observed windows) must not."""
        t = v4_touch.detect(kick(gap=(90, 115)))
        self.assertEqual(len(t["frame"]), 0)

    def test_straight_ball_has_no_touch(self):
        self.assertEqual(len(v4_touch.detect(kick(turn=10 ** 6))["frame"]), 0)

    def test_features_are_symmetric(self):
        atd = fake_tracking(n=1500, slots=8, seed=3)
        atd["ball"] = atd["ball"] + np.where(np.arange(1500)[:, None] % 40 < 20, 0.0, 1.0)
        T, names = v4_touch.features(atd)
        T2, _ = v4_touch.features(swapped(atd))
        np.testing.assert_allclose(T2, frame_features.swap_teams(T, names), equal_nan=True)

    def test_features_shape_and_no_touch_input(self):
        T, names = v4_touch.features(kick(turn=10 ** 6))
        self.assertEqual(T.shape, (200, len(v4_touch.NAMES)))
        self.assertTrue((T[:, names.index("x_tlast_team")] == 0).all())


class Stage2ExtrasTest(unittest.TestCase):

    def test_swapping_teams_flips_the_extras(self):
        P = np.random.default_rng(0).dirichlet([1, 1, 1], 800)
        a = v4_stack.swap_teams(v4_stack.features(P))
        b = v4_stack.features(P[:, [1, 0, 2]])
        np.testing.assert_allclose(a, b, atol=1e-9, equal_nan=True)

    def test_reversal_only_when_past_and_future_disagree(self):
        n = 400
        P = np.full((n, 3), [0.8, 0.1, 0.1])
        P[200:] = [0.1, 0.8, 0.1]
        E = v4_stack.features(P)
        rev = E[:, v4_stack.NAMES.index("s2_rev25")]
        self.assertEqual(rev[50], 0.0)
        self.assertLess(rev[200], 0.0)                   # the future favours away


class ModelVersionsTest(unittest.TestCase):
    """v4 is the submitted model; v3 is kept, with its own file and costs."""

    def test_versions(self):
        self.assertEqual(C.MODEL_VERSION, "v4c")
        self.assertEqual(C.MODELS["v4c"]["costs"], {"switch_cost": 9.0, "dead_cost": 48.0})
        self.assertEqual(C.MODELS["v3"]["costs"], {"switch_cost": 6.0, "dead_cost": 48.0})
        self.assertNotEqual(model.model_path("v4c"), model.model_path("v3"))
        self.assertEqual(model.model_path(), model.model_path("v4c"))
        self.assertEqual(model.MODEL_PATH, model.model_path("v3"))      # bench/train_model.py

    def test_unknown_version_is_rejected(self):
        with self.assertRaises(ValueError):
            model.model_path("v5")
        with self.assertRaises(ValueError):
            pipeline.params("model", version="v5")
        with self.assertRaises(ValueError):
            model.load_version("v5")

    def test_missing_model_file(self):
        with self.assertRaises(FileNotFoundError):
            model.load(os.path.join(C.ROOT, "models", "does_not_exist.pkl"))

    def test_decoder_costs_per_version(self):
        """What hmm.decode receives (it reads switch_cost / dead_cost from these params)."""
        for version, (s, d) in {"v4c": (9.0, 48.0), "v3": (6.0, 48.0)}.items():
            p = pipeline.params("model", version=version)
            self.assertEqual((p["switch_cost"], p["dead_cost"]), (s, d))
            self.assertEqual(hmm.transition_matrix(p)[0, 1], -s)
            self.assertEqual(hmm.transition_matrix(p)[0, 2], -d)
        self.assertEqual(pipeline.params("model"), pipeline.params("model", version="v4c"))

    def test_touch_features_appended_last(self):
        atd = fake_tracking(n=600)
        X, names = model.matrix(atd, C.PARAMS, PITCH, [(0, 599)])
        Xt, names_t = model.matrix(atd, C.PARAMS, PITCH, [(0, 599)], touch=True)
        self.assertEqual(names_t, list(names) + v4_touch.NAMES)
        self.assertEqual(Xt.dtype, np.float32)
        np.testing.assert_array_equal(Xt[:, :len(names)], np.asarray(X, dtype=np.float32))


@unittest.skipUnless(HAS_V4C, "models/possession_v4c.pkl not found")
class SubmittedV4ModelTest(unittest.TestCase):
    """The v4 model through the submission pipeline (src/pipeline.py, defaults of config)."""

    @classmethod
    def setUpClass(cls):
        cls.bundle = model.load(V4C_PATH)
        cls.atd = fake_tracking(n=1500)
        cls.halves = [(0, 699), (750, 1499)]

    def predict(self, atd):
        return pipeline.predict(atd, [], None, PITCH, self.halves)

    def test_bundle(self):
        self.assertTrue(model.uses_touch(self.bundle))
        self.assertFalse(model.uses_touch(model.load(model.model_path("v3"))))
        self.assertEqual(self.bundle["costs"], C.MODELS["v4c"]["costs"])
        self.assertEqual(len(self.bundle["train"]), 10)
        _, names = model.matrix(self.atd, C.PARAMS, PITCH, self.halves, touch=True)
        self.assertEqual(list(self.bundle["names"]), names)
        self.assertEqual(len(names), 107)
        self.assertEqual(names[-18:], v4_touch.NAMES)
        _, base = model.matrix(self.atd, C.PARAMS, PITCH, self.halves)
        self.assertEqual(names[:89], list(base))
        self.assertEqual(list(model.load_version("v3")["names"]), list(base))

    def test_incompatible_features_are_rejected(self):
        X, names = model.matrix(self.atd, C.PARAMS, PITCH, self.halves)        # no touch
        with self.assertRaises(ValueError):
            model.predict_proba(X, names, self.bundle)

    def test_costs_of_model_and_config_must_agree(self):
        saved = C.MODELS["v4c"]
        try:
            C.MODELS["v4c"] = dict(saved, costs={"switch_cost": 6.0, "dead_cost": 48.0})
            with self.assertRaises(ValueError):
                model.load_version("v4c")
        finally:
            C.MODELS["v4c"] = saved
        self.assertIs(model.load_version("v4c"), self.bundle)

    def test_v3_and_v4_do_not_share_state(self):
        """Loading one model never changes the other (separate cache entries, separate files)."""
        v3 = model.load_version("v3")
        self.assertIsNot(v3, self.bundle)
        lab4 = self.predict(self.atd)
        lab3 = pipeline.predict(self.atd, [], None, PITCH, self.halves, "model", "v3")
        np.testing.assert_array_equal(self.predict(self.atd), lab4)
        np.testing.assert_array_equal(
            pipeline.predict(self.atd, [], None, PITCH, self.halves, "model", "v3"), lab3)

    def test_inference_needs_only_src_and_the_model(self):
        """A fresh interpreter predicts with the submission pipeline without importing bench/
        (no external match, no feature cache, no bench loader)."""
        import subprocess
        import sys
        code = ("import sys; from src import pipeline; from tests.test_model import fake_tracking;"
                "lab = pipeline.predict(fake_tracking(n=800), [], None, (105.0, 68.0), "
                "[(0, 799)]); assert set(lab) <= set('HAD'), set(lab);"
                "bad = [m for m in sys.modules if m == 'bench' or m.startswith('bench.')];"
                "assert not bad, bad; print('ok')")
        out = subprocess.run([sys.executable, "-c", code], cwd=C.ROOT, capture_output=True,
                             text=True)
        self.assertEqual(out.returncode, 0, out.stderr[-2000:])
        self.assertIn("ok", out.stdout)

    def test_metadata(self):
        with open(V4C_PATH.replace(".pkl", ".json")) as fh:
            info = json.load(fh)
        self.assertEqual(info["features"], list(self.bundle["names"]))
        self.assertEqual(info["decoder_costs"], C.MODELS["v4c"]["costs"])
        import hashlib
        import sklearn
        self.assertEqual(info["sklearn"], sklearn.__version__)
        with open(V4C_PATH, "rb") as fh:
            self.assertEqual(info["pkl_md5"], hashlib.md5(fh.read()).hexdigest())

    def test_labels(self):
        lab = self.predict(self.atd)
        self.assertEqual(len(lab), 1500)
        self.assertLessEqual(set(lab), set("HADX"))
        self.assertTrue((lab[700:750] == "X").all())
        self.assertNotIn("X", lab[:700])

    def test_deterministic(self):
        np.testing.assert_array_equal(self.predict(self.atd), self.predict(self.atd))

    def test_swapping_teams_swaps_h_and_a(self):
        lab = self.predict(self.atd)
        other = self.predict(swapped(self.atd))
        expected = np.where(lab == "H", "A", np.where(lab == "A", "H", lab))
        np.testing.assert_array_equal(other, expected)

    def test_probabilities_are_valid(self):
        X, names = model.matrix(self.atd, C.PARAMS, PITCH, self.halves, touch=True)
        P = model.predict_proba(X, names, self.bundle)
        self.assertEqual(P.shape, (1500, 3))
        self.assertTrue(((P >= 0) & (P <= 1)).all())
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=2e-4)


if __name__ == "__main__":
    unittest.main()
