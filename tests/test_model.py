"""Tests of the learned evidence (src/frame_features.py, src/stack.py, src/model.py).

    python -m unittest discover -s tests -v

Synthetic data only. The properties checked are the ones the submission relies on: the model
cannot favour a side (swapping the teams swaps H and A), its probabilities reach the decoder
unchanged, and the output is deterministic.
"""

import os
import unittest

import numpy as np

from src import config as C
from src import frame_features, hmm, model, pipeline, stack

PITCH = (105.0, 68.0)
HAS_MODEL = os.path.isfile(model.MODEL_PATH)


def fake_tracking(n=1500, slots=8, seed=0):
    """Random-walk players and ball, with holes, in the load_atd() format."""
    rng = np.random.default_rng(seed)
    team = np.array([1] * (slots // 2) + [-1] * (slots // 2), dtype=np.int8)
    start = np.c_[rng.uniform(10, 95, slots), rng.uniform(5, 63, slots)]
    start[team > 0, 0] -= 8                       # home stands closer to x = 0
    players = start + np.cumsum(rng.normal(0, 0.15, (n, slots, 2)), axis=0)
    ball = np.array([52.0, 34.0]) + np.cumsum(rng.normal(0, 0.4, (n, 2)), axis=0)
    players[rng.random((n, slots)) < 0.05] = np.nan
    ball[400:460] = np.nan
    ball[rng.random(n) < 0.1] = np.nan
    return {"players": players.astype(np.float32), "team": team, "ball": ball.astype(np.float32)}


def swapped(atd):
    return dict(atd, team=-atd["team"])


class FeatureSymmetryTest(unittest.TestCase):

    def setUp(self):
        self.atd = fake_tracking()
        self.halves = [(0, 699), (750, 1499)]

    def matrix(self, atd):
        return model.matrix(atd, C.PARAMS, PITCH, self.halves)

    def test_swapping_teams_swaps_the_blocks(self):
        X, names = self.matrix(self.atd)
        X2, names2 = self.matrix(swapped(self.atd))
        self.assertEqual(names, names2)
        np.testing.assert_allclose(X2, frame_features.swap_teams(X, names), rtol=1e-4,
                                   atol=1e-4, equal_nan=True)

    def test_swap_twice_is_identity(self):
        X, names = self.matrix(self.atd)
        back = frame_features.swap_teams(frame_features.swap_teams(X, names), names)
        np.testing.assert_array_equal(back, X)

    def test_layer2_swap_matches_swapped_inputs(self):
        X, names = self.matrix(self.atd)
        P = np.random.default_rng(1).dirichlet([1, 1, 1], len(X))
        a = stack.swap_teams(stack.features(P, X, names))
        b = stack.features(P[:, [1, 0, 2]], frame_features.swap_teams(X, names), names)
        np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True)


class EvidenceTest(unittest.TestCase):

    def test_decoder_receives_the_model_probabilities(self):
        P = np.random.default_rng(0).dirichlet([1, 1, 1], 500)
        team_llr, dead_logit = model.to_evidence(P)
        np.testing.assert_allclose(np.exp(hmm.emissions(team_llr, dead_logit)), P, rtol=1e-9)

    def test_zero_probability_is_floored(self):
        team_llr, dead_logit = model.to_evidence(np.array([[1.0, 0.0, 0.0]]))
        self.assertTrue(np.isfinite(team_llr).all() and np.isfinite(dead_logit).all())

    def test_unknown_source_is_rejected(self):
        with self.assertRaises(ValueError):
            pipeline.params("magic")

    def test_each_source_has_its_own_costs(self):
        self.assertEqual(pipeline.params("rules")["switch_cost"], C.PARAMS["switch_cost"])
        for version, spec in C.MODELS.items():
            p = pipeline.params("model", version=version)
            self.assertEqual(p["switch_cost"], spec["costs"]["switch_cost"])
            self.assertEqual(p["dead_cost"], spec["costs"]["dead_cost"])
        self.assertEqual(pipeline.params("model"), pipeline.params("model", version=C.MODEL_VERSION))


@unittest.skipUnless(HAS_MODEL, "models/possession_hgb.pkl not found")
class StoredModelTest(unittest.TestCase):
    """The v3 model (models/possession_hgb.pkl), kept to regenerate the previous submission."""

    @classmethod
    def setUpClass(cls):
        cls.atd = fake_tracking(n=1500)
        cls.halves = [(0, 699), (750, 1499)]

    def predict(self, atd):
        return pipeline.predict(atd, [], None, PITCH, self.halves, "model", "v3")

    def test_labels(self):
        lab = self.predict(self.atd)
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
        X, names = model.matrix(self.atd, C.PARAMS, PITCH, self.halves)
        P = model.predict_proba(X, names, model.load(model.MODEL_PATH))
        self.assertEqual(P.shape, (len(X), 3))
        self.assertTrue(((P >= 0) & (P <= 1)).all())
        np.testing.assert_allclose(P.sum(axis=1), 1.0, atol=2e-4)


@unittest.skipUnless(HAS_MODEL, "models/possession_hgb.pkl not found")
class BenchWithModelTest(unittest.TestCase):

    def test_bench_scores_model_and_rules_on_the_same_data(self):
        from bench import degrade, run
        n = 3000
        t = np.arange(n) / 25.0
        atd = fake_tracking(n=n, slots=22, seed=2)
        labels = np.where(np.sin(t / 9) > 0, "H", "A").astype("<U1")
        labels[(t % 60) < 5] = "D"
        m = {"players": atd["players"], "team": atd["team"],
             "ball": np.nan_to_num(atd["ball"], nan=50.0), "ball_z": np.zeros(n, np.float32),
             "labels": labels, "halves": [(0, n - 1)], "pitch": PITCH}
        cfgs = run.configs({"switch_cost": [6], "dead_cost": [12, 48]})
        scores, ref = run.run_match(m, cfgs, [0], log=lambda *a: None, source="model",
                                    reference=True)
        self.assertEqual(set(scores), set(ref))
        self.assertEqual(len(scores), len(degrade.SCENARIOS) * 2)
        for key in scores:
            self.assertEqual(len(scores[key]), len(cfgs))
            self.assertIn("spell_f1", ref[key])


if __name__ == "__main__":
    unittest.main()
