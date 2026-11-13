"""Tests of the learned evidence (src/learned.py). Synthetic data only.

    python -m unittest discover -s tests -v
"""

import json
import os
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np

from src import config as C
from src import learned, pipeline

from tests.test_bench import fake_match


class CentredMeanTest(unittest.TestCase):
    def test_matches_a_direct_computation(self):
        x = np.random.default_rng(0).normal(size=40)
        got = learned.centred_mean(x, 5)
        want = [x[max(i - 2, 0):i + 3].mean() for i in range(40)]
        np.testing.assert_allclose(got, want)


class FeaturesTest(unittest.TestCase):
    def setUp(self):
        self.m = fake_match()
        self.m["ball"][100:200] = np.nan          # a ball gap, as in real tracking

    def test_shapes_and_no_missing_values(self):
        dead_X, team_X = learned.features(self.m, self.m["pitch"], self.m["halves"])
        n = len(self.m["ball"])
        self.assertEqual(dead_X.shape, (n, len(learned.DEAD_FEATURES)))
        self.assertEqual(team_X.shape, (n, len(learned.TEAM_FEATURES)))
        self.assertTrue(np.isfinite(dead_X).all() and np.isfinite(team_X).all())

    def test_swapping_teams_flips_team_features_only(self):
        swapped = dict(self.m, team=-self.m["team"])
        d1, t1 = learned.features(self.m, self.m["pitch"], self.m["halves"])
        d2, t2 = learned.features(swapped, self.m["pitch"], self.m["halves"])
        np.testing.assert_allclose(d1, d2, atol=1e-5)
        np.testing.assert_allclose(t1, -t2, atol=1e-5)


class LogisticTest(unittest.TestCase):
    def test_recovers_known_coefficients(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20000, 3))
        y = (rng.random(20000) < 1 / (1 + np.exp(-(0.5 + X @ [2.0, -1.0, 0.0])))).astype(float)
        model = learned.fit_logistic(X, y, ridge=0.0)
        coef = np.asarray(model["coef"]) / np.asarray(model["scale"])
        np.testing.assert_allclose(coef, [2.0, -1.0, 0.0], atol=0.1)
        logit = learned.predict(model, X)
        self.assertGreater(np.corrcoef(logit, X @ [2.0, -1.0, 0.0])[0, 1], 0.99)

    def test_without_intercept_is_antisymmetric(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(5000, 4))
        y = (X[:, 0] + rng.normal(size=5000) > 0.3).astype(float)   # biased data
        model = learned.fit_logistic(X, y, intercept=False)
        np.testing.assert_allclose(learned.predict(model, -X), -learned.predict(model, X))


class LearnedPipelineTest(unittest.TestCase):
    def test_pipeline_runs_with_a_model_file(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        rng = np.random.default_rng(2)
        dead_X, team_X = rng.normal(size=(500, len(learned.DEAD_FEATURES))), \
            rng.normal(size=(500, len(learned.TEAM_FEATURES)))
        model = {"dead_features": learned.DEAD_FEATURES, "team_features": learned.TEAM_FEATURES,
                 "dead": learned.fit_logistic(dead_X, (dead_X[:, 0] > 0).astype(float)),
                 "team": learned.fit_logistic(team_X, (team_X[:, 0] > 0).astype(float),
                                              intercept=False)}
        path = os.path.join(tmp, "model.json")
        with open(path, "w") as fh:
            json.dump(model, fh)
        m = fake_match()
        with mock.patch.object(C, "ROOT", tmp), mock.patch.object(C, "EVIDENCE", "learned"), \
                mock.patch.object(C, "EVIDENCE_MODEL", "model.json"):
            labels = pipeline.predict(m, [], C.PARAMS, m["pitch"], m["halves"])
        graded = m["labels"] != "X"
        self.assertLessEqual(set(labels[graded]), set("HAD"))
        self.assertEqual(set(labels[~graded]), {"X"})

    def test_model_with_other_features_is_refused(self):
        tmp = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, tmp)
        path = os.path.join(tmp, "old.json")
        with open(path, "w") as fh:
            json.dump({"dead_features": ["x"], "team_features": ["y"]}, fh)
        with self.assertRaises(ValueError):
            learned.load_model(path)


if __name__ == "__main__":
    unittest.main()
