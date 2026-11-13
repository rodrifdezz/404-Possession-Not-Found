"""Tests of the temporal model: normalised emissions and detection of D.

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from src import hmm
from src.hmm import A, D, H


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class EmissionsTest(unittest.TestCase):

    def test_each_frame_sums_to_one(self):
        rng = np.random.default_rng(0)
        team_llr = rng.normal(0, 3, 1000)
        dead_logit = rng.normal(0, 5, 1000)
        em = hmm.emissions(team_llr, dead_logit)
        np.testing.assert_allclose(np.exp(em).sum(axis=1), 1.0, rtol=1e-12)

    def test_alive_dead_separate_from_team(self):
        team_llr = np.array([-3.0, 0.0, 0.7, 3.0])
        dead_logit = np.array([1.0, -2.0, 0.0, 4.0])
        p = np.exp(hmm.emissions(team_llr, dead_logit))
        p_dead = sigmoid(dead_logit)

        np.testing.assert_allclose(p[:, D], p_dead)
        np.testing.assert_allclose(p[:, H] + p[:, A], 1.0 - p_dead)
        # p(H | alive) depends only on the team evidence
        np.testing.assert_allclose(p[:, H] / (p[:, H] + p[:, A]), sigmoid(team_llr))

    def test_team_evidence_does_not_change_p_dead(self):
        dead_logit = np.full(5, 0.5)
        team_llr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        em = hmm.emissions(team_llr, dead_logit)
        np.testing.assert_allclose(em[:, D], em[0, D])

    def test_h_versus_a_same_as_v0(self):
        # the H - A difference is still team_llr: the split between teams does not change
        team_llr = np.linspace(-3, 3, 13)
        em = hmm.emissions(team_llr, np.zeros(13))
        np.testing.assert_allclose(em[:, H] - em[:, A], team_llr, atol=1e-12)

    def test_stable_with_extreme_values(self):
        z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        em = hmm.emissions(z, z[::-1])
        self.assertTrue(np.all(np.isfinite(em)))
        self.assertTrue(np.all(em <= 0.0))
        np.testing.assert_allclose(np.exp(em).sum(axis=1), 1.0, rtol=1e-12)


class DeadDetectionTest(unittest.TestCase):

    def test_stoppage_with_player_near_ball_is_d(self):
        # Dead ball (p(D) = 0.73) but with a Barça player on the ball (llr = +3), as in a
        # throw-in or a free kick. In v0 the team evidence added +1.5 to H and hid the stoppage;
        # with normalised emissions the stretch must come out as D.
        n_alive, n_dead = 200, 100
        team_llr = np.full(2 * n_alive + n_dead, 3.0)
        dead_logit = np.full_like(team_llr, -4.0)
        dead_logit[n_alive:n_alive + n_dead] = 1.0

        path = hmm.viterbi(hmm.emissions(team_llr, dead_logit), hmm.transition_matrix())

        self.assertTrue(np.all(path[:n_alive] == H))
        self.assertTrue(np.all(path[n_alive:n_alive + n_dead] == D))
        self.assertTrue(np.all(path[n_alive + n_dead:] == H))

    def test_no_dead_evidence_no_d(self):
        team_llr = np.concatenate((np.full(300, 2.0), np.full(300, -2.0)))
        path = hmm.viterbi(hmm.emissions(team_llr, np.full(600, -3.0)),
                           hmm.transition_matrix())
        self.assertNotIn(D, set(path.tolist()))
        self.assertEqual(path[0], H)
        self.assertEqual(path[-1], A)


if __name__ == "__main__":
    unittest.main()
