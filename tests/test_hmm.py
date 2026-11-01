"""Pruebas del modelo temporal: emisiones normalizadas y detección de D.

    python -m unittest discover -s tests -v
"""

import unittest

import numpy as np

from src import hmm
from src.hmm import A, D, H


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class EmisionesTest(unittest.TestCase):

    def test_cada_fotograma_suma_uno(self):
        rng = np.random.default_rng(0)
        team_llr = rng.normal(0, 3, 1000)
        dead_logit = rng.normal(0, 5, 1000)
        em = hmm.emissions(team_llr, dead_logit)
        np.testing.assert_allclose(np.exp(em).sum(axis=1), 1.0, rtol=1e-12)

    def test_vivo_muerto_separado_de_equipo(self):
        team_llr = np.array([-3.0, 0.0, 0.7, 3.0])
        dead_logit = np.array([1.0, -2.0, 0.0, 4.0])
        p = np.exp(hmm.emissions(team_llr, dead_logit))
        p_dead = sigmoid(dead_logit)

        np.testing.assert_allclose(p[:, D], p_dead)
        np.testing.assert_allclose(p[:, H] + p[:, A], 1.0 - p_dead)
        # p(H | vivo) depende solo de la evidencia de equipo
        np.testing.assert_allclose(p[:, H] / (p[:, H] + p[:, A]), sigmoid(team_llr))

    def test_evidencia_de_equipo_no_cambia_p_muerto(self):
        dead_logit = np.full(5, 0.5)
        team_llr = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        em = hmm.emissions(team_llr, dead_logit)
        np.testing.assert_allclose(em[:, D], em[0, D])

    def test_h_frente_a_a_igual_que_v0(self):
        # la diferencia H - A sigue siendo team_llr: el reparto entre equipos no cambia
        team_llr = np.linspace(-3, 3, 13)
        em = hmm.emissions(team_llr, np.zeros(13))
        np.testing.assert_allclose(em[:, H] - em[:, A], team_llr, atol=1e-12)

    def test_estable_con_valores_extremos(self):
        z = np.array([-1000.0, -50.0, 0.0, 50.0, 1000.0])
        em = hmm.emissions(z, z[::-1])
        self.assertTrue(np.all(np.isfinite(em)))
        self.assertTrue(np.all(em <= 0.0))
        np.testing.assert_allclose(np.exp(em).sum(axis=1), 1.0, rtol=1e-12)


class DeteccionDTest(unittest.TestCase):

    def test_parada_con_jugador_cerca_del_balon_es_d(self):
        # Balón muerto (p(D) = 0.73) pero con un jugador del Barça encima del balón (llr = +3),
        # como en un saque de banda o una falta. En v0 la evidencia de equipo sumaba +1.5 a H y
        # tapaba la parada; con emisiones normalizadas el tramo tiene que salir D.
        n_vivo, n_muerto = 200, 100
        team_llr = np.full(2 * n_vivo + n_muerto, 3.0)
        dead_logit = np.full_like(team_llr, -4.0)
        dead_logit[n_vivo:n_vivo + n_muerto] = 1.0

        path = hmm.viterbi(hmm.emissions(team_llr, dead_logit), hmm.transition_matrix())

        self.assertTrue(np.all(path[:n_vivo] == H))
        self.assertTrue(np.all(path[n_vivo:n_vivo + n_muerto] == D))
        self.assertTrue(np.all(path[n_vivo + n_muerto:] == H))

    def test_sin_evidencia_de_muerto_no_hay_d(self):
        team_llr = np.concatenate((np.full(300, 2.0), np.full(300, -2.0)))
        path = hmm.viterbi(hmm.emissions(team_llr, np.full(600, -3.0)),
                           hmm.transition_matrix())
        self.assertNotIn(D, set(path.tolist()))
        self.assertEqual(path[0], H)
        self.assertEqual(path[-1], A)


if __name__ == "__main__":
    unittest.main()
