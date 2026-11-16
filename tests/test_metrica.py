"""Tests of the Metrica reader (labels derived from events) and of the training degradations.

    python -m unittest discover -s tests -v

Synthetic events and tracking only.
"""

import unittest

import numpy as np

from bench import degrade, metrica


def ev(team, kind, start, end=None, sub="", period=1):
    return {"team": team, "type": kind, "sub": sub, "period": period,
            "start": start, "end": start if end is None else end}


class LabelsFromEventsTest(unittest.TestCase):

    def labels(self, events, n=400, min_spell_s=0.5):
        return "".join(metrica.labels_from_events(events, n, min_spell_s))

    def test_possession_is_the_team_of_the_last_control_event(self):
        lab = self.labels([ev(1, "SET PIECE", 10, sub="KICK OFF"), ev(1, "PASS", 20, 40),
                           ev(-1, "RECOVERY", 100), ev(-1, "PASS", 110, 300)])
        self.assertEqual(lab[:10], "X" * 10)                 # before the kick-off: not graded
        self.assertEqual(lab[10:100], "H" * 90)
        self.assertEqual(lab[100:301], "A" * 201)
        self.assertEqual(lab[301:], "X" * 99)                # after the last event

    def test_dead_from_ball_out_to_the_restart(self):
        lab = self.labels([ev(1, "SET PIECE", 0, sub="KICK OFF"), ev(1, "BALL OUT", 50, 80),
                           ev(-1, "SET PIECE", 200, sub="THROW IN"), ev(-1, "PASS", 210, 300)])
        self.assertEqual(lab[:80], "H" * 80)
        self.assertEqual(lab[80:200], "D" * 120)
        self.assertEqual(lab[200:301], "A" * 101)

    def test_goal_kick_annotated_as_pass_is_a_restart(self):
        lab = self.labels([ev(1, "SET PIECE", 0, sub="KICK OFF"),
                           ev(1, "SHOT", 50, 70, sub="OFF TARGET-OUT"),
                           ev(-1, "PASS", 180, 220, sub="GOAL KICK"), ev(-1, "PASS", 230, 300)])
        self.assertEqual(lab[70:180], "D" * 110)
        self.assertEqual(lab[180:301], "A" * 121)

    def test_foul_played_on_is_not_a_stoppage(self):
        # a foul followed by open play before the next set piece: advantage, the ball stays alive
        lab = self.labels([ev(1, "SET PIECE", 0, sub="KICK OFF"), ev(1, "FAULT RECEIVED", 50),
                           ev(1, "PASS", 60, 90), ev(1, "BALL OUT", 100, 120),
                           ev(-1, "SET PIECE", 200, sub="THROW IN"), ev(-1, "PASS", 210, 300)])
        self.assertEqual(lab[50:120], "H" * 70)
        self.assertEqual(lab[120:200], "D" * 80)

    def test_touch_without_control_goes_back(self):
        events = [ev(1, "SET PIECE", 0, sub="KICK OFF"), ev(-1, "RECOVERY", 100),
                  ev(1, "RECOVERY", 105), ev(1, "PASS", 110, 300)]
        self.assertNotIn("A", self.labels(events))                    # 5 frames = 0.2 s
        self.assertIn("A" * 5, self.labels(events, min_spell_s=0.0))  # rule switched off

    def test_real_possession_is_kept(self):
        lab = self.labels([ev(1, "SET PIECE", 0, sub="KICK OFF"), ev(-1, "RECOVERY", 100),
                           ev(1, "RECOVERY", 150), ev(1, "PASS", 160, 300)])
        self.assertEqual(lab[100:150], "A" * 50)                      # 2 s


def clean_match(n=3000, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(n) / 25.0
    ball = np.c_[52 + 30 * np.sin(t / 7), 34 + 20 * np.cos(t / 5)].astype(np.float32)
    players = (ball[:, None, :] + rng.normal(0, 6, (n, 22, 2))).astype(np.float32)
    team = np.array([1] * 11 + [-1] * 11, dtype=np.int8)
    labels = np.where(np.sin(t / 9) > 0, "H", "A").astype("<U1")
    return {"players": players, "team": team, "ball": ball, "labels": labels,
            "ball_z": np.full(n, np.nan, np.float32), "halves": [(0, n - 1)],
            "pitch": (105.0, 68.0)}


class TrainingDegradationTest(unittest.TestCase):

    def test_format_and_levels(self):
        m = clean_match()
        d = degrade.degrade_markov(m, "normal", 0)
        self.assertEqual(d["players"].shape, (len(m["ball"]), 46, 2))
        self.assertEqual(int((d["team"] > 0).sum()), 23)
        seen = (~np.isnan(d["ball"][:, 0])).mean()
        self.assertTrue(0.4 < seen < 0.8, seen)
        tracked = np.sum(~np.isnan(d["players"][:, :, 0]), axis=1).mean()
        self.assertTrue(19 < tracked < 22, tracked)

    def test_no_player_changes_team(self):
        m = clean_match()
        d = degrade.degrade_markov(m, "hard", 1)
        home_x = np.nanmean(d["players"][:, d["team"] > 0, 0])
        self.assertTrue(np.isfinite(home_x))
        per_frame = np.sum(~np.isnan(d["players"][:, d["team"] > 0, 0]), axis=1)
        self.assertLessEqual(per_frame.max(), 11)

    def test_reproducible(self):
        m = clean_match()
        a, b = degrade.degrade_markov(m, "easy", 3), degrade.degrade_markov(m, "easy", 3)
        np.testing.assert_array_equal(a["ball"], b["ball"])
        np.testing.assert_array_equal(a["players"], b["players"])


class FillDeadBallTest(unittest.TestCase):

    def test_holes_are_rebuilt_between_their_ends(self):
        ball = np.c_[np.linspace(0, 10, 50), np.zeros(50)].astype(np.float32)
        ball[20:30] = np.nan
        out = metrica.fill_dead_ball(ball)
        self.assertFalse(np.isnan(out).any())
        np.testing.assert_allclose(out[20], ball[19])        # stays where it went out
        np.testing.assert_allclose(out[29], ball[30])        # waits at the restart spot
        np.testing.assert_array_equal(out[:20], ball[:20])   # seen frames untouched


if __name__ == "__main__":
    unittest.main()
