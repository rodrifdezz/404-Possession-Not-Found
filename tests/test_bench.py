"""Tests of the external bench and of the decoder costs. Synthetic data only.

    python -m unittest discover -s tests -v
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

from bench import degrade, idsse, run, targets
from src import hmm


class ViterbiBatchTest(unittest.TestCase):
    def test_same_as_viterbi(self):
        rng = np.random.default_rng(0)
        em = rng.normal(size=(400, 3)) * 2
        em[100:120] = 0.0                         # ties: the tie-breaking must match
        cfgs = [{"switch_cost": s, "dead_cost": d} for s in (0.0, 2.0, 9.0) for d in (0.0, 4.0)]
        trans = np.stack([hmm.transition_matrix(c) for c in cfgs])
        batch = hmm.viterbi_batch(np.broadcast_to(em, (len(cfgs),) + em.shape), trans)
        for k in range(len(cfgs)):
            np.testing.assert_array_equal(batch[k], hmm.viterbi(em, trans[k]))


class DeadCostTest(unittest.TestCase):
    def test_brief_dead_evidence_does_not_split_a_possession(self):
        # 2 s of H, 0.4 s of weak "stoppage" evidence (e.g. players slowing down), 2 s of H
        llr = np.full(110, 3.0)
        dead = np.full(len(llr), -4.0)
        dead[50:60] = 2.0
        half = [(0, len(llr) - 1)]
        cheap = hmm.decode(llr, dead, {"switch_cost": 8.0, "dead_cost": 6.0}, half)
        costly = hmm.decode(llr, dead, {"switch_cost": 8.0, "dead_cost": 48.0}, half)
        self.assertIn("D", cheap)                 # v1: one possession becomes two
        self.assertNotIn("D", costly)             # a real stoppage needs longer evidence


class TargetsTest(unittest.TestCase):
    def test_short_touch_goes_back_to_opponent(self):
        lab = np.array(list("H" * 100 + "A" * 20 + "H" * 100))
        self.assertNotIn("A", targets.absorb_uncontrolled(lab))

    def test_long_possession_is_kept(self):
        lab = np.array(list("H" * 100 + "A" * 60 + "H" * 100))
        np.testing.assert_array_equal(targets.absorb_uncontrolled(lab), lab)

    def test_before_a_stoppage_is_untouched(self):
        lab = np.array(list("H" * 100 + "A" * 20 + "D" * 100))
        np.testing.assert_array_equal(targets.absorb_uncontrolled(lab), lab)

    def test_alternating_touches_do_not_clash(self):
        lab = np.array(list("H" * 100 + "A" * 10 + "H" * 10 + "A" * 100))
        out = targets.absorb_uncontrolled(lab)
        self.assertEqual(set(np.unique(out)), {"H", "A"})
        self.assertTrue((out[:100] == "H").all() and (out[-100:] == "A").all())


def fake_match(n_half=3000, seed=0):
    """Synthetic match in the idsse.build format: 2 players per team."""
    rng = np.random.default_rng(seed)
    n = 2 * n_half + idsse.SEPARATOR
    t = np.arange(n) / 25.0
    ball = np.c_[52 + 30 * np.sin(t / 7), 34 + 20 * np.cos(t / 5)].astype(np.float32)
    owner = np.where(np.sin(t / 9) > 0, 1, -1)
    players = np.full((n, 4, 2), np.nan, dtype=np.float32)
    team = np.array([1, 1, -1, -1], dtype=np.int8)
    for j, s in enumerate(team):
        near = owner == s
        players[:, j] = ball + np.where(near[:, None], 1.0, 8.0) + rng.normal(size=(n, 2)) * .1
    labels = np.where(owner > 0, "H", "A").astype("<U1")
    labels[(t % 60) < 5] = "D"
    halves = [(0, n_half - 1), (n_half + idsse.SEPARATOR, n - 1)]
    labels[n_half:n_half + idsse.SEPARATOR] = "X"
    return {"players": players, "team": team, "ball": ball, "ball_z": np.zeros(n, np.float32),
            "labels": labels, "halves": halves, "pitch": (105.0, 68.0)}


class DegradationTest(unittest.TestCase):
    def test_ball_missing_41_percent(self):
        m = fake_match()
        d = degrade.degrade(m, "short_gaps", 0)
        graded = m["labels"] != "X"
        missing = np.isnan(d["ball"][graded, 0]).mean()
        self.assertGreaterEqual(missing, degrade.BALL_MISSING)
        self.assertLess(missing, degrade.BALL_MISSING + 0.05)

    def test_reproducible(self):
        m = fake_match()
        a, b = degrade.degrade(m, "long_gaps", 3), degrade.degrade(m, "long_gaps", 3)
        np.testing.assert_array_equal(a["ball"], b["ball"])


META = """<?xml version="1.0" encoding="utf-8"?>
<PutDataRequest><MatchInformation>
 <General MatchId="DFL-MAT-TEST01"/>
 <Environment PitchX="105.00" PitchY="68.00"/>
 <Teams>
  <Team TeamId="T-HOME" Role="home"><Players><Player PersonId="P1"/></Players></Team>
  <Team TeamId="T-AWAY" Role="guest"><Players><Player PersonId="P2"/></Players></Team>
 </Teams>
</MatchInformation></PutDataRequest>
"""


def positions_xml():
    out = ['<?xml version="1.0" encoding="utf-8"?>', "<PutDataRequest><Positions>"]
    for sec, n0 in (("firstHalf", 10000), ("secondHalf", 100000)):
        for team, person, x in (("T-HOME", "P1", -10), ("T-AWAY", "P2", 10)):
            out.append('<FrameSet GameSection="{}" TeamId="{}" PersonId="{}">'.format(
                sec, team, person))
            out += ['<Frame N="{}" X="{}" Y="0" />'.format(n0 + i, x) for i in range(4)]
            out.append("</FrameSet>")
        out.append('<FrameSet GameSection="{}" TeamId="BALL" PersonId="B">'.format(sec))
        for i, (status, owner) in enumerate([(1, 1), (1, 2), (0, 2), (1, 1)]):
            out.append('<Frame N="{}" X="0" Y="0" Z="0.1" BallStatus="{}" '
                       'BallPossession="{}" />'.format(n0 + i, status, owner))
        out.append("</FrameSet>")
        out.append('<FrameSet GameSection="{}" TeamId="referee" PersonId="R">'.format(sec))
        out.append('<Frame N="{}" X="0" Y="0" /></FrameSet>'.format(n0))
    out.append("</Positions></PutDataRequest>")
    return "\n".join(out)


class IdsseReaderTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        for name, text in (("DFL_02_01_matchinformation_DFL-MAT-TEST01.xml", META),
                           ("DFL_04_03_positions_raw_observed_DFL-MAT-TEST01.xml",
                            positions_xml())):
            with open(os.path.join(self.dir, name), "w", encoding="utf-8") as fh:
                fh.write(text)

    def tearDown(self):
        shutil.rmtree(self.dir)

    def test_format(self):
        m = idsse.load("TEST01", self.dir)
        self.assertEqual(m["halves"], [(0, 3), (4 + idsse.SEPARATOR, 7 + idsse.SEPARATOR)])
        self.assertEqual(list(m["team"]), [1, -1])
        self.assertEqual("".join(m["labels"][:4]), "HADH")
        self.assertTrue((m["labels"][4:4 + idsse.SEPARATOR] == "X").all())
        # origin at the centre -> corner: a ball at (0, 0) moves to the centre spot
        np.testing.assert_allclose(m["ball"][0], [52.5, 34.0])
        np.testing.assert_allclose(m["players"][0, 0], [42.5, 34.0])

    def test_no_cache_writes_nothing(self):
        idsse.load("TEST01", self.dir, use_cache=False)
        self.assertFalse(os.path.exists(os.path.join(self.dir, "cache")))

    def test_cache_gives_the_same(self):
        a, b = idsse.load("TEST01", self.dir), idsse.load("TEST01", self.dir)
        np.testing.assert_array_equal(a["labels"], b["labels"])
        np.testing.assert_array_equal(a["players"], b["players"])


class BenchEndToEndTest(unittest.TestCase):
    def test_scores_with_official_scorer(self):
        cfgs = run.configs({"switch_cost": [8], "dead_cost": [6, 48]})
        res = run.run_match(fake_match(), cfgs, [0], log=lambda *a: None)
        self.assertEqual(len(res), len(run.SCENARIOS) * 2)
        for scores in res.values():
            self.assertEqual(len(scores), 2)
            for s in scores:
                self.assertGreaterEqual(s["spell_f1"], 0.0)
                self.assertLessEqual(s["spell_f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
