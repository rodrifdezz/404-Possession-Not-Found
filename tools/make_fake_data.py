"""Genera un partido FALSO con el mismo formato que los datos reales, más su ground truth.

Para qué: poder desarrollar, depurar y pedir ayuda a un LLM (ChatGPT, Claude...) con datos que
NO son del partido. Los datos reales no se pueden enseñar a ningún LLM (términos del reto).

    python tools/make_fake_data.py                       # escribe fake_data/ (gitignored)
    MTAH_DATA_DIR=fake_data python -m src.make_submission --name fake_v0
    python -m src.evaluate --gt fake_data/fake_gt.csv

Es una simulación muy simplificada: sirve para comprobar que el código funciona y tiene sentido,
no para medir cómo de bueno será en el partido real.
"""

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import config as C  # noqa: E402

L, W = C.PITCH
N_SLOTS = 46
PRE_KICKOFF_S = 10.0   # tramo D simulado al empezar la 2ª parte, antes del saque


def simulate(rng):
    n = C.N_FRAMES
    pos = np.full((n, N_SLOTS, 2), np.nan)
    ball = np.full((n, 2), np.nan)
    gt = np.full(n, "X", dtype="<U1")
    restarts = []   # (frame, team_sign, type)
    phases = []     # (start, end, team_sign)

    home_slots, away_slots = list(range(23)), list(range(23, 46))
    base = np.zeros((22, 2))
    for k in range(11):
        base[k] = (rng.uniform(8, 50), rng.uniform(6, 62))
        base[11 + k] = (L - base[k, 0], W - base[k, 1])
    team_of_player = np.array([1] * 11 + [-1] * 11)
    slot_of = np.array(home_slots[:11] + away_slots[:11])
    noise = np.zeros((22, 2))

    for h, (a, b) in enumerate(C.HALVES):
        f = a
        state_team = 1 if h == 0 else -1
        if h == 1:   # tramo D antes del saque de la 2ª parte
            kb = a + int(PRE_KICKOFF_S * C.FPS) - 1
            for t in range(a, kb + 1):
                noise += rng.normal(0, 0.03, noise.shape)
                pos[t, slot_of] = base + noise
                ball[t] = (L / 2, W / 2)
                gt[t] = "D"
            restarts.append((kb + 1, state_team, "Kick Off (Start)"))
            f = kb + 1
        else:
            restarts.append((a, state_team, "Kick Off (Start)"))
        carrier = int(rng.choice(np.flatnonzero(team_of_player == state_team)))
        while f <= b:
            dur = int(np.clip(rng.exponential(12), 0.8, 45) * C.FPS)
            seg_end = min(f + dur, b + 1)
            phases.append((f, seg_end, state_team))
            next_pass = f + int(rng.uniform(1.5, 4) * C.FPS)
            flight = None
            for t in range(f, seg_end):
                shift = (ball[t - 1, 0] - L / 2) * 0.3 if not np.isnan(ball[t - 1, 0]) else 0
                noise += rng.normal(0, 0.08, noise.shape) - 0.01 * noise
                p = base + noise
                p[:, 0] += shift
                pos[t, slot_of] = p
                if flight is None and t >= next_pass:
                    mates = np.flatnonzero((team_of_player == state_team) &
                                           (np.arange(22) != carrier))
                    new = int(rng.choice(mates))
                    flight = (t, t + int(0.6 * C.FPS), carrier, new)
                if flight is not None:
                    t0, t1, c0, c1 = flight
                    u = min(1.0, (t - t0) / float(t1 - t0))
                    ball[t] = (1 - u) * p[c0] + u * p[c1]
                    if t >= t1:
                        carrier, flight = c1, None
                        next_pass = t + int(rng.uniform(1.5, 4) * C.FPS)
                else:
                    ball[t] = p[carrier] + 0.6
                gt[t] = "H" if state_team > 0 else "A"
            f = seg_end
            if f > b:
                break
            if rng.random() < 0.75:      # robo: el otro equipo controla
                state_team = -state_team
                carrier = int(rng.choice(np.flatnonzero(team_of_player == state_team)))
            else:                        # balón muerto
                dd = int(np.clip(rng.exponential(12), 4, 30) * C.FPS)
                d_end = min(f + dd, b + 1)
                spot = (rng.uniform(5, L - 5), -1.5 if rng.random() < 0.5 else W + 1.5)
                for t in range(f, d_end):
                    noise += rng.normal(0, 0.02, noise.shape)
                    pos[t, slot_of] = base + noise
                    ball[t] = spot
                    gt[t] = "D"
                state_team = int(rng.choice([1, -1]))
                carrier = int(rng.choice(np.flatnonzero(team_of_player == state_team)))
                restarts.append((d_end, state_team, "Throw-in"))
                f = d_end
            if rng.random() < 0.05:      # el tracker cambia de slot a un jugador
                k = int(rng.integers(22))
                pool = home_slots if team_of_player[k] > 0 else away_slots
                free = [s for s in pool if s not in set(slot_of)]
                slot_of[k] = rng.choice(free)

    # el tracker pierde el balón ~40% del tiempo y a algunos jugadores
    vis, seen = True, np.ones(n, dtype=bool)
    for t in range(n):
        vis = (rng.random() > 1 / 45) if vis else (rng.random() < 1 / 30)
        seen[t] = vis
    ball[~seen] = np.nan
    pos[rng.random(pos.shape[:2]) < 0.05] = np.nan
    return pos, ball, gt, restarts, phases


def write(out, pos, ball, gt, restarts, phases):
    folder = os.path.join(out, "tactical")
    os.makedirs(folder, exist_ok=True)
    stem = os.path.join(folder, C.STEM)

    players = "".join('<Player id="P{0}" teamId="{1}"><Name>Track {0}</Name>'
                      '<ShirtNumber>{0}</ShirtNumber></Player>'.format(
                          i + 1, "FIFATMA" if i < 23 else "FIFATMB") for i in range(N_SLOTS))
    with open(stem + "_atd.xml", "w") as fh:
        fh.write('<?xml version="1.0"?><Main><Metadata><Teams>'
                 '<Team id="FIFATMA"><Name>Barcelona</Name></Team>'
                 '<Team id="FIFATMB"><Name>Atletic Club</Name></Team>'
                 '</Teams><Players>{}</Players></Metadata></Main>'.format(players))

    print("escribiendo tracking falso (~1 min)...")
    with open(stem + "_atd.txt", "w") as fh:
        for t in range(len(ball)):
            groups = []
            for s in range(N_SLOTS):
                x, y = pos[t, s]
                groups.append("NaN,NaN,NaN" if np.isnan(x) else
                              "{:.5f},{:.5f},0".format(x / L, y / W))
            bx, by = ball[t]
            b = "NaN,NaN,NaN" if np.isnan(bx) else "{:.5f},{:.5f},0".format(bx / L, by / W)
            fh.write("{}:{}:{}\n".format(t + 1, ";".join(groups), b))

    def team_name(sign):
        return "Barcelona" if sign > 0 else "Atletic Club"

    inst = []
    for i, (fr, sign, typ) in enumerate(restarts):
        team = "N/A" if typ == "Throw-in" else team_name(sign)
        inst.append((fr / C.FPS - 3.5, fr / C.FPS + 1.5, "SET PIECES",
                     {"Team": team, "Type": typ}))
    for s, e, sign in phases:
        if (e - s) > 8 * C.FPS:
            inst.append((s / C.FPS + 1, e / C.FPS - 1, "BUILD UP", {"Team": team_name(sign)}))
    with open(stem + "_smart_tagging.xml", "w") as fh:
        fh.write("<file><ALL_INSTANCES>")
        for i, (s, e, code, tags) in enumerate(sorted(inst)):
            labels = "".join("<label><group>{}</group><text>{}</text></label>".format(g, v)
                             for g, v in tags.items())
            fh.write("<instance><ID>{}</ID><start>{:.2f}</start><end>{:.2f}</end>"
                     "<code>{}</code>{}</instance>".format(i + 1, s, e, code, labels))
        fh.write("</ALL_INSTANCES></file>")

    with open(os.path.join(out, "fake_gt.csv"), "w") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(gt))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default=os.path.join(C.ROOT, "fake_data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    write(args.out, *simulate(np.random.default_rng(args.seed)))
    print("listo:", args.out)
