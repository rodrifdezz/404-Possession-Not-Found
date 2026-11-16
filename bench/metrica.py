"""Reader for the 3 open Metrica Sports sample matches (other teams, anonymised).

    python -m bench.metrica --download      # git clone into external/metrica/ (~250 MB)
    python -m bench.metrica                 # summary of the matches and their labels

https://github.com/metrica-sports/sample-data : tracking at 25 fps plus HUMAN-ANNOTATED events.
These are other matches: nothing from the challenge match goes through here.

Unlike the DFL data (bench/idsse.py) these files carry no possession flag, so the per-frame label
is derived from the events:

  possession  team of the last control event (pass, set piece, shot, recovery, ball lost, carry);
  dead ball   from the event that stops play (ball out, foul, offside, goal, shot out) to the next
              restart (set piece; goal kicks, which these files annotate as passes), provided
              nobody plays the ball in between;
  touches     CHALLENGE.md: "a simple touch of the ball without control" does not change
              possession. A spell shorter than MIN_SPELL_S that directly follows a spell of the
              opponent (recovery and loss annotated almost together) goes back to the opponent.

A match is returned in the same format as bench/idsse.py: load(game) ->
    {"players": (N, slots, 2) m, "team": (slots,), "ball": (N, 2) m, "ball_z": (N,) NaN,
     "labels": (N,) 'H'/'A'/'D'/'X', "halves": [(start, inclusive_end)], "pitch": (L, W)}
"""

import argparse
import json
import os
import subprocess
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
METRICA_DIR = os.environ.get("METRICA_DIR") or os.path.join(ROOT, "external", "metrica")
URL = "https://github.com/metrica-sports/sample-data.git"
GAMES = [1, 2, 3]
PITCH = (105.0, 68.0)
FPS = 25
MIN_SPELL_S = 0.5

CONTROL_TYPES = {"PASS", "SET PIECE", "SHOT", "RECOVERY", "BALL LOST", "CARRY"}


# ----------------------------------------------------------------------------- events

def _events_csv(path):
    import pandas as pd
    out = []
    for r in pd.read_csv(path).itertuples(index=False):
        out.append({"team": 1 if r.Team == "Home" else -1, "type": str(r.Type).upper(),
                    "sub": r.Subtype.upper() if isinstance(r.Subtype, str) else "",
                    "period": int(r.Period), "start": int(r._4) - 1, "end": int(r._6) - 1})
    return out


def _events_json(path):
    with open(path) as fh:
        data = json.load(fh)["data"]
    out = []
    for r in data:
        sub = r.get("subtypes")
        if isinstance(sub, list):
            sub = "-".join(s["name"] for s in sub)
        elif isinstance(sub, dict):
            sub = sub["name"]
        out.append({"team": 1 if r["team"]["id"] == "FIFATMA" else -1,
                    "type": r["type"]["name"].upper(), "sub": (sub or "").upper(),
                    "period": int(r["period"]), "start": int(r["start"]["frame"]) - 1,
                    "end": int(r["end"]["frame"]) - 1})
    return out


def _stop_frame(ev):
    """Frame at which this event stops play, or None."""
    t, s = ev["type"], ev["sub"]
    if t == "BALL OUT":
        return ev["end"]
    if t == "SHOT" and ("GOAL" in s or "OUT" in s):
        return ev["end"]
    if t == "FAULT RECEIVED":
        return ev["start"]
    if t == "CHALLENGE" and "FAULT" in s and "ADVANTAGE" not in s:
        return ev["start"]
    if t == "BALL LOST" and ("OFFSIDE" in s or "END HALF" in s):
        return ev["end"]
    return None


def merge_touches(seg, min_len):
    """In place: a spell shorter than min_len right after a spell of the opponent goes back."""
    if min_len <= 1:
        return
    changed = True
    while changed:
        changed = False
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        st = np.concatenate(([0], ch))
        en = np.concatenate((ch, [len(seg)]))
        for i in range(1, len(st)):
            s, e, prev = st[i], en[i], seg[st[i - 1]]
            if seg[s] in "HA" and e - s < min_len and prev in "HA" and prev != seg[s]:
                seg[s:e] = prev
                changed = True
                break


def labels_from_events(events, n_frames, min_spell_s=MIN_SPELL_S):
    """Per-frame 'H'/'A'/'D' inside each period (kick-off -> last event), 'X' outside."""
    ev = sorted(events, key=lambda e: (e["start"], e["end"]))
    lab = np.full(n_frames, "X", dtype="<U1")
    for period in sorted(set(e["period"] for e in ev)):
        pe = [e for e in ev if e["period"] == period]
        kick = [e for e in pe if e["type"] == "SET PIECE" and "KICK OFF" in e["sub"]]
        if not kick:
            continue
        a = kick[0]["start"]
        b = max(max(e["start"], e["end"]) for e in pe)
        ctrl = [e for e in pe if e["type"] in CONTROL_TYPES and e["start"] >= a]
        team, k = 0, 0
        seg = np.zeros(b - a + 1, dtype=np.int8)
        for f in range(a, b + 1):
            while k < len(ctrl) and ctrl[k]["start"] <= f:
                team = ctrl[k]["team"]
                k += 1
            seg[f - a] = team
        lab[a:b + 1] = np.where(seg > 0, "H", "A")
        restarts = [e for e in pe if e["start"] > a and (
            e["type"] == "SET PIECE" or
            (e["type"] in ("PASS", "BALL LOST") and "GOAL KICK" in e["sub"]))]
        for sp in restarts:
            before = [e for e in pe if e["start"] < sp["start"]]
            played = [c["start"] for c in ctrl if c["start"] < sp["start"]]
            stops = [f for f in (_stop_frame(e) for e in before)
                     if f is not None and f < sp["start"] and not any(c > f for c in played)]
            if stops:
                d0 = max(stops)
            else:
                d0 = max([max(e["start"], e["end"]) for e in before] or [sp["start"]])
            d0 = max(d0, a)
            if d0 < sp["start"]:
                lab[d0:sp["start"]] = "D"
        merge_touches(lab[a:b + 1], int(round(min_spell_s * FPS)))
    return lab


# ----------------------------------------------------------------------------- tracking

def _folder(game):
    return os.path.join(METRICA_DIR, "data", "Sample_Game_{}".format(game))


def _read_csv_game(game):
    import pandas as pd
    base = os.path.join(_folder(game), "Sample_Game_{}_".format(game))
    home = pd.read_csv(base + "RawTrackingData_Home_Team.csv", skiprows=2)
    away = pd.read_csv(base + "RawTrackingData_Away_Team.csv", skiprows=2)

    def players(df):
        idx = [df.columns.get_loc(c) for c in df.columns if c.startswith("Player")]
        return np.stack([df.iloc[:, [i, i + 1]].to_numpy(dtype=np.float64) for i in idx], axis=1)

    ph, pa = players(home), players(away)
    bi = home.columns.get_loc("Ball")
    ball = home.iloc[:, [bi, bi + 1]].to_numpy(dtype=np.float64)
    team = np.array([1] * ph.shape[1] + [-1] * pa.shape[1], dtype=np.int8)
    return np.concatenate([ph, pa], axis=1), ball, team, _events_csv(base + "RawEventsData.csv")


def _read_epts_game(game):
    base = os.path.join(_folder(game), "Sample_Game_{}_".format(game))
    root = ET.parse(base + "metadata.xml").getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    ids = [p.get("id") for p in root.iter("Player")]
    side = {p.get("id"): (1 if p.get("teamId") == team_ids[0] else -1)
            for p in root.iter("Player")}
    channel = {c.get("id"): c.get("playerId") for c in root.iter("PlayerChannel")}
    col = {p: i for i, p in enumerate(ids)}
    # the order of the players on a line changes with substitutions: one spec per frame range
    specs = []
    for spec in root.iter("DataFormatSpecification"):
        refs = [channel[r.get("playerChannelId")] for r in spec.iter("PlayerChannelRef")
                if r.get("playerChannelId").endswith("_x")]
        specs.append((int(spec.get("startFrame")), int(spec.get("endFrame")),
                      np.array([col[p] for p in refs])))
    with open(base + "tracking.txt") as fh:
        rows = fh.read().splitlines()
    players = np.full((len(rows), len(ids), 2), np.nan)
    ball = np.full((len(rows), 2), np.nan)
    for line in rows:
        f, p, b = line.split(":")
        fi = int(f)
        idx = next(s[2] for s in specs if s[0] <= fi <= s[1])
        vals = np.array(p.replace(";", ",").split(","), dtype=float).reshape(len(idx), -1)
        players[fi - 1, idx] = vals[:, :2]
        bb = b.split(",")
        if bb[0][:1] != "N":
            ball[fi - 1] = (float(bb[0]), float(bb[1]))
    team = np.array([side[p] for p in ids], dtype=np.int8)
    return players, ball, team, _events_json(base + "events.json")


def fill_dead_ball(ball):
    """Rebuild a ball position wherever the files have none.

    In these files the ball is hardly recorded while play is stopped. An automatic tracker does
    often see it (on the touchline, placed for the free kick...), so a model trained on the raw
    files would learn "no ball = dead ball", a shortcut that does not exist in the challenge
    data. The rebuilt ball stays where it went out, moves to the restart spot and waits there.
    bench/train_model.py removes it again in part of the stoppages, so that both cases are seen.
    """
    out = ball.copy()
    nan = np.isnan(ball[:, 0])
    pad = np.diff(np.concatenate(([0], nan.astype(int), [0])))
    for s, e in zip(np.flatnonzero(pad == 1), np.flatnonzero(pad == -1)):
        if s == 0 or e >= len(ball):
            continue
        n = e - s
        t = np.clip((np.arange(n) / max(n - 1, 1) - 0.3) / 0.5, 0, 1)
        out[s:e] = ball[s - 1] + t[:, None] * (ball[e] - ball[s - 1])
    return out


def available(folder=None):
    folder = folder or METRICA_DIR
    return [g for g in GAMES if os.path.isdir(os.path.join(folder, "data",
                                                           "Sample_Game_{}".format(g)))]


def load(game, use_cache=True):
    """One match as a dict (see the module docstring). "ball_raw" keeps the ball as recorded
    (NaN during most stoppages); "ball" has those holes rebuilt by fill_dead_ball()."""
    cache = os.path.join(METRICA_DIR, "cache", "game{}.npz".format(game))
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        m = {k: z[k] for k in ("players", "team", "ball", "ball_raw", "ball_z", "labels")}
        m["halves"] = [tuple(h) for h in z["halves"].tolist()]
        m["pitch"] = tuple(z["pitch"].tolist())
        return m
    players, ball, team, events = (_read_csv_game if game in (1, 2) else _read_epts_game)(game)
    scale = np.array(PITCH)
    labels = labels_from_events(events, len(ball))
    graded = np.diff(np.concatenate(([0], (labels != "X").astype(int), [0])))
    halves = [(int(s), int(e) - 1)
              for s, e in zip(np.flatnonzero(graded == 1), np.flatnonzero(graded == -1))]
    raw = (ball * scale).astype(np.float32)
    m = {"players": (players * scale).astype(np.float32), "team": team,
         "ball": fill_dead_ball(raw), "ball_raw": raw,
         "ball_z": np.full(len(ball), np.nan, dtype=np.float32),
         "labels": labels, "halves": halves, "pitch": PITCH}
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, halves=np.array(halves), pitch=np.array(PITCH),
                            **{k: m[k] for k in ("players", "team", "ball", "ball_raw",
                                                 "ball_z", "labels")})
    return m


def summary(game):
    m = load(game)
    lab = m["labels"]
    graded = lab != "X"
    spells = []
    for a, b in m["halves"]:
        seg = lab[a:b + 1]
        ch = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        st, en = np.concatenate(([0], ch)), np.concatenate((ch, [len(seg)]))
        spells += [(e - s) / FPS for s, e in zip(st, en) if seg[s] in "HA"]
    minutes = graded.sum() / FPS / 60
    print("game {}: {:.0f} min, H {:.1%} A {:.1%} D {:.1%}, {} possessions ({:.1f}/min), "
          "mean {:.1f} s, median {:.1f} s".format(
              game, minutes, np.mean(lab[graded] == "H"), np.mean(lab[graded] == "A"),
              np.mean(lab[graded] == "D"), len(spells), len(spells) / minutes,
              np.mean(spells), np.median(spells)))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--download", action="store_true")
    if ap.parse_args().download and not os.path.isdir(os.path.join(METRICA_DIR, "data")):
        os.makedirs(os.path.dirname(METRICA_DIR), exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", URL, METRICA_DIR], check=True)
    if not available():
        raise SystemExit("No matches in {} (run: python -m bench.metrica --download)."
                         .format(METRICA_DIR))
    for g in available():
        summary(g)
