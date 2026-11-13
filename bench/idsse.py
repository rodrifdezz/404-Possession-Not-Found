"""Reader for the 7 open IDSSE / DFL Bundesliga matches (Bassek et al., Sci Data 2025).

Why this data: the challenge ground truth is derived mechanically from two flags of the corrected
tracking, "ball dead" and "team owning the ball" (CHALLENGE.md). DFL tracking publishes exactly
those two flags per frame (BallStatus, BallPossession), at 25 fps, under CC BY 4.0. These are
OTHER matches: nothing from the challenge match goes through here.

Download: `python -m bench.download` (https://doi.org/10.6084/m9.figshare.28196177).
Two XML files per match inside external/idsse/ (the folder is in .gitignore):
    ..._matchinformation_..._<ID>.xml        metadata (teams, pitch)
    ..._positions_raw_observed_..._<ID>.xml  tracking
A file only needs the match ID and "matchinformation" or "positions" in its name.

Reading a match takes ~20 s the first time; afterwards it loads from external/idsse/cache/.
"""

import glob
import os
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IDSSE_DIR = os.environ.get("IDSSE_DIR") or os.path.join(ROOT, "external", "idsse")
MATCHES = ["J03WMX", "J03WN1", "J03WOH", "J03WOY", "J03WPY", "J03WQQ", "J03WR9"]
SECTIONS = ["firstHalf", "secondHalf"]    # none of these matches has extra time
SEPARATOR = 50                            # 'X' frames between halves (the scorer never joins them)


def find_files(match_id, folder=None):
    """(metadata, positions) of the match, or None if either is missing."""
    folder = folder or IDSSE_DIR
    xmls = [p for p in glob.glob(os.path.join(folder, "*.xml")) if match_id in os.path.basename(p)]
    meta = [p for p in xmls if "matchinformation" in os.path.basename(p).lower()]
    pos = [p for p in xmls if "position" in os.path.basename(p).lower()]
    return (meta[0], pos[0]) if meta and pos else None


def read_meta(path):
    """{"home": TeamId, "away": TeamId, "pitch": (length, width)}."""
    root = ET.parse(path).getroot()
    out = {}
    for team in root.iter("Team"):
        role = team.get("Role", "").lower()
        if role == "home":
            out["home"] = team.get("TeamId")
        elif role in ("guest", "away"):
            out["away"] = team.get("TeamId")
    env = next(root.iter("Environment"))
    out["pitch"] = (float(env.get("PitchX")), float(env.get("PitchY")))
    return out


def read_positions(path, meta):
    """Single pass over the XML. Returns, per half, the frames of every object."""
    teams = {meta["home"]: 1, meta["away"]: -1}
    sections = {s: {"ball": {}, "players": {}} for s in SECTIONS}
    current = None
    for event, elem in ET.iterparse(path, events=("start", "end")):
        if event == "start" and elem.tag == "FrameSet":
            sec, team_id = elem.get("GameSection"), elem.get("TeamId")
            if sec not in sections:
                current = None
            elif team_id and team_id.upper() == "BALL":
                current = ("ball", sec, None)
            elif team_id in teams:
                current = ("player", sec, (elem.get("PersonId"), teams[team_id]))
            else:
                current = None                      # referees and others
        elif event == "end" and elem.tag == "Frame":
            if current is not None:
                kind, sec, who = current
                n = int(elem.get("N"))
                x, y = float(elem.get("X")), float(elem.get("Y"))
                if kind == "ball":
                    sections[sec]["ball"][n] = (x, y, float(elem.get("Z", "nan")),
                                                int(elem.get("BallStatus", "0")),
                                                int(elem.get("BallPossession", "0")))
                else:
                    sections[sec]["players"].setdefault(who, {})[n] = (x, y)
            elem.clear()
        elif event == "end" and elem.tag == "FrameSet":
            current = None
            elem.clear()
    return sections


def build(meta, sections):
    """Arrays in the same format as load.load_atd() of the challenge, plus the DFL labels."""
    L, W = meta["pitch"]
    slots = sorted({who for s in SECTIONS for who in sections[s]["players"]})
    team = np.array([sign for _, sign in slots], dtype=np.int8)
    index = {who: i for i, who in enumerate(slots)}

    blocks, halves, start = [], [], 0
    for s in SECTIONS:
        ball = sections[s]["ball"]
        if not ball:
            continue
        n0, n1 = min(ball), max(ball)
        n = n1 - n0 + 1
        b = np.full((n, 5), np.nan)
        for k, v in ball.items():
            b[k - n0] = v
        p = np.full((n, len(slots), 2), np.nan, dtype=np.float32)
        for who, frames in sections[s]["players"].items():
            ks = np.fromiter(frames.keys(), dtype=np.int64) - n0
            xy = np.array(list(frames.values()), dtype=np.float32)
            ok = (ks >= 0) & (ks < n)
            p[ks[ok], index[who]] = xy[ok]
        if blocks:
            blocks.append((np.full((SEPARATOR, 5), np.nan),
                           np.full((SEPARATOR, len(slots), 2), np.nan, dtype=np.float32)))
            start += SEPARATOR
        blocks.append((b, p))
        halves.append((start, start + n - 1))
        start += n

    b = np.concatenate([x for x, _ in blocks])
    players = np.concatenate([y for _, y in blocks])
    # DFL: origin at the centre spot -> origin at a corner, like the challenge ATD
    shift = np.array([L / 2, W / 2], dtype=np.float32)
    players = players + shift
    ball_xy = (b[:, :2] + shift).astype(np.float32)

    labels = np.full(len(b), "X", dtype="<U1")
    status, owner = b[:, 3], b[:, 4]
    labels[status == 0] = "D"
    labels[(status == 1) & (owner == 1)] = "H"
    labels[(status == 1) & (owner == 2)] = "A"
    return {"players": players, "team": team, "ball": ball_xy, "ball_z": b[:, 2].astype(np.float32),
            "labels": labels, "halves": halves, "pitch": (L, W)}


def load(match_id, folder=None, use_cache=True):
    """One match as a dict (see build). None if its XML files have not been downloaded.

    use_cache=False neither reads nor writes external/idsse/cache/.
    """
    folder = folder or IDSSE_DIR
    files = find_files(match_id, folder)
    if files is None:
        return None
    cache = os.path.join(folder, "cache", match_id + ".npz")
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > max(
            os.path.getmtime(f) for f in files):
        z = np.load(cache)
        return {"players": z["players"], "team": z["team"], "ball": z["ball"],
                "ball_z": z["ball_z"], "labels": z["labels"],
                "halves": [tuple(h) for h in z["halves"].tolist()], "pitch": tuple(z["pitch"])}
    meta = read_meta(files[0])
    m = build(meta, read_positions(files[1], meta))
    if use_cache:
        os.makedirs(os.path.dirname(cache), exist_ok=True)
        np.savez_compressed(cache, players=m["players"], team=m["team"], ball=m["ball"],
                            ball_z=m["ball_z"], labels=m["labels"], halves=np.array(m["halves"]),
                            pitch=np.array(m["pitch"]))
    return m


def available(folder=None):
    return [m for m in MATCHES if find_files(m, folder)]
