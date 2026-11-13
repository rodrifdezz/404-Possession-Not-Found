"""Read the tracking (ATD) and the Smart Tagging into numpy arrays.

The ATD is parsed once (~1-2 min) and stored in cache/atd_<fingerprint>.npz; later runs load it
in seconds. cache/ is in .gitignore: it is data derived from the match and must NOT be published.

Format of each ATD line:  <frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<x,y,...ball>
frame_id starts at 1; the video frame (CSV row) is frame_id - 1.
"""

import hashlib
import os
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C


def track_teams(meta_path):
    """+1 if the slot belongs to the home team (first team in the XML), -1 if away."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    home = team_ids[0]
    return np.array([1 if p.get("teamId") == home else -1 for p in root.iter("Player")],
                    dtype=np.int8)


def _parse_raw(raw_path, n_slots):
    players = np.full((C.N_FRAMES, n_slots, 2), np.nan, dtype=np.float32)
    ball = np.full((C.N_FRAMES, 2), np.nan, dtype=np.float32)
    with open(raw_path) as handle:
        for line in handle:
            parts = line.rstrip("\n").split(":")
            if len(parts) < 3:
                continue
            f = int(parts[0]) - 1
            if not 0 <= f < C.N_FRAMES:
                continue
            vals = np.array(parts[1].replace(";", ",").split(","), dtype=np.float64)
            k = len(vals) // n_slots          # fields per player (x, y, speed...)
            if k >= 2 and k * n_slots == len(vals):
                players[f] = vals.reshape(n_slots, k)[:, :2]
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N":
                ball[f] = (float(b[0]), float(b[1]))
    # normalised 0-1 -> metres
    scale = np.array(C.PITCH, dtype=np.float32)
    return players * scale, ball * scale


def _cache_path(*sources):
    """cache/atd_<fingerprint>.npz: the fingerprint changes with each file's path, size or
    modification time, so another dataset (e.g. fake_data/) or an edited file never reuses a
    cache that is not its own."""
    key = hashlib.sha1()
    for p in sources:
        st = os.stat(p)
        key.update("{}|{}|{}\n".format(os.path.abspath(p), st.st_size, st.st_mtime_ns).encode())
    return os.path.join(C.CACHE_DIR, "atd_{}.npz".format(key.hexdigest()[:16]))


def load_atd(use_cache=True):
    """dict with players (N,46,2) m, ball (N,2) m, team (46,) +1/-1. NaN = not seen."""
    meta, raw = C.asset("atd_meta"), C.asset("atd_raw")
    for p in (meta, raw):
        if not os.path.isfile(p):
            raise FileNotFoundError("{} not found. Check MTAH_DATA_DIR / MTAH_REPO.".format(p))
    cache = _cache_path(meta, raw)
    if use_cache and os.path.isfile(cache):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    team = track_teams(meta)
    print("reading {} (first time only, 1-2 min)...".format(os.path.basename(raw)))
    players, ball = _parse_raw(raw, len(team))
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez(cache, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


def load_smart_tagging():
    """List of dicts {code, start, end (frames), tags}. Empty if there is no file."""
    path = C.asset("smart_tagging")
    if not os.path.isfile(path):
        return []
    out = []
    for inst in ET.parse(path).iter("instance"):
        out.append({
            "code": (inst.findtext("code") or "").strip().upper(),
            "start": int(round(float(inst.findtext("start")) * C.FPS)),
            "end": int(round(float(inst.findtext("end")) * C.FPS)),
            "tags": {(l.findtext("group") or "").strip().lower(): (l.findtext("text") or "").strip()
                     for l in inst.findall("label")},
        })
    return out
