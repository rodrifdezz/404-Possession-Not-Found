"""Lectura del tracking (ATD) y del Smart Tagging a arrays de numpy.

El ATD se lee una vez (~1-2 min) y se guarda en cache/atd.npz; las siguientes veces carga en
segundos. cache/ está en .gitignore: son datos derivados del partido y NO se pueden publicar.

Formato de cada línea del ATD:  <frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<x,y,...balón>
frame_id empieza en 1; el fotograma de vídeo (fila del CSV) es frame_id - 1.
"""

import os
import xml.etree.ElementTree as ET

import numpy as np

from . import config as C


def track_teams(meta_path):
    """+1 si el slot es del local (primer equipo del XML), -1 si es visitante."""
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
            k = len(vals) // n_slots          # campos por jugador (x, y, velocidad...)
            if k >= 2 and k * n_slots == len(vals):
                players[f] = vals.reshape(n_slots, k)[:, :2]
            b = parts[2].split(",")
            if len(b) >= 2 and b[0][:1] != "N":
                ball[f] = (float(b[0]), float(b[1]))
    # normalizado 0-1 -> metros
    scale = np.array(C.PITCH, dtype=np.float32)
    return players * scale, ball * scale


def load_atd(use_cache=True):
    """dict con players (N,46,2) m, ball (N,2) m, team (46,) +1/-1. NaN = no visto."""
    cache = os.path.join(C.CACHE_DIR, "atd.npz")
    meta, raw = C.asset("atd_meta"), C.asset("atd_raw")
    for p in (meta, raw):
        if not os.path.isfile(p):
            raise FileNotFoundError("No encuentro {}. Revisa MTAH_DATA_DIR / MTAH_REPO.".format(p))
    if use_cache and os.path.isfile(cache) and os.path.getmtime(cache) > os.path.getmtime(raw):
        z = np.load(cache)
        return {k: z[k] for k in z.files}
    team = track_teams(meta)
    print("leyendo {} (solo la primera vez, 1-2 min)...".format(os.path.basename(raw)))
    players, ball = _parse_raw(raw, len(team))
    os.makedirs(C.CACHE_DIR, exist_ok=True)
    np.savez(cache, players=players, ball=ball, team=team)
    return {"players": players, "ball": ball, "team": team}


def load_smart_tagging():
    """Lista de dicts {code, start, end (fotogramas), tags}. Vacía si no hay fichero."""
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
