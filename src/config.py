"""Rutas, constantes del partido y parámetros del algoritmo. Todo lo configurable vive aquí.

Rutas (se pueden cambiar con variables de entorno, sin tocar código):
  MTAH_REPO      carpeta del repo oficial de Metrica      (por defecto ../more-than-a-hack-2027)
  MTAH_DATA_DIR  carpeta con tactical/ dentro             (por defecto <MTAH_REPO>/data)
"""

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

OFFICIAL_REPO = os.path.abspath(
    os.environ.get("MTAH_REPO") or os.path.join(ROOT, "..", "more-than-a-hack-2027"))
DATA_DIR = os.path.abspath(
    os.environ.get("MTAH_DATA_DIR") or os.path.join(OFFICIAL_REPO, "data"))

OUT_DIR = os.path.join(ROOT, "outputs")   # gitignored
CACHE_DIR = os.path.join(ROOT, "cache")   # gitignored: contiene datos derivados, NUNCA se sube

# Nombre del equipo: da nombre al CSV y al documento que se entregan.
TEAM_NAME = "404-possession-not-found"

# ---------------------------------------------------------------- partido (de DATA.md)
STEM = "BAR-ATH_tactical"
N_FRAMES = 199319
FPS = 25
PITCH = (105.3, 68.0)                         # metros (largo, ancho)
HALVES = [(11185, 90651), (110808, 190197)]   # fotogramas puntuados, inclusivos


def asset(name):
    suffix = {"atd_meta": "_atd.xml", "atd_raw": "_atd.txt",
              "smart_tagging": "_smart_tagging.xml", "homography": "_homography.csv"}[name]
    return os.path.join(DATA_DIR, "tactical", STEM + suffix)


# ---------------------------------------------------------------- parámetros v0
# Elegidos por razonamiento físico, NO ajustados contra la ground truth pública (está prohibido).
# Cada cambio debe justificarse y anotarse en docs/ para el documento técnico.
PARAMS = {
    # limpieza del balón
    "ball_max_gap_s": 0.6,          # huecos más cortos se interpolan linealmente
    "ball_speed_smooth": 5,         # frames de mediana móvil para la velocidad del balón
    # evidencia de equipo
    "control_radius_m": 2.0,        # balón "pegado" a un jugador
    "near_radius_m": 4.0,           # balón cerca, evidencia débil
    "team_gain_per_m": 1.0,         # log-ratio por metro de ventaja del más cercano
    "team_llr_clip": 3.0,
    "ball_flight_speed": 15.0,      # m/s: por encima, balón en vuelo -> evidencia atenuada
    "flight_factor": 0.2,
    "smart_tag_team_llr": 0.3,      # empujoncito por fase etiquetada con equipo
    # balón muerto
    "dead_prior_logit": -1.5,
    "out_margin_m": 0.5,            # balón fuera del campo por más de esto
    "slow_players_mps": 1.0,        # velocidad media de jugadores "todos andando"
    "fast_players_mps": 2.0,
    "still_ball_mps": 0.5,
    "still_ball_s": 2.0,
    "long_gap_s": 2.0,
    "set_piece_tail_s": 1.5,        # el tag de saque acaba ~1.5 s después del saque real
    # modelo temporal (en nats; coste de cambiar de estado)
    "switch_cost": 8.0,             # H <-> A
    "dead_cost": 6.0,               # vivo <-> D
}

# Fases del Smart Tagging en las que el 'Team' es el equipo CON balón (hipótesis a verificar
# visualmente en el vídeo; las de "DEFENDING ..." se dejan fuera hasta comprobar su semántica).
ATTACKING_CODES = {"BUILD UP", "PROGRESSION", "BALL IN FINAL THIRD", "CREATING CHANCES",
                   "ATTACKING TRANSITION", "LONG BALL"}
TEAM_TAG = {"barcelona": +1, "atletic club": -1, "athletic club": -1}
