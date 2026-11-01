# Diario de experimentos

Una entrada por cambio. Fecha · quién · qué · por qué · efecto en el partido completo
(posesiones, duración media, % D) · decisión (se queda / se descarta).

## 2026-10-25 · v0
- Esqueleto: limpieza de balón + balón muerto por reglas + equipo por distancia + Viterbi 3 estados.
- Parámetros iniciales por razonamiento físico (ver `src/config.py`).
- Pendiente de anotar: resultado en el partido real (resumen de `make_submission`).

## 2026-11-01 · JAN · v1 (correcciones de la auditoría)
- Fuera `KNOWN_DEAD` (tramo de fotogramas fijo antes del saque de la 2ª parte): las reglas prohíben
  constantes atadas a fotogramas concretos del partido.
- `hmm.emissions` normalizada: p(D) = sigmoid(dead_logit), p(H) = p(vivo)·sigmoid(team_llr),
  p(A) = p(vivo)·sigmoid(-team_llr). En v0 la evidencia de equipo sumaba hasta +1.5 nats a H/A
  frente a D y tapaba paradas con un jugador cerca del balón. H − A sigue siendo team_llr.
- `make_submission` falla si el validador oficial rechaza el CSV.
- Umbrales, costes y arquitectura sin tocar.
- Efecto en el partido completo: D 32.3% → 47.4%; posesiones 571 → 516, duración media 7.5 → 6.5 s.
  Tramo antes del saque de la 2ª parte: 99.9% D con KNOWN_DEAD; sin él, 34% con emisiones v0 y
  63% con las nuevas.
- Comprobación de cordura en los 5 min públicos (no se ha ajustado nada con ellos):
  macro F1 0.7047 → 0.7814 (D 0.55 → 0.74); spell F1 0.2824 → 0.2588 (dentro del ruido ~0.04).
- Decisión: se queda (corrige un error del modelo y una infracción de las reglas). Vigilar el
  47% de D en el partido completo: parece alto.
