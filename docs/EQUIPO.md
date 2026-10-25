# Guía del equipo (interna)

## Puesta en marcha (una vez)
1. Tener el repo oficial en `~/hackathon/more-than-a-hack-2027` con los datos en `data/tactical/`.
2. Clonar este repo al lado: `cd ~/hackathon && git clone <url-de-este-repo>`.
3. Dentro de este repo: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
4. `python -m src.make_submission` → debe terminar en `Valid.`
5. `python -m src.evaluate` → tabla con nuestra puntuación en los 5 minutos públicos.

## Ciclo de trabajo
```bash
git checkout main && git pull                 # 1. traer lo último
git checkout -b nombre/tarea                  # 2. rama propia
# ... cambiar código, probar con make_submission ...
git add . && git commit -m "qué he cambiado"  # 3. guardar
git push -u origin nombre/tarea               # 4. subir
# 5. en GitHub: Compare & pull request -> otro revisa -> Merge
```

## Reglas
- Nunca trabajar en `main` directamente. `main` siempre genera un CSV válido.
- `git status` antes de cada commit: nada de `data/`, `cache/`, `fake_data/`, `outputs/`.
- Datos reales NUNCA a un LLM (ChatGPT, Claude, Gemini...). Para pedir ayuda: código + datos falsos (`tools/make_fake_data.py`).
- Los 5 minutos públicos solo para comprobar que funciona, NO para elegir parámetros (regla del reto).
- Cada cambio de parámetro o regla: anotar en `docs/DIARIO.md` qué, por qué y efecto en el partido completo (nº posesiones, % D). Servirá para el documento técnico.

## Comprobaciones de sentido común (partido completo)
`make_submission` imprime: % de H/A/D, nº de posesiones, posesiones/min y duración media.
Referencia aproximada de un partido real: ~2-3 posesiones/min, duración media ~10-15 s, D ~20 %.
