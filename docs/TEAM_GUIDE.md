# Team guide (internal)

## Setup (once)
1. Have the official repo at `~/hackathon/more-than-a-hack-2027` with the data in `data/tactical/`.
2. Clone this repo next to it: `cd ~/hackathon && git clone <this-repo-url>`.
3. Inside this repo: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
4. `python -m src.make_submission` → must end with `Valid.`
5. `python -m src.evaluate` → table with our score on the public 5 minutes.

## Workflow
```bash
git checkout main && git pull                 # 1. get the latest
git checkout -b name/task                     # 2. your own branch
# ... change code, test with make_submission ...
git add . && git commit -m "what I changed"   # 3. save
git push -u origin name/task                  # 4. push
# 5. on GitHub: Compare & pull request -> someone else reviews -> Merge
```

## Rules
- Never work on `main` directly. `main` always produces a valid CSV.
- `git status` before every commit: nothing from `data/`, `cache/`, `fake_data/`, `outputs/`, `external/`.
- Real data NEVER goes to an LLM (ChatGPT, Claude, Gemini...). To ask for help: code + fake data (`tools/make_fake_data.py`).
- The public 5 minutes are only for checking that things work, NOT for choosing parameters or
  versions (challenge rule, clarified by email on 3 Nov).
- Every parameter or rule change: log in `docs/JOURNAL.md` what, why and the effect on the full
  match (number of possessions, % D). It feeds the technical document.

## Sanity checks (full match)
`make_submission` prints: % of H/A/D, number of possessions, possessions/min and mean duration.
Rough reference for a real match: ~2-3 possessions/min, mean duration ~10-15 s.
D ~45-50 % of graded frames: in LaLiga the effective playing time is around 50-55 min out of ~106
graded (the 7 external DFL matches have 41-52 % D). (This used to say ~20 %, which is what the
public 5 minutes show: a stretch with few stoppages.)
These figures only catch gross errors. Parameters are chosen on the external bench
(`python -m bench.run`), never to make the challenge match "fit" them.

## External bench (choosing parameters without touching the challenge match)
1. `python -m bench.download`: fetches the 7 IDSSE/DFL matches (~2.6 GB, md5-checked) into
   `external/idsse/` (gitignored). With Avast or another antivirus that intercepts HTTPS:
   `SSL_CERT_FILE="C:/ProgramData/Avast Software/Avast/wscert.pem" python -m bench.download`.
2. `python -m bench.run --quick` (check, ~5 min), then `python -m bench.run` (~20 min).
   `python -m bench.run --report-only` rewrites the report from the saved raw results.
3. Result in `docs/BENCH.md`: the leave-one-match-out table and the final parameters.
4. Copy the final parameters into `src/config.py`, regenerate the submission and log it in the JOURNAL.
