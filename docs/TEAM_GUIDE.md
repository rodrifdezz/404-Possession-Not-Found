# Team guide (internal)

## Setup (once)
1. Have the official repo at `~/hackathon/more-than-a-hack-2027` with the data in `data/tactical/`.
2. Clone this repo next to it: `cd ~/hackathon && git clone <this-repo-url>`.
3. Inside this repo: `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`.
4. `python -m src.make_submission` → must end with `Valid.` and an md5. **The md5 must be the same
   on every machine of the team** (challenge rule: the repository regenerates the same file).
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
- Python 3.9-3.12 and the exact versions of `requirements.txt` (`pip install -r requirements.txt`
  again after pulling, if the file changed).
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

## Learned evidence (v3, `EVIDENCE = "model"` in `src/config.py`)
- The model (`models/possession_hgb.pkl`) is committed: `make_submission` never trains.
- `python -m src.make_submission --evidence rules --name rules_v2` regenerates the v2 file, to
  compare summaries. Which system is submitted is decided on external matches only
  (`docs/BENCH_MODEL_CV.md`, `docs/BENCH_MODEL.md`), never on the public 5 minutes.
- Check on unseen leagues: `python -m bench.run --evidence model` (~30 min, needs the IDSSE
  matches) scores the stored model on the 7 DFL matches next to the rules -> `docs/BENCH_MODEL.md`.
- Refit (only if `src/frame_features.py` or `src/stack.py` change):
  `python -m bench.metrica --download`, `python -m bench.train_model --cv` (~50 min, writes
  `docs/BENCH_MODEL_CV.md`), copy the costs into `MODEL_COSTS`, `python -m bench.train_model`
  (~25 min) and commit the new `models/` files.

## Submission checklist (deadline 17 Nov 2026)
1. `git pull` on `main`, `pip install -r requirements.txt`, `python -m unittest discover -s tests`.
2. `python -m src.make_submission` on **two machines** (ideally macOS and Windows): both end with
   `Valid.` and print the **same md5**. Log the summary and the md5 in `docs/JOURNAL.md`.
3. Sanity of the summary (gross errors only): 2-5 possessions per minute, mean duration 8-20 s,
   D between 35 % and 55 %, neither team above ~70 % of the live time.
4. `git status` clean, nothing from `data/`, `cache/`, `outputs/`, `external/` committed; the
   repository is **public**.
5. Google Form (one response per team, editable until the deadline):
   - team name exactly as in the registration form;
   - `outputs/404-possession-not-found.csv`, renamed to `<team name>.csv` if the registered name
     differs;
   - repository URL;
   - the technical document (`docs/TECHNICAL.md` exported to PDF), named `<team name>.pdf`.
6. Upload the file generated in step 2, not an older one: validate the exact file you upload
   (`python ../more-than-a-hack-2027/evaluation/validate_submission.py <file>`).
