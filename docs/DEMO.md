# Demo: who has the ball?

A self-contained web page that shows the possession model working, frame by frame, and that can be
recorded and published. It is generated from the repository, so anyone can reproduce it.

```bash
python -m bench.metrica --download     # once: the open Metrica sample matches
python -m src.make_submission          # optional: enables scene 2 (needs the challenge data)
python -m bench.demo                   # -> outputs/demo/possession_demo.html
```

The first run fits a model for the demo (about 10 minutes); later runs reuse it from
`outputs/demo/`. Open the page in any browser (no server, no internet needed). Options:
`--game 1|2|3` (replayed match), `--start` (match minute from the kick-off; past the first half it
continues in the second), `--minutes` (length of the window).

## What it shows

**Scene 1 · live replay on a public match.** Metrica Sports open sample data, game 2 by default.

* **Pitch.** Players and ball as the model sees them: the tracking is degraded exactly as in the
  external bench (ball lost in bursts, noise, players dropping out). When the ball is not
  tracked, its last known position is shown faded.
* **Probabilities.** P(home), P(away) and P(dead ball) for the last 30 s, with the detected
  touches.
* **Timeline.** It fills in as the match plays: the truth, our model and the official baseline
  (nearest player plus a ±2 s majority vote).
* **Running metrics.** spellF1 and macroF1 on the elapsed time, computed every 5 s with the
  official scorer.

The model is the v4 recipe (frame and touch features, two layers, decoder costs of v4) fitted
**only on the other two Metrica matches**, so the replayed match is unseen. The submitted model
was fitted on all ten external matches, this one included, so it is not used here: its numbers
on this match would be optimistic.

**Scene 2 · the challenge match, public window.** It shows only labels:

* the public ground truth of the first 5 minutes;
* our submission file;
* the official baselines;
* the official scores of each.

No tracking, no video and nothing else from the licensed match data reaches the page. This scene
appears only if `outputs/<team>.csv` exists.




