"""Fits the submitted v4 model (V4-C) on the 10 external matches -> models/possession_v4c.pkl.

    python -m bench.v4_final        # needs the feature cache of bench/v4.py (python -m bench.v4 --features)

The recipe is exactly the one validated in docs/BENCH_V4.md and docs/BENCH_V4_FIX.md: V4-C of
bench/v4.py (layer 1 with the touch features of src/touch_features.py, layer 2 of src/stack.py on
out-of-sample layer-1 probabilities of 3 groups of whole matches, both symmetric), fitted on the
3 Metrica and 7 IDSSE/DFL matches. The decoder costs (switch 9, dead 48: the median of the five
nested inner choices of bench/v4_fix.py) are in config.MODELS["v4c"]. No data of the challenge
match is used. The submission itself never trains: python -m src.make_submission loads the
committed model.
"""

import hashlib
import json
import pickle
import time

from src import config as C
from src import model as M

from . import train_model, v4

MODEL_PATH = M.model_path("v4c")


def _md5(path):
    with open(path, "rb") as fh:
        return hashlib.md5(fh.read()).hexdigest()


def train():
    import sklearn
    t0 = time.time()
    fitted = v4.fit_v4(v4.ALL, touch=True, layer2=(False,))
    bundle = {"stage1": fitted["stage1"], "stage2": fitted["stage2"][False],
              "names": fitted["names"], "train": fitted["train"],
              "costs": dict(C.MODELS["v4c"]["costs"])}
    with open(MODEL_PATH, "wb") as fh:
        pickle.dump(bundle, fh, protocol=4)
    info = {"model": "v4 (V4-C): 2 x HistGradientBoostingClassifier (stacked), with touch features",
            "recipe": "bench/v4.py fit_v4(touch=True), layer 2 without extras",
            "train_matches": fitted["train"], "training_degradations": v4.TRAIN_TAGS,
            "step": v4.STEP_V4, "inner_groups": v4.INNER, "stage1": train_model.HGB1,
            "stage2": train_model.HGB2, "decoder_costs": C.MODELS["v4c"]["costs"],
            "features": fitted["names"], "sklearn": sklearn.__version__,
            "pkl_md5": _md5(MODEL_PATH)}
    with open(MODEL_PATH.replace(".pkl", ".json"), "w") as fh:
        json.dump(info, fh, indent=1)
    v4.log("v4 model fitted on {} matches ({:.0f} s) -> {}".format(
        len(fitted["train"]), time.time() - t0, MODEL_PATH))


if __name__ == "__main__":
    train()
