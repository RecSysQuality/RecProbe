"""
Reviews4Rec integration (Sachdeva and McAuley, 2020) for review-aware models.

Reviews4Rec is a research code base run from its own scripts, not a Python
package, so the integration is an adapter driven by evaluation.yaml:

    reviews4rec:
      repo_path: /path/to/reviews4rec
      command: "python main.py --dataset {data_dir} --model {model} --seed {seed} {params}"
      models:
        - name: NARRE
          params: {latent_dim: 32}

For every run the adapter writes train/validation/test (user_id, item_id,
rating, review_text, timestamp) as TSV files into {data_dir}, runs ``command``
inside ``repo_path`` and reads back from {out_dir}:

  * metrics.json     (optional) metrics computed by Reviews4Rec, e.g. {"MSE": 1.02}
  * predictions.tsv  rating predictions for the test pairs: user_id, item_id, prediction
  * ranked.tsv       (optional) top-K lists: user_id, item_id, rank, score

Placeholders available in ``command``: {data_dir}, {out_dir}, {model}, {seed},
{params} (rendered as --key value). Adapt ``command`` to the version of
Reviews4Rec in use, or write a small wrapper script that honours this contract.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess

import pandas as pd

from baselines.base_runner import BaseRunner, RunOutput
from utils import ITEM, RATING, USER


class Reviews4RecRunner(BaseRunner):
    framework = "reviews4rec"

    def run(self, model_cfg, train, val, test, seed):
        repo = self.fw_cfg.get("repo_path")
        command = self.fw_cfg.get("command")
        if not repo or not command:
            raise ValueError("reviews4rec: set 'repo_path' and 'command' in evaluation.yaml")
        name = model_cfg["name"]
        run_dir = os.path.join(self.work_dir, "reviews4rec", f"{name}_s{seed}")
        data_dir, out_dir = os.path.join(run_dir, "data"), os.path.join(run_dir, "out")
        os.makedirs(data_dir, exist_ok=True)
        os.makedirs(out_dir, exist_ok=True)
        for split, df in (("train", train), ("validation", val), ("test", test)):
            if df is not None:
                df.to_csv(os.path.join(data_dir, f"{split}.tsv"), sep="\t", index=False)
        params = {**(self.fw_cfg.get("params") or {}), **(model_cfg.get("params") or {})}
        rendered = " ".join(f"--{k} {shlex.quote(str(v))}" for k, v in params.items())
        cmd = command.format(data_dir=data_dir, out_dir=out_dir, model=name, seed=seed, params=rendered)
        self.logger.info(f"reviews4rec: {cmd}")
        proc = subprocess.run(cmd, shell=True, cwd=repo, capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"Reviews4Rec failed ({proc.returncode}):\n{proc.stderr[-2000:]}")

        lib = {}
        mpath = os.path.join(out_dir, "metrics.json")
        if os.path.exists(mpath):
            with open(mpath) as fh:
                lib = {k: float(v) for k, v in json.load(fh).items()}
        preds = None
        ppath = os.path.join(out_dir, "predictions.tsv")
        if os.path.exists(ppath):
            p = pd.read_csv(ppath, sep="\t", dtype={USER: str, ITEM: str})
            preds = test[[USER, ITEM, RATING]].merge(p[[USER, ITEM, "prediction"]], on=[USER, ITEM])
        ranked = None
        rpath = os.path.join(out_dir, "ranked.tsv")
        if os.path.exists(rpath):
            ranked = pd.read_csv(rpath, sep="\t", dtype={USER: str, ITEM: str})
        return RunOutput(lib, ranked, None, preds)
