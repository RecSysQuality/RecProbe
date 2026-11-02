"""
Evaluation phase (Section 4.1, phase 4).

Runs the configured models of the integrated library on the clean dataset
produced by data processing and on the perturbed dataset produced by the
injection, repeats every run over multiple random seeds (mean and standard
deviation), computes accuracy (by the library) and beyond-accuracy metrics (by
RecProbe, from the exported ranked lists) for the whole population and for each
user segment, and reports everything in a unified table.
"""

from __future__ import annotations

import glob
import os
from datetime import datetime

import numpy as np
import pandas as pd

from baselines.cornac.cornac_runner import CornacExperimentRunner
from baselines.custom.custom_runner import CustomExperimentRunner
from baselines.recbole.recbole_runner import RecBoleExperimentRunner
from baselines.reviews4rec.reviews4rec_runner import Reviews4RecRunner
from evaluation import segments as seg
from evaluation.metrics import accuracy_per_user, aggregate_users, beyond_accuracy, custom_metrics
from utils import ITEM, USER, set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RUNNERS = {
    "cornac": CornacExperimentRunner,
    "recbole": RecBoleExperimentRunner,
    "reviews4rec": Reviews4RecRunner,
    "custom": CustomExperimentRunner,
}

KEYS = ["dataset", "perturbation", "framework", "model", "segment", "metric", "source"]


class BaselinesOrchestrator:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config
        self.ev = config.evaluation
        self.framework = self.ev.framework
        if self.framework not in RUNNERS:
            raise ValueError(f"Unknown framework '{self.framework}'. Available: {sorted(RUNNERS)}")
        self.fw_cfg = self.ev.get(self.framework) or {}
        if not self.fw_cfg.get("models"):
            raise ValueError(f"No models configured under '{self.framework}' in evaluation.yaml")
        self.results_dir = self.ev.results_dir or os.path.join(BASE_DIR, "baselines", "results", config.dataset)
        os.makedirs(self.results_dir, exist_ok=True)
        self.runner = RUNNERS[self.framework](self.fw_cfg, self.ev, logger,
                                              os.path.join(self.results_dir, "_work"))

    # ================================================================ runs
    def evaluate(self, train, val, test, clean_train, condition: str, perturbation: str = "none",
                 fake_users: set | None = None) -> pd.DataFrame:
        """Train/evaluate every model x seed on one dataset version; returns per-seed long rows."""
        fake_users = set(fake_users or ())
        segments = seg.build_segments(clean_train, self.ev.user_segments, self.ev.user_segment_ids, fake_users)
        if condition == "clean":
            self.logger.info("User segments (computed on the clean training set):\n"
                             + seg.describe(segments, clean_train).to_string(index=False))
        # beyond-accuracy reference: clean training set (fixed across perturbed runs)
        catalog = clean_train[ITEM].unique()
        item_pop = clean_train.groupby(ITEM)[USER].nunique()
        n_users = clean_train[USER].nunique()
        test = test[~test[USER].isin(fake_users)]                 # fake users are never evaluated
        test_users = set(test[USER])

        rows = []
        for model_cfg in self.fw_cfg["models"]:
            label = model_cfg.get("label") or model_cfg["name"]
            for seed in self.ev.seeds:
                set_seed(int(seed), verbose=False)
                self.logger.info(f"[{condition}] {self.framework}/{label} seed={seed}")
                out = self.runner.run(model_cfg, train, val, test, int(seed))
                recprobe_acc = None
                if out.user_metrics is None:
                    metrics = self.fw_cfg.get("metrics") or [{"NDCG": {"k": self.ev.top_k}}]
                    recprobe_acc = accuracy_per_user(metrics, out.ranked, test, out.predictions,
                                                     self.fw_cfg.get("rating_threshold"), self.ev.top_k)
                for sname, susers in segments.items():
                    users = susers & test_users
                    base = {"dataset": self.config.dataset, "perturbation": perturbation, "condition": condition,
                            "framework": self.framework, "model": label, "seed": int(seed),
                            "segment": sname, "n_users": len(users)}
                    # accuracy, as computed by the library
                    if out.user_metrics is not None:
                        src = "recprobe" if self.framework == "custom" else "library"
                        acc = aggregate_users(out.user_metrics, users)
                    elif sname == "all" and not fake_users:
                        acc, src = out.library_metrics, "library"
                    else:
                        acc, src = aggregate_users(recprobe_acc, users), "recprobe"
                    rows += [{**base, "metric": m, "value": v, "source": src} for m, v in acc.items()]
                    if out.ranked is None:
                        continue
                    for k in self.ev.cutoffs:
                        ba = beyond_accuracy(out.ranked, users, int(k), catalog, item_pop, n_users,
                                             tuple(self.ev.beyond_accuracy))
                        rows += [{**base, "metric": m, "value": v, "source": "recprobe"} for m, v in ba.items()]
                    cm = custom_metrics(self.ev.get("custom_metrics") or [], out.ranked, test, train,
                                        users, self.ev.top_k)
                    rows += [{**base, "metric": m, "value": v, "source": "custom"} for m, v in cm.items()]
        return pd.DataFrame(rows)

    def run(self, clean: tuple, noisy: tuple | None = None, tag: str | None = None,
            fake_users: set | None = None) -> pd.DataFrame | None:
        """clean/noisy: (train, validation, test). Returns the clean-vs-noisy comparison."""
        train_c, val_c, test_c = clean
        cache = os.path.join(self.results_dir, f"clean_{self.config.evaluation_hash()}.tsv")
        if self.ev.reuse_clean_results and os.path.exists(cache):
            self.logger.info(f"Reusing clean results: {cache}")
            clean_raw = pd.read_csv(cache, sep="\t")
        else:
            clean_raw = self.evaluate(train_c, val_c, test_c, train_c, "clean")
            clean_raw.to_csv(cache, sep="\t", index=False)
        if noisy is None:
            summary = summarize(clean_raw)
            path = os.path.join(self.results_dir, "clean_summary.tsv")
            summary.to_csv(path, sep="\t", index=False)
            self.logger.info(f"Clean results: {path}")
            return summary

        train_n, val_n, test_n = noisy
        noisy_raw = self.evaluate(train_n, val_n if val_n is not None else val_c,
                                  test_n if test_n is not None else test_c, train_c, "noisy", tag, fake_users)
        out_dir = os.path.join(self.results_dir, tag)
        os.makedirs(out_dir, exist_ok=True)
        raw = pd.concat([clean_raw.assign(perturbation=tag), noisy_raw], ignore_index=True)
        raw.to_csv(os.path.join(out_dir, "raw.tsv"), sep="\t", index=False)
        comp = compare(summarize(raw))
        comp.to_csv(os.path.join(out_dir, "comparison.tsv"), sep="\t", index=False)
        self.logger.info(f"Comparison saved to {out_dir}/comparison.tsv")
        return comp


# ======================================================================= tables
def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    """Mean and standard deviation of each metric across seeds."""
    if raw.empty:
        return raw
    g = raw.groupby(KEYS + ["condition"], dropna=False)
    out = g["value"].agg(mean="mean", std=lambda s: s.std(ddof=1) if len(s) > 1 else 0.0, n_seeds="count")
    out["n_users"] = g["n_users"].max()
    return out.reset_index()


def compare(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, segment, metric) with clean and noisy values and the relative change."""
    wide = summary.pivot_table(index=KEYS, columns="condition", values=["mean", "std"], aggfunc="first")
    wide.columns = [f"{c}_{s}" for s, c in wide.columns]
    wide = wide.reset_index()
    for c in ("clean_mean", "noisy_mean", "clean_std", "noisy_std"):
        if c not in wide.columns:
            wide[c] = np.nan
    wide["delta"] = wide["noisy_mean"] - wide["clean_mean"]
    wide["delta_pct"] = np.where(wide["clean_mean"].abs() > 0, 100 * wide["delta"] / wide["clean_mean"].abs(), np.nan)
    users = summary.groupby(KEYS)["n_users"].max().reset_index()
    wide = wide.merge(users, on=KEYS, how="left")
    order = KEYS + ["n_users", "clean_mean", "clean_std", "noisy_mean", "noisy_std", "delta", "delta_pct"]
    return wide[order].sort_values(KEYS).reset_index(drop=True)


def create_unique_table(results_dir: str) -> pd.DataFrame | None:
    """Unified table of all models, datasets, segments and perturbations evaluated so far
    (``results_dir`` can be the results root, to merge several datasets)."""
    files = glob.glob(os.path.join(results_dir, "**", "comparison.tsv"), recursive=True)
    if not files:
        return None
    table = pd.concat([pd.read_csv(f, sep="\t") for f in files], ignore_index=True)
    path = os.path.join(results_dir, f"comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv")
    table.to_csv(path, sep="\t", index=False)
    print(f"Unified table saved to {path}")
    return table
