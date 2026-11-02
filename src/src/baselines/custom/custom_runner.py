"""
Custom models (Section 4.1): any class following this interface can be evaluated
in the same pipeline as the library models.

    class MyModel:
        def __init__(self, **params): ...
        def fit(self, train: pd.DataFrame, val: pd.DataFrame | None, seed: int): ...
        def recommend(self, users: list, k: int, exclude: dict) -> pd.DataFrame
            # columns user_id, item_id, rank, score; ``exclude`` maps users to seen items
        def predict(self, pairs: pd.DataFrame) -> np.ndarray      # optional (rating models)

In evaluation.yaml:   custom: {models: [{name: "mypackage.module:MyModel", params: {...}}]}
Accuracy for custom models is computed by RecProbe (src/evaluation/metrics.py).
Two reference models are built in: MostPop and Random.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from baselines.base_runner import BaseRunner, RunOutput
from evaluation.metrics import accuracy_per_user, aggregate_users
from utils import ITEM, RATING, USER


class MostPop:
    def __init__(self, **params):
        self.pop = None

    def fit(self, train, val, seed):
        self.pop = train[ITEM].value_counts()

    def recommend(self, users, k, exclude):
        rows = []
        for u in users:
            seen = exclude.get(u, set())
            top = [i for i in self.pop.index[:k + len(seen)] if i not in seen][:k]
            rows.extend((u, i, r + 1, float(self.pop[i])) for r, i in enumerate(top))
        return pd.DataFrame(rows, columns=[USER, ITEM, "rank", "score"])


class Random:
    def __init__(self, **params):
        self.items, self.rng = None, None

    def fit(self, train, val, seed):
        self.items = train[ITEM].unique()
        self.rng = np.random.default_rng(seed)

    def recommend(self, users, k, exclude):
        rows = []
        for u in users:
            cand = np.setdiff1d(self.items, list(exclude.get(u, set())))
            pick = self.rng.choice(cand, size=min(k, len(cand)), replace=False)
            rows.extend((u, i, r + 1, 0.0) for r, i in enumerate(pick))
        return pd.DataFrame(rows, columns=[USER, ITEM, "rank", "score"])


BUILTIN = {"MostPop": MostPop, "Random": Random}


def load_model_class(name: str):
    if name in BUILTIN:
        return BUILTIN[name]
    if ":" not in name:
        raise ValueError(f"Custom model '{name}': use 'package.module:ClassName' or one of {list(BUILTIN)}")
    mod, attr = name.split(":", 1)
    return getattr(importlib.import_module(mod), attr)


class CustomExperimentRunner(BaseRunner):
    framework = "custom"

    def run(self, model_cfg, train, val, test, seed):
        model = load_model_class(model_cfg["name"])(**(model_cfg.get("params") or {}))
        model.fit(train, val, seed)
        users = list(test[USER].unique())
        exclude = train.groupby(USER)[ITEM].agg(set).to_dict()
        ranked = model.recommend(users, self.k_max, exclude)
        preds = None
        if hasattr(model, "predict") and RATING in test.columns and test[RATING].notna().any():
            preds = test[[USER, ITEM, RATING]].assign(prediction=np.asarray(model.predict(test[[USER, ITEM]])))
        metrics = self.fw_cfg.get("metrics") or [{"NDCG": {"k": self.eval_cfg.top_k}}]
        per_user = accuracy_per_user(metrics, ranked, test, preds,
                                     self.fw_cfg.get("rating_threshold"), self.eval_cfg.top_k)
        return RunOutput(aggregate_users(per_user, per_user.index), ranked, per_user, preds)
