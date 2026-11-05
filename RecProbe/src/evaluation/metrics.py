"""
Metrics computed by RecProbe from the ranked lists exported by each library.

  * Beyond-accuracy (Section 4): coverage, novelty (Ge et al., 2010),
    Shannon entropy (Ricci et al., 2022) and Gini index (Jannach et al., 2015).
    They are always computed here, so they are comparable across libraries.
  * Accuracy: by default accuracy comes from the integrated library. The
    functions below are used when a library cannot report per-user values
    (needed for user segments) and for custom models.

Custom metrics: ``"package.module:function"`` with signature
``f(ranked, test, train, k) -> float | pd.Series (per user)``.
"""

from __future__ import annotations

import importlib

import numpy as np
import pandas as pd

from utils import ITEM, RATING, USER


# --------------------------------------------------------------------------- #
# helpers                                                                     #
# --------------------------------------------------------------------------- #
def topk_lists(ranked: pd.DataFrame, k: int) -> dict:
    r = ranked[ranked["rank"] <= k].sort_values([USER, "rank"])
    return r.groupby(USER)[ITEM].apply(list).to_dict()


def relevant_items(test: pd.DataFrame, threshold: float | None = None) -> dict:
    t = test
    if threshold is not None and RATING in t.columns and t[RATING].notna().any():
        t = t[t[RATING] >= threshold]
    return t.groupby(USER)[ITEM].agg(set).to_dict()


def parse_metric_list(metrics) -> list[tuple[str, dict]]:
    """[{"NDCG": {"k": 10}}, "RMSE", {"name": "Recall", "k": 20}] -> [(name, params)]."""
    out = []
    for m in metrics or []:
        if isinstance(m, str):
            out.append((m, {}))
        elif isinstance(m, dict) and "name" in m:
            out.append((m["name"], {k: v for k, v in m.items() if k != "name"}))
        elif isinstance(m, dict):
            for name, params in m.items():
                out.append((name, dict(params or {})))
    return out


# --------------------------------------------------------------------------- #
# accuracy                                                                    #
# --------------------------------------------------------------------------- #
def _user_ranking_metric(name: str, recs: list, rel: set, k: int) -> float:
    recs = recs[:k]
    hits = np.array([1.0 if i in rel else 0.0 for i in recs])
    n_rel = len(rel)
    if name == "precision":
        return hits.sum() / k
    if name == "recall":
        return hits.sum() / n_rel if n_rel else 0.0
    if name in ("hitratio", "hit", "hr", "hitrate"):
        return float(hits.any())
    if name == "ndcg":
        disc = 1.0 / np.log2(np.arange(2, len(recs) + 2))
        idcg = disc[:min(n_rel, k)].sum() if len(disc) else 0.0
        return float((hits * disc).sum() / idcg) if idcg > 0 else 0.0
    if name == "mrr":
        pos = np.flatnonzero(hits)
        return 1.0 / (pos[0] + 1) if len(pos) else 0.0
    if name == "map":
        if not n_rel:
            return 0.0
        prec = np.cumsum(hits) / np.arange(1, len(hits) + 1)
        return float((prec * hits).sum() / min(n_rel, k))
    raise ValueError(f"Unknown ranking metric '{name}'")


RANKING = {"precision", "recall", "hitratio", "hit", "hr", "hitrate", "ndcg", "mrr", "map"}
RATING_METRICS = {"rmse", "mae", "mse"}


def accuracy_per_user(metrics, ranked: pd.DataFrame | None, test: pd.DataFrame,
                      predictions: pd.DataFrame | None = None, threshold=None,
                      default_k: int = 10) -> pd.DataFrame:
    """Per-user accuracy (rows: users, columns: e.g. 'NDCG@10', 'RMSE')."""
    cols = {}
    rel = relevant_items(test, threshold)
    for name, params in parse_metric_list(metrics):
        low = name.lower()
        if low in RANKING:
            if ranked is None:
                continue
            k = int(params.get("k", default_k))
            recs = topk_lists(ranked, k)
            cols[f"{name}@{k}"] = {u: _user_ranking_metric(low, recs.get(u, []), r, k) for u, r in rel.items()}
        elif low in RATING_METRICS:
            if predictions is None or predictions.empty:
                continue
            err = predictions["prediction"] - predictions[RATING]
            g = err.groupby(predictions[USER])
            # user-based averaging, as in Cornac (user_based=True): per-user error, then mean
            if low == "mae":
                cols[name] = g.apply(lambda e: e.abs().mean()).to_dict()
            elif low == "rmse":
                cols[name] = g.apply(lambda e: float(np.sqrt((e ** 2).mean()))).to_dict()
            else:
                cols[name] = g.apply(lambda e: (e ** 2).mean()).to_dict()
        elif ":" in name:
            continue                                                        # custom metric
        else:
            raise ValueError(f"Unknown metric '{name}'")
    return pd.DataFrame(cols)


def aggregate_users(per_user: pd.DataFrame, users) -> dict:
    sub = per_user.loc[per_user.index.intersection(list(users))]
    out = {}
    for c in sub.columns:
        v = sub[c].dropna()
        if not v.empty:
            out[c] = float(v.mean())
    return out


# --------------------------------------------------------------------------- #
# beyond accuracy                                                             #
# --------------------------------------------------------------------------- #
def beyond_accuracy(ranked: pd.DataFrame, users, k: int, catalog, item_pop: pd.Series, n_users: int,
                    which=("coverage", "novelty", "entropy", "gini")) -> dict:
    """
    ranked   : top-k lists (user_id, item_id, rank)
    catalog  : items that can be recommended (clean training catalogue)
    item_pop : number of training users per item (clean training set, fake users excluded)
    n_users  : number of training users (same reference)
    """
    r = ranked[(ranked["rank"] <= k) & ranked[USER].isin(set(users))]
    out = {}
    if r.empty:
        return out
    catalog = list(catalog)
    counts = r[ITEM].value_counts()
    if "coverage" in which:
        out[f"coverage@{k}"] = float(counts.index.isin(catalog).sum() / max(len(catalog), 1))
    if "novelty" in which:
        # mean self-information of recommended items: -log2(popularity / #users)
        p = r[ITEM].map(item_pop).fillna(0).clip(lower=1) / max(n_users, 1)
        out[f"novelty@{k}"] = float((-np.log2(p)).mean())
    if "entropy" in which:
        share = counts / counts.sum()
        out[f"entropy@{k}"] = float(-(share * np.log2(share)).sum())
    if "gini" in which:
        full = counts.reindex(catalog, fill_value=0).sort_values().to_numpy(dtype=float)
        n = len(full)
        if n > 1 and full.sum() > 0:
            j = np.arange(1, n + 1)
            out[f"gini@{k}"] = float(((2 * j - n - 1) * full).sum() / ((n - 1) * full.sum()))
    return out


# --------------------------------------------------------------------------- #
# custom metrics                                                              #
# --------------------------------------------------------------------------- #
def custom_metrics(specs, ranked, test, train, users, k) -> dict:
    out = {}
    for name, params in parse_metric_list(specs):
        if ":" not in name:
            continue
        mod, attr = name.split(":", 1)
        fn = getattr(importlib.import_module(mod), attr)
        users = set(users)
        val = fn(ranked[ranked[USER].isin(users)], test[test[USER].isin(users)], train,
                 int(params.get("k", k)))
        if isinstance(val, pd.Series):
            val = float(val.loc[val.index.intersection(list(users))].mean())
        out[f"{attr}@{params.get('k', k)}"] = float(val)
    return out
