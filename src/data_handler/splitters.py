"""
Train / validation / test split strategies (Section 4.1, phase 2):

  * random_holdout          interactions randomly partitioned by the given ratios
  * temporal_leave_one_out  per user, most recent -> test, second most recent -> validation
  * global_temporal         all interactions after a timestamp -> test (no temporal leakage)
  * random_leave_one_out    one random interaction per user -> test (and one -> validation)

Custom strategies can be plugged in with ``strategy: "package.module:function"``
or with ``register_splitter``; the function signature is
``f(df, split_cfg, rng) -> (train, validation, test)``.
"""

from __future__ import annotations

import importlib
from typing import Callable

import numpy as np
import pandas as pd

from utils import ITEM, TS, USER, date_to_unix

SPLITTERS: dict[str, Callable] = {}


def register_splitter(name: str):
    def deco(fn):
        SPLITTERS[name] = fn
        return fn
    return deco


def _ratios(cfg) -> tuple[float, float, float]:
    tr, va, te = float(cfg.training), float(cfg.validation or 0), float(cfg.test or 0)
    s = tr + va + te
    if s <= 0:
        raise ValueError("Split ratios must sum to a positive value")
    return tr / s, va / s, te / s


def _require_ts(df: pd.DataFrame, strategy: str):
    if TS not in df.columns or df[TS].isna().all():
        raise ValueError(f"The '{strategy}' split requires a timestamp field")


@register_splitter("random_holdout")
def random_holdout(df, cfg, rng):
    _, va, te = _ratios(cfg)
    idx = rng.permutation(len(df))
    n_te, n_va = int(round(te * len(df))), int(round(va * len(df)))
    test = df.iloc[idx[:n_te]]
    val = df.iloc[idx[n_te:n_te + n_va]]
    train = df.iloc[idx[n_te + n_va:]]
    return train, val, test


def _leave_one_out(df, cfg, order: pd.Series):
    """order: per-row key; the largest key per user goes to test, then validation."""
    use_val = float(cfg.validation or 0) > 0
    tmp = df.assign(_key=order.values)
    tmp = tmp.sort_values([USER, "_key"], kind="mergesort")
    rank_from_end = tmp.groupby(USER).cumcount(ascending=False)
    size = tmp.groupby(USER)[USER].transform("size")
    min_size = 3 if use_val else 2           # always keep >= 1 interaction in train
    test_mask = (rank_from_end == 0) & (size >= min_size)
    val_mask = (rank_from_end == 1) & (size >= min_size) if use_val else pd.Series(False, index=tmp.index)
    tmp = tmp.drop(columns="_key")
    return tmp[~test_mask & ~val_mask], tmp[val_mask], tmp[test_mask]


@register_splitter("temporal_leave_one_out")
def temporal_leave_one_out(df, cfg, rng):
    _require_ts(df, "temporal_leave_one_out")
    # ties broken randomly but reproducibly
    order = df[TS].astype("float64").fillna(-np.inf) + rng.random(len(df)) * 1e-3
    return _leave_one_out(df, cfg, order)


@register_splitter("random_leave_one_out")
def random_leave_one_out(df, cfg, rng):
    return _leave_one_out(df, cfg, pd.Series(rng.random(len(df)), index=df.index))


@register_splitter("global_temporal")
def global_temporal(df, cfg, rng):
    _require_ts(df, "global_temporal")
    ts = df[TS].astype("float64")
    _, va, te = _ratios(cfg)
    t_test = date_to_unix(cfg.get("split_timestamp")) if cfg.get("split_timestamp") is not None \
        else float(ts.quantile(1 - te))
    if cfg.get("validation_timestamp") is not None:
        t_val = date_to_unix(cfg.validation_timestamp)
    elif va > 0:
        t_val = float(ts.quantile(1 - te - va))
    else:
        t_val = t_test
    test = df[ts > t_test]
    val = df[(ts > t_val) & (ts <= t_test)]
    train = df[(ts <= t_val) | ts.isna()]
    return train, val, test


def _ensure_coverage(train, val, test, drop: bool = False):
    """
    Records of users/items unseen in training cannot be evaluated: they are moved to
    the training set, or dropped when moving them would break a global temporal cut
    (records after the split timestamp must never reach the training set).
    """
    users, items = set(train[USER]), set(train[ITEM])
    moved = []
    out = []
    for part in (val, test):
        bad = ~part[USER].isin(users) | ~part[ITEM].isin(items)
        moved.append(part[bad])
        out.append(part[~bad])
    if not drop:
        train = pd.concat([train] + moved)
    return train, out[0], out[1]


def split(df: pd.DataFrame, cfg, rng: np.random.Generator, logger=None):
    name = cfg.strategy
    if name in SPLITTERS:
        fn = SPLITTERS[name]
    elif ":" in name:
        mod, attr = name.split(":", 1)
        fn = getattr(importlib.import_module(mod), attr)
    else:
        raise ValueError(f"Unknown split strategy '{name}'. Available: {sorted(SPLITTERS)} "
                         "or 'package.module:function'.")
    train, val, test = fn(df, cfg, rng)
    if cfg.get("ensure_train_coverage", True):
        train, val, test = _ensure_coverage(train, val, test, drop=(name == "global_temporal"))
    if logger:
        logger.info(f"Split '{name}': train={len(train)} validation={len(val)} test={len(test)}")
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)
