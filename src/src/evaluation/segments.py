"""
User segments (Section 4.1, phase 4).

Segments are defined by percentile thresholds on the distribution of user
interaction counts (e.g. 75 -> users whose count is >= the 75th percentile;
0 -> the whole population), or by explicit lists of user ids. They are computed
once on the CLEAN training set and reused for every perturbed run. Fake users
introduced by the injection are excluded from the percentiles and from the
evaluation.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd

from utils import USER


def segment_name(p: float) -> str:
    return "all" if p <= 0 else f"p{int(p) if float(p).is_integer() else p}"


def build_segments(clean_train: pd.DataFrame, percentiles, explicit: dict | None = None,
                   fake_users: set | None = None) -> dict[str, set]:
    fake_users = fake_users or set()
    counts = clean_train.loc[~clean_train[USER].isin(fake_users), USER].value_counts()
    segments: dict[str, set] = {}
    for p in percentiles:
        thr = np.percentile(counts.to_numpy(), float(p)) if p > 0 else -np.inf
        segments[segment_name(p)] = set(counts.index[counts >= thr])
    for name, spec in (explicit or {}).items():
        if isinstance(spec, str) and os.path.isfile(spec):
            with open(spec) as fh:
                ids = {line.strip() for line in fh if line.strip()}
        else:
            ids = {str(x) for x in (spec or [])}
        segments[str(name)] = ids - fake_users
    return segments


def describe(segments: dict[str, set], clean_train: pd.DataFrame) -> pd.DataFrame:
    counts = clean_train[USER].value_counts()
    rows = []
    for name, users in segments.items():
        c = counts.reindex(list(users)).dropna()
        rows.append({"segment": name, "users": len(users),
                     "min_interactions": int(c.min()) if len(c) else None,
                     "median_interactions": float(c.median()) if len(c) else None})
    return pd.DataFrame(rows)
