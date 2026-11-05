"""
General-purpose data operations (Section 4): not part of the taxonomy because
they leave no corrupted signal, but useful for broader robustness analyses.

  record_removal         remove existing records (data incompleteness, sparsity,
                         missing feedback)
  metadata_modification  modify record metadata (timestamps) without making it invalid
"""

from __future__ import annotations

import pandas as pd

from injectors.base import MODIFY, REMOVE, Perturbation
from injectors.engine import RowNoiseInjector
from injectors.registry import register
from utils import TS, date_to_unix, window_to_seconds


@register
class RecordRemoval(Perturbation):
    name = "record_removal"
    signal = "general"
    observability = None
    operation = REMOVE
    default_target = "user"
    description = "Remove existing records of the selected users/items (simulates incompleteness)."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "keep_at_least": 1,              # never remove the last k records of a target
    }

    def perturb(self, df):
        keep = int(self.p.get("keep_at_least", 0))
        sizes = df.groupby(self.target_col)[self.target_col].transform("size")
        # restrict each target to (size - keep) removable records
        rank = df.groupby(self.target_col).cumcount()
        eligible = rank >= keep if keep else pd.Series(True, index=df.index)
        idx = self.select_records(df, eligible & (sizes > keep))
        if idx:
            self.log_remove(df, idx)
            df = df.drop(index=idx)
        return df


@register
class MetadataModification(Perturbation):
    name = "metadata_modification"
    signal = "general"
    observability = None
    operation = MODIFY
    default_target = "user"
    requires = (TS,)
    description = "Modify timestamps (shift or resample) while keeping them valid."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "mode": "shift",                 # shift | resample
        "shift": {"days": 30, "hours": 0, "minutes": 0},     # +/- shift, sign chosen at random
        "temporal_behavior": {"start_timestamp": None, "end_timestamp": None},   # resample range
    }

    def perturb(self, df):
        idx = self.select_records(df, df[TS].notna())
        if not idx:
            return df
        if self.p["mode"] == "shift":
            delta = window_to_seconds(self.p["shift"])
            signs = self.rng.choice([-1, 1], size=len(idx))
            new = [max(0, int(df.at[i, TS]) + int(s) * delta) for i, s in zip(idx, signs)]
        elif self.p["mode"] == "resample":
            tb = self.p.get("temporal_behavior") or {}
            lo = date_to_unix(tb.get("start_timestamp")) or self.ctx.ts_min
            hi = date_to_unix(tb.get("end_timestamp")) or self.ctx.ts_max
            new = [int(x) for x in self.rng.integers(lo, hi + 1, size=len(idx))]
        else:
            raise ValueError(f"[{self.name}] unknown mode '{self.p['mode']}'")
        self.log_modify(df, idx, TS, new)
        df.loc[idx, TS] = pd.array(new, dtype=df[TS].dtype)
        return df


class GeneralNoiseInjector(RowNoiseInjector):
    signal = "general"
