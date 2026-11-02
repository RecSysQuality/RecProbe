"""
Row A of Table 1 - Interaction-level defects (all methods).

  point       Invalid or placeholder timestamp
  contextual  Sequence corruption
  collective  Burst, Shilling
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from config.config import deep_merge
from injectors.base import ADD, CONTEXTUAL, MODIFY, POINT, Perturbation, _sample_excluding
from injectors.collective import BurstPerturbation, ShillingPerturbation
from injectors.engine import RowNoiseInjector
from injectors.registry import register
from utils import TS, USER, date_to_unix

DAY = 86400


@register
class InvalidTimestamp(Perturbation):
    """A timestamp outside the admissible range of the platform, or a default value
    encoding a missing time (e.g. a date in the future, the Unix epoch)."""

    name = "invalid_timestamp"
    signal = "interaction"
    observability = POINT
    operation = MODIFY
    default_target = "user"
    requires = (TS,)
    description = "Invalid or placeholder timestamp: future dates, Unix epoch, sentinels, pre-platform dates."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        # future | epoch | sentinel | pre_platform
        "modes": ["future", "epoch"],
        "mode_weights": None,
        "future_offset_days": [1, 3650],     # days after the reference time
        "reference_time": "dataset_end",     # dataset_end | now (now is not reproducible)
        "platform_start": None,              # pre_platform: before this date (default: first record)
        "sentinel_value": -1,
    }

    def perturb(self, df):
        idx = self.select_records(df, df[TS].notna())
        if not idx:
            return df
        modes = list(self.p["modes"])
        w = self.p.get("mode_weights")
        w = np.asarray(w, float) / np.sum(w) if w else None
        chosen = self.rng.choice(modes, size=len(idx), p=w)
        ref = int(pd.Timestamp.now().timestamp()) if self.p["reference_time"] == "now" else self.ctx.ts_max
        start = date_to_unix(self.p.get("platform_start")) or self.ctx.ts_min
        lo, hi = self.p["future_offset_days"]
        new = []
        for m in chosen:
            if m == "future":
                new.append(ref + int(self.rng.integers(int(lo) * DAY, int(hi) * DAY + 1)))
            elif m == "epoch":
                new.append(0)
            elif m == "sentinel":
                new.append(int(self.p["sentinel_value"]))
            elif m == "pre_platform":
                new.append(int(self.rng.integers(DAY, max(start, 2 * DAY))))
            else:
                raise ValueError(f"[{self.name}] unknown mode '{m}'")
        self.log_modify(df, idx, TS, new)
        df.loc[idx, TS] = pd.array(new, dtype=df[TS].dtype)
        self.stats["modes"] = {m: int((chosen == m).sum()) for m in modes}
        return df


@register
class SequenceCorruption(Perturbation):
    """A spurious interaction, valid in isolation, inserted in the user's chronological
    history but inconsistent with it or with item co-occurrence patterns
    (e.g. an accidental click on a mainstream blockbuster inside a niche sequence)."""

    name = "sequence_corruption"
    signal = "interaction"
    observability = CONTEXTUAL
    operation = ADD
    default_target = "user"
    description = "Insert items inconsistent with the user's history at a position inside the sequence."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 3,
        # cooccurrence -> lowest co-occurrence with the history (normalised by item popularity)
        # popularity   -> farthest from the user's typical item popularity (blockbuster vs niche)
        # category     -> item of a category absent from the user's history (needs item metadata)
        "inconsistency": "cooccurrence",
        "candidate_pool": 200,
        "min_history": 2,
        # the inserted rating is valid and plausible for the user: Gaussian on the user's ratings
        "rating_behavior": {"min_rating": None, "max_rating": None, "sampling_strategy": "gaussian"},
    }

    def perturb(self, df):
        if self.target_col != USER:
            raise ValueError("sequence_corruption targets users (target: user)")
        ctx, rng = self.ctx, self.rng
        user_items = ctx.user_items(df)
        user_ts = df.groupby(USER)[TS].agg(["min", "max"]) if ctx.has_ts else None
        user_ratings = ctx.ratings_of(USER)
        strategy = self.p["inconsistency"]
        if strategy == "category" and ctx.category is None:
            self.logger.warning("[sequence_corruption] no item categories: falling back to cooccurrence")
            strategy = "cooccurrence"
        X, users_idx, items_idx = ctx.interaction_matrix()
        pop = np.asarray(X.sum(axis=0)).ravel()
        all_items = items_idx.to_numpy()

        eligible = {u for u, s in user_items.items() if len(s) >= int(self.p["min_history"])}
        remaining, records = self.budget, []
        for user in self.candidate_nodes(eligible):
            if remaining <= 0:
                break
            hist = user_items[user]
            n = self.per_node(len(all_items) - len(hist), remaining)
            if n <= 0:
                continue
            cands = _sample_excluding(all_items, int(self.p["candidate_pool"]), hist, rng)
            if len(cands) == 0:
                continue
            chosen = self._most_inconsistent(strategy, cands, hist, n, X, items_idx, pop)
            if ctx.has_ts and user in user_ts.index and pd.notna(user_ts.at[user, "min"]):
                lo, hi = int(user_ts.at[user, "min"]), int(user_ts.at[user, "max"])
                times = rng.integers(lo, hi + 1, size=len(chosen))
            else:
                times = [None] * len(chosen)
            ratings = self.sample_rating_values(len(chosen), user_ratings.get(user)) if ctx.has_rating \
                else [None] * len(chosen)
            for it, t, r in zip(chosen, times, ratings):
                records.append(self.new_record(user, it, t, r))
            remaining -= len(chosen)
        self.stats["targets"] = len({r[USER] for r in records})
        self.stats["inconsistency"] = strategy
        return self.append_records(df, records, "target")

    def _most_inconsistent(self, strategy, cands, hist, n, X, items_idx, pop):
        rng = self.rng
        cidx = items_idx.get_indexer(cands)
        hidx = items_idx.get_indexer(list(hist))
        if strategy == "cooccurrence":
            h = np.asarray(X[:, hidx].sum(axis=1)).ravel()            # users weighted by overlap
            co = np.asarray(X[:, cidx].T @ h).ravel()
            score = co / np.maximum(pop[cidx], 1)                       # P(history | candidate)
        elif strategy == "popularity":
            user_logpop = np.log1p(pop[hidx]).mean()
            score = -np.abs(np.log1p(pop[cidx]) - user_logpop)
        elif strategy == "category":
            seen = {self.ctx.category.get(i) for i in hist}
            score = np.array([0.0 if self.ctx.category.get(c) not in seen else 1.0 for c in cands])
        else:
            raise ValueError(f"[{self.name}] unknown inconsistency '{strategy}'")
        order = np.lexsort((rng.random(len(cands)), score))           # lowest score first, random ties
        return list(np.asarray(cands)[order[:n]])


@register
class InteractionBurst(BurstPerturbation):
    """A concentration of interactions on the same target items within a short time window."""

    name = "interaction_burst"
    signal = "interaction"
    description = "Burst of interactions on target items inside a time window (existing or fake users)."
    DEFAULTS = deep_merge(BurstPerturbation.BURST_DEFAULTS, {})


@register
class InteractionShilling(ShillingPerturbation):
    """A group of coordinated fake profiles interacting with the same target items, with
    filler interactions that make each profile plausible in isolation."""

    name = "interaction_shilling"
    signal = "interaction"
    description = "Coordinated fake profiles on the same targets, with random/popular/segment fillers."
    DEFAULTS = deep_merge(ShillingPerturbation.SHILLING_DEFAULTS, {})


class InteractionNoiseInjector(RowNoiseInjector):
    signal = "interaction"
