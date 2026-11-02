"""
Row B of Table 1 - Rating-level defects (explicit-feedback methods).

  point       Invalid or placeholder rating
  contextual  History-inconsistent rating, Rating flip
  collective  Burst, Shilling (random, average, bandwagon, segment, love/hate; push or nuke)
"""

from __future__ import annotations

import numpy as np

from config.config import deep_merge
from injectors.base import CONTEXTUAL, MODIFY, POINT, Perturbation, _resolve_size
from injectors.collective import BurstPerturbation, ShillingPerturbation
from injectors.engine import RowNoiseInjector
from injectors.registry import register
from utils import ITEM, RATING, USER


def _grid(ctx) -> np.ndarray:
    step = ctx.rstep or (ctx.rmax - ctx.rmin) / 4
    return np.arange(ctx.rmin, ctx.rmax + step / 2, step)


@register
class InvalidRating(Perturbation):
    """A rating that violates the rating schema: out of range, of the wrong granularity,
    or a placeholder for a missing value (e.g. 0 or 6 on a 1-5 scale, 3.7 on an integer scale)."""

    name = "invalid_rating"
    signal = "rating"
    observability = POINT
    operation = MODIFY
    default_target = "user"
    requires = (RATING,)
    description = "Out-of-range, wrong-granularity or placeholder ratings."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "modes": ["out_of_range", "wrong_granularity", "placeholder"],
        "mode_weights": None,
        "out_of_range_values": None,     # default: [min - step, max + step]
        "placeholder_value": 0,
    }

    def perturb(self, df):
        ctx, rng = self.ctx, self.rng
        modes = list(self.p["modes"])
        if "wrong_granularity" in modes and not ctx.rstep:
            self.logger.warning("[invalid_rating] continuous rating scale: 'wrong_granularity' disabled")
            modes.remove("wrong_granularity")
        idx = self.select_records(df, df[RATING].notna())
        if not idx or not modes:
            return df
        w = self.p.get("mode_weights")
        w = np.asarray(w[:len(modes)], float) / np.sum(w[:len(modes)]) if w else None
        chosen = rng.choice(modes, size=len(idx), p=w)
        step = ctx.rstep or 1.0
        oor = self.p.get("out_of_range_values") or [ctx.rmin - step, ctx.rmax + step]
        new = []
        for i, m in zip(idx, chosen):
            r = float(df.at[i, RATING])
            if m == "out_of_range":
                new.append(float(rng.choice(oor)))
            elif m == "placeholder":
                new.append(float(self.p["placeholder_value"]))
            elif m == "wrong_granularity":
                # a value strictly between two grid points, kept inside the admissible range
                off = rng.uniform(0.15, 0.85) * step
                v = r + off if r + step <= ctx.rmax else r - off
                new.append(round(v, 2))
            else:
                raise ValueError(f"[{self.name}] unknown mode '{m}'")
        self.log_modify(df, idx, RATING, new)
        df.loc[idx, RATING] = new
        self.stats["modes"] = {m: int((chosen == m).sum()) for m in modes}
        return df


@register
class HistoryInconsistentRating(Perturbation):
    """A rating that deviates strongly from the user's rating history or from the ratings
    the item usually receives (e.g. 1 star from a user who consistently rates 4-5)."""

    name = "history_inconsistent_rating"
    signal = "rating"
    observability = CONTEXTUAL
    operation = MODIFY
    default_target = "user"
    requires = (RATING,)
    description = "Replace ratings with values far from the user's (or item's) rating distribution."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 3,
        "reference": "user",             # user history | item feedback
        "min_history": 3,                # ratings needed to define the reference distribution
        "min_deviation": 2.0,            # |new - reference mean| >= min_deviation
        "value": "extreme",              # extreme (farthest value) | sampled (any value far enough)
    }

    def perturb(self, df):
        ctx, rng = self.ctx, self.rng
        ref_col = USER if self.p["reference"] == "user" else ITEM
        g = ctx.reference.groupby(ref_col)[RATING]
        mean, count = g.mean(), g.count()
        grid = _grid(ctx)
        dev = float(self.p["min_deviation"])
        m = df[ref_col].map(mean)
        ok = df[RATING].notna() & (df[ref_col].map(count) >= int(self.p["min_history"]))
        # at least one admissible value must be far enough from the reference mean
        far = np.maximum(np.abs(grid.max() - m), np.abs(m - grid.min())) >= dev
        idx = self.select_records(df, ok & far.fillna(False))
        new = []
        for i in idx:
            mu, r = float(m.at[i]), float(df.at[i, RATING])
            cands = grid[(np.abs(grid - mu) >= dev) & (grid != r)]
            if len(cands) == 0:
                cands = grid[np.argmax(np.abs(grid - mu))][None]
            if self.p["value"] == "extreme":
                v = cands[np.argmax(np.abs(cands - mu))]
            else:
                v = rng.choice(cands)
            new.append(float(v))
        if idx:
            self.log_modify(df, idx, RATING, new)
            df.loc[idx, RATING] = new
        return df


@register
class RatingFlip(Perturbation):
    """A rating whose polarity is reversed (e.g. 5 stars -> 1 star), making it inconsistent
    with the paired review. Mapping: r' = min + max - r."""

    name = "rating_flip"
    signal = "rating"
    observability = CONTEXTUAL
    operation = MODIFY
    default_target = "user"
    requires = (RATING,)
    description = "Reverse the polarity of ratings: r' = min + max - r."
    DEFAULTS = {
        "selection_strategy": "uniform",
        "min_per_node": 1,
        "max_per_node": 5,
        "skip_neutral": True,            # ratings at the scale midpoint are unchanged by a flip
    }

    def perturb(self, df):
        ctx = self.ctx
        flipped = ctx.rmin + ctx.rmax - df[RATING]
        ok = df[RATING].notna()
        if self.p.get("skip_neutral", True):
            ok &= flipped != df[RATING]
        idx = self.select_records(df, ok)
        if idx:
            new = flipped.loc[idx].astype(float).tolist()
            self.log_modify(df, idx, RATING, new)
            df.loc[idx, RATING] = new
        return df


@register
class RatingBurst(BurstPerturbation):
    """A concentration of ratings on the same target items within a short time window,
    shifting their rating distribution (e.g. a wave of low ratings on a new item)."""

    name = "rating_burst"
    signal = "rating"
    requires = (RATING,)
    description = "Burst of ratings on target items inside a time window, shifting their distribution."
    DEFAULTS = deep_merge(BurstPerturbation.BURST_DEFAULTS, {})


@register
class RatingShilling(ShillingPerturbation):
    """A group of coordinated profiles rating the same target items in the same direction
    (push or nuke), with filler ratings that make each profile plausible in isolation.

    Attack models (Lam and Riedl, 2004; Gunes et al., 2014):
      random     fillers: random items, ratings ~ N(global mean, global std)
      average    fillers: random items, ratings ~ N(item mean, item std)
      bandwagon  selected: most popular items rated max; fillers as in random
      segment    selected: items of the target's segment rated max; fillers rated min
      love_hate  fillers: random items rated max (nuke) or min (push)
    The target is rated max (push) or min (nuke).
    """

    name = "rating_shilling"
    signal = "rating"
    requires = (RATING,)
    description = "Shilling attack: random, average, bandwagon, segment, love/hate; push or nuke."
    DEFAULTS = deep_merge(ShillingPerturbation.SHILLING_DEFAULTS, {
        "attack_type": "average",
        "selected_size": 5,              # bandwagon / segment: number of selected items
    })

    def profile_items_and_ratings(self, targets):
        ctx, rng = self.ctx, self.rng
        attack = self.p["attack_type"]
        push = self.p.get("direction", "push") == "push"
        n_fill = _resolve_size(self.p.get("filler_size", 20), len(ctx.item_pop), rng)
        exclude = set(targets)
        sel_items, sel_ratings = [], []
        if attack == "bandwagon":
            k = int(self.p.get("selected_size", 5))
            sel_items = [i for i in ctx.item_pop.index[:k + len(exclude)] if i not in exclude][:k]
            sel_ratings = [ctx.rmax if push else ctx.rmin] * len(sel_items)
        elif attack == "segment":
            k = int(self.p.get("selected_size", 5))
            pool = [i for i in self.segment_pool(targets) if i not in exclude]
            sel_items = list(rng.choice(pool, size=min(k, len(pool)), replace=False)) if pool else []
            sel_ratings = [ctx.rmax if push else ctx.rmin] * len(sel_items)
        elif attack not in ("random", "average", "love_hate"):
            raise ValueError(f"[{self.name}] unknown attack_type '{attack}'")
        exclude |= set(sel_items)
        fillers = list(self.filler_items(self.p.get("filler_strategy", "random"), n_fill, exclude, targets))
        if attack == "average":
            f_ratings = list(self.item_rating_sample(fillers))
        elif attack == "segment":
            f_ratings = [ctx.rmin if push else ctx.rmax] * len(fillers)
        elif attack == "love_hate":
            f_ratings = [ctx.rmin if push else ctx.rmax] * len(fillers)
        else:   # random, bandwagon
            f_ratings = list(self.global_rating_sample(len(fillers)))
        return sel_items + fillers, sel_ratings + f_ratings


class RatingNoiseInjector(RowNoiseInjector):
    signal = "rating"
