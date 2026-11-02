"""
Shared machinery of the defect-injection phase (Section 4.1, phase 3).

Every perturbation of Table 1 derives from ``Perturbation`` and reuses:

  * the budget            -> number of interactions to add or corrupt, absolute
                             or as a fraction of the edges of the split;
  * the target selection  -> users or items, chosen uniformly at random or by
                             popularity (top-k / bottom-k), or an explicit list;
  * the iterative loop    -> until the budget is reached, a target is selected
                             and the perturbation is applied to its interactions,
                             with min/max interactions per target;
  * rating generation     -> uniform on a user-defined range, or Gaussian with
                             the mean/std of the target user or item, then
                             clipped and mapped to the rating scale;
  * temporal behaviour    -> interval [start, end] and burst window;
  * record producers      -> existing users or fake users with ``random`` or
                             ``mimic`` behaviour;
  * text generation       -> LLM-generated reviews (see text_generation.py).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import sparse

from utils import (ITEM, RATING, REVIEW, TS, USER, date_to_unix, ordered_nodes, sample_ratings,
                   snap_to_scale, window_to_seconds)

ADD, MODIFY, REMOVE = "add", "modify", "remove"
POINT, CONTEXTUAL, COLLECTIVE = "point", "contextual", "collective"


@dataclass
class InjectionResult:
    splits: dict                       # {"train": df, "validation": df|None, "test": df|None}
    log: pd.DataFrame                  # one row per added / modified / removed record
    fake_users: set = field(default_factory=set)
    summary: dict = field(default_factory=dict)


class PerturbationContext:
    """What a perturbation can look at: the reference data used as context."""

    def __init__(self, reference: pd.DataFrame, items: pd.DataFrame | None, rating_scale,
                 rng: np.random.Generator, logger, split_name: str = "train"):
        self.reference = reference
        self.items = items
        self.rng = rng
        self.logger = logger
        self.split_name = split_name
        self.has_rating = RATING in reference.columns and reference[RATING].notna().any()
        self.has_review = REVIEW in reference.columns
        self.has_ts = TS in reference.columns and reference[TS].notna().any()
        if rating_scale:
            self.rmin, self.rmax, self.rstep = (float(rating_scale[0]), float(rating_scale[1]),
                                                rating_scale[2])
        else:
            self.rmin, self.rmax, self.rstep = 1.0, 5.0, 1.0
        if self.has_ts:
            ts = reference[TS].dropna().astype("int64")
            self.ts_min, self.ts_max = int(ts.min()), int(ts.max())
        else:
            self.ts_min = self.ts_max = None
        self.user_pop = reference[USER].value_counts()
        self.item_pop = reference[ITEM].value_counts()
        self.category = None
        if items is not None and "category" in items.columns:
            self.category = items.set_index(ITEM)["category"].dropna().to_dict()
        self.title = None
        if items is not None and "title" in items.columns:
            self.title = items.set_index(ITEM)["title"].dropna().to_dict()
        self._matrix = None

    # ----------------------------------------------------------- lookups
    def user_items(self, df: pd.DataFrame | None = None) -> dict:
        d = self.reference if df is None else df
        return d.groupby(USER)[ITEM].agg(set).to_dict()

    def item_users(self, df: pd.DataFrame | None = None) -> dict:
        d = self.reference if df is None else df
        return d.groupby(ITEM)[USER].agg(set).to_dict()

    def ratings_of(self, col: str) -> dict:
        if not self.has_rating:
            return {}
        return self.reference.groupby(col)[RATING].apply(lambda s: s.dropna().values).to_dict()

    def reviews_of_item(self) -> dict:
        if not self.has_review:
            return {}
        r = self.reference[[ITEM, REVIEW]].dropna()
        r = r[r[REVIEW].astype(str).str.strip() != ""]
        return r.groupby(ITEM)[REVIEW].apply(list).to_dict()

    def interaction_matrix(self):
        """Binary user x item matrix of the reference data (for co-occurrence)."""
        if self._matrix is None:
            users = pd.Index(self.reference[USER].unique())
            items = pd.Index(self.reference[ITEM].unique())
            r = users.get_indexer(self.reference[USER])
            c = items.get_indexer(self.reference[ITEM])
            m = sparse.csr_matrix((np.ones(len(r)), (r, c)), shape=(len(users), len(items)))
            m.data[:] = 1.0
            self._matrix = (m.tocsc(), users, items)
        return self._matrix

    def polarity(self, rating: float, low=None, high=None) -> str:
        span = self.rmax - self.rmin
        low = self.rmin + 0.25 * span if low is None else low
        high = self.rmax - 0.25 * span if high is None else high
        if rating >= high:
            return "positive"
        if rating <= low:
            return "negative"
        return "neutral"


class Perturbation:
    """Base class of every perturbation."""

    name: str = ""
    signal: str = ""                  # interaction | rating | review | general
    observability: str | None = None  # point | contextual | collective | None (generic op)
    operation: str = MODIFY           # add | modify | remove
    default_target: str = "item"
    requires: tuple = ()              # fields required in the data (rating, review_text, timestamp)
    description: str = ""
    DEFAULTS: dict = {}

    def __init__(self, params, budget: int, ctx: PerturbationContext, logger, text_generator=None):
        self.p = _normalize_params(params)
        self.budget = int(budget)
        self.ctx = ctx
        self.rng = ctx.rng
        self.logger = logger
        self.text = text_generator
        self.log_rows: list[dict] = []
        self.fake_users: set = set()
        self.stats: dict = {"budget": self.budget, "target_records": 0, "filler_records": 0}
        self._fake_counter = 0
        self._next_id = None

    # ======================================================== public API
    def check_requirements(self, df: pd.DataFrame):
        missing = []
        for f in self.requires:
            if f == RATING and not self.ctx.has_rating:
                missing.append(f)
            elif f == REVIEW and not self.ctx.has_review:
                missing.append(f)
            elif f == TS and not self.ctx.has_ts:
                missing.append(f)
        if missing:
            raise ValueError(f"Perturbation '{self.name}' requires the field(s) {missing}, "
                             "which are missing from the dataset.")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        self.check_requirements(df)
        self._next_id = int(df.index.max()) + 1 if len(df) else 0
        out = self.perturb(df.copy())
        if self.stats["target_records"] < self.budget:
            self.logger.warning(
                f"[{self.name}] budget not fully used: {self.stats['target_records']}/{self.budget} "
                "records (not enough eligible targets/records with the current settings).")
        return out

    def perturb(self, df: pd.DataFrame) -> pd.DataFrame:  # pragma: no cover - abstract
        raise NotImplementedError

    # ================================================== target selection
    @property
    def target_col(self) -> str:
        t = self.p.get("target", self.default_target)
        if t not in ("user", "item"):
            raise ValueError(f"[{self.name}] target must be 'user' or 'item', got '{t}'")
        return USER if t == "user" else ITEM

    def candidate_nodes(self, eligible: set | None = None, col: str | None = None) -> list:
        """Targets in selection order (uniform / top / bottom), or the explicit id list."""
        col = col or self.target_col
        ids = self.p.get("target_ids")
        if ids:
            if isinstance(ids, str):
                with open(ids) as fh:
                    ids = [line.strip() for line in fh if line.strip()]
            nodes = [str(x) for x in ids]
        else:
            pop = self.ctx.user_pop if col == USER else self.ctx.item_pop
            nodes = ordered_nodes(pop, self.p.get("selection_strategy", "uniform"), self.rng,
                                  self.p.get("top_k"))
        if eligible is not None:
            nodes = [n for n in nodes if n in eligible]
        return nodes

    def per_node(self, cap: float, remaining: int) -> int:
        lo, hi = int(self.p.get("min_per_node", 1)), int(self.p.get("max_per_node", 1))
        n = int(self.rng.integers(lo, max(lo, hi) + 1))
        return int(max(0, min(n, cap, remaining)))

    def select_records(self, df: pd.DataFrame, eligible: pd.Series) -> list:
        """Modify/remove loop: pick record indices target by target until the budget is reached."""
        sub = df[eligible]
        groups = sub.groupby(self.target_col).indices      # positions inside ``sub``
        remaining, chosen = self.budget, []
        for node in self.candidate_nodes(set(groups)):
            if remaining <= 0:
                break
            pos = groups[node]
            n = self.per_node(len(pos), remaining)
            if n <= 0:
                continue
            pick = self.rng.choice(pos, size=n, replace=False)
            chosen.extend(sub.index[pick])
            remaining -= n
        self.stats["target_records"] = len(chosen)
        self.stats["targets"] = int(df.loc[chosen, self.target_col].nunique()) if chosen else 0
        return chosen

    # ============================================================ logging
    def log_modify(self, df, idx, fieldname, new_values):
        old = df.loc[idx, fieldname].tolist()
        for i, o, n in zip(idx, old, new_values):
            self.log_rows.append({"perturbation": self.name, "operation": MODIFY, "record_id": i,
                                  "user_id": df.at[i, USER], "item_id": df.at[i, ITEM],
                                  "field": fieldname, "old_value": o, "new_value": n, "role": "target"})

    def log_remove(self, df, idx):
        for i in idx:
            self.log_rows.append({"perturbation": self.name, "operation": REMOVE, "record_id": i,
                                  "user_id": df.at[i, USER], "item_id": df.at[i, ITEM],
                                  "field": None, "old_value": None, "new_value": None, "role": "target"})

    def append_records(self, df: pd.DataFrame, records: list[dict], role: str = "target") -> pd.DataFrame:
        """Add new records to the split, with fresh record ids, and log them."""
        if not records:
            return df
        new = pd.DataFrame(records)
        new.index = range(self._next_id, self._next_id + len(new))
        self._next_id += len(new)
        for c in df.columns:
            if c not in new.columns:
                new[c] = pd.NA
        new = new[list(df.columns)]
        for c in df.columns:                     # keep the dtypes of the split
            try:
                new[c] = new[c].astype(df[c].dtype)
            except (TypeError, ValueError):
                pass
        for i, r in new.iterrows():
            self.log_rows.append({"perturbation": self.name, "operation": ADD, "record_id": i,
                                  "user_id": r[USER], "item_id": r[ITEM], "field": None,
                                  "old_value": None, "new_value": r.get(RATING, None), "role": role})
        key = "target_records" if role == "target" else "filler_records"
        self.stats[key] += len(new)
        return pd.concat([df, new])

    def new_record(self, user, item, ts=None, rating=None, review=None) -> dict:
        rec = {USER: str(user), ITEM: str(item)}
        if self.ctx.has_ts or ts is not None:
            rec[TS] = pd.NA if ts is None else int(ts)
        if self.ctx.has_rating:
            rec[RATING] = np.nan if rating is None else float(rating)
        if self.ctx.has_review:
            rec[REVIEW] = pd.NA if review is None else review
        return rec

    # ============================================================ ratings
    def rating_range(self) -> tuple[float, float]:
        rb = self.p.get("rating_behavior") or {}
        rmin = rb.get("min_rating") if rb.get("min_rating") is not None else self.ctx.rmin
        rmax = rb.get("max_rating") if rb.get("max_rating") is not None else self.ctx.rmax
        return float(rmin), float(rmax)

    def sample_rating_values(self, n: int, reference=None) -> np.ndarray:
        rb = self.p.get("rating_behavior") or {}
        rmin, rmax = self.rating_range()
        vals = sample_ratings(n, rmin, rmax, self.ctx.rstep, rb.get("sampling_strategy", "uniform"),
                              self.rng, reference)
        return snap_to_scale(vals, self.ctx.rmin, self.ctx.rmax, self.ctx.rstep)

    # ============================================================== time
    def temporal_sampler(self) -> "TemporalSampler":
        return TemporalSampler(self.p.get("temporal_behavior") or {}, self.ctx, self.rng)

    # ========================================================== producers
    def new_fake_user(self) -> str:
        self._fake_counter += 1
        uid = f"fake_{self.name}_{self.ctx.split_name}_{self._fake_counter}"
        self.fake_users.add(uid)
        return uid

    def fake_cfg(self) -> dict:
        return self.p.get("fake_users") or {}

    def use_fake_users(self) -> bool:
        return bool(self.fake_cfg().get("use_fake_users", False))

    def filler_records(self, user: str, exclude: set, behavior: str | None = None,
                       size=None, around_ts: int | None = None) -> list[dict]:
        """
        Filler interactions of a fake user, making the profile plausible in isolation:
          random -> items chosen uniformly at random;
          mimic  -> a real user is used as template: profile length, rating values
                    and timestamps follow the template, and items are drawn
                    proportionally to their popularity (observed interaction patterns).
        """
        behavior = behavior or self.fake_cfg().get("fake_behavior", "random")
        ctx, rng = self.ctx, self.rng
        items = ctx.item_pop.index.to_numpy()
        reviews = self._item_reviews()
        if behavior == "mimic":
            tpl_user = rng.choice(ctx.user_pop.index.to_numpy())
            tpl = ctx.reference[ctx.reference[USER] == tpl_user]
            n = max(0, len(tpl) - len(exclude))
            probs = ctx.item_pop.to_numpy(dtype=float)
            probs /= probs.sum()
            chosen = _sample_excluding(items, n, exclude, rng, probs)
            ratings = rng.choice(tpl[RATING].dropna().to_numpy(), size=len(chosen)) \
                if ctx.has_rating and tpl[RATING].notna().any() else [None] * len(chosen)
            times = rng.choice(tpl[TS].dropna().to_numpy(dtype="int64"), size=len(chosen)) \
                if ctx.has_ts and tpl[TS].notna().any() else [None] * len(chosen)
        elif behavior == "random":
            n = _resolve_size(size if size is not None else self.fake_cfg().get("filler_size", 0),
                              len(items), rng)
            chosen = _sample_excluding(items, n, exclude, rng)
            ratings = self.sample_rating_values(len(chosen)) if ctx.has_rating else [None] * len(chosen)
            times = rng.integers(ctx.ts_min, ctx.ts_max + 1, size=len(chosen)) if ctx.has_ts \
                else [None] * len(chosen)
        else:
            raise ValueError(f"fake_behavior must be 'random' or 'mimic', got '{behavior}'")
        recs = []
        for it, r, t in zip(chosen, ratings, times):
            rv = reviews.get(it)
            recs.append(self.new_record(user, it, t, r, str(rng.choice(rv)) if rv else None))
        return recs

    def _item_reviews(self) -> dict:
        if not hasattr(self, "_reviews_cache"):
            self._reviews_cache = self.ctx.reviews_of_item()
        return self._reviews_cache

    def existing_producers(self, item: str, n: int, item_users: dict) -> list:
        """Existing users who have not interacted with ``item`` yet."""
        taken = item_users.get(item, set())
        pool = self.ctx.user_pop.index.to_numpy()
        return list(_sample_excluding(pool, n, taken, self.rng))

    # ========================================================== additions
    def plan_additions(self, df: pd.DataFrame) -> list[tuple[str, int]]:
        """Add loop: (target, number of records) until the budget is reached."""
        remaining, plan = self.budget, []
        if self.target_col == ITEM:
            if self.use_fake_users():
                cap_fn = lambda node: math.inf
            else:
                iu = self.ctx.item_users(df)
                n_users = len(self.ctx.user_pop)
                cap_fn = lambda node: n_users - len(iu.get(node, ()))
        else:
            ui = self.ctx.user_items(df)
            n_items = len(self.ctx.item_pop)
            cap_fn = lambda node: n_items - len(ui.get(node, ()))
        for node in self.candidate_nodes():
            if remaining <= 0:
                break
            n = self.per_node(cap_fn(node), remaining)
            if n > 0:
                plan.append((node, n))
                remaining -= n
        return plan

    def generate_texts(self, requests: list[dict]) -> list[str]:
        """requests: [{"prompt": str, "polarity": str, "kind": str}] -> generated reviews."""
        if not requests:
            return []
        if self.text is None:
            raise ValueError(f"[{self.name}] needs an llm_configuration to generate reviews")
        lo = self.p.get("min_length_of_review", 1)
        hi = self.p.get("max_length_of_review", 100)
        return self.text.generate(requests, int(lo or 1), int(hi or 100))


class TemporalSampler:
    """
    Temporal scope of the injection: records fall in [start_timestamp, end_timestamp].
    With a non-zero ``window``, all the records of one target fall in a single window
    of that length placed at random inside the interval (a burst).
    """

    def __init__(self, cfg, ctx: PerturbationContext, rng: np.random.Generator):
        self.rng = rng
        self.start = date_to_unix(cfg.get("start_timestamp")) if cfg.get("start_timestamp") else ctx.ts_min
        self.end = date_to_unix(cfg.get("end_timestamp")) if cfg.get("end_timestamp") else ctx.ts_max
        self.window = window_to_seconds(cfg.get("window"))
        if self.start is not None and self.end is not None and self.end < self.start:
            raise ValueError("temporal_behavior: end_timestamp precedes start_timestamp")

    @property
    def active(self) -> bool:
        return self.start is not None and self.end is not None

    def sample(self, n: int) -> list:
        if not self.active:
            return [None] * n
        if self.window <= 0 or self.window >= self.end - self.start:
            return list(self.rng.integers(self.start, self.end + 1, size=n))
        ws = int(self.rng.integers(self.start, self.end - self.window + 1))
        return list(self.rng.integers(ws, ws + self.window + 1, size=n))


# ---------------------------------------------------------------------- utils
def _normalize_params(params) -> dict:
    """Accept the names used in Figure 3 (e.g. max_reviews_per_node) as aliases."""
    p = dict(params)
    for k in list(p):
        if k.startswith("max_") and k.endswith("_per_node") and k != "max_per_node":
            p["max_per_node"] = p.pop(k)
        elif k.startswith("min_") and k.endswith("_per_node") and k != "min_per_node":
            p["min_per_node"] = p.pop(k)
    return p


def _resolve_size(size, total: int, rng) -> int:
    if size is None:
        return 0
    if isinstance(size, (list, tuple)):
        lo, hi = int(size[0]), int(size[1])
        return int(rng.integers(lo, hi + 1))
    if isinstance(size, float) and 0 < size < 1:
        return int(round(size * total))
    return int(size)


def _sample_excluding(pool: np.ndarray, n: int, exclude: set, rng, probs=None) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=pool.dtype)
    mask = ~np.isin(pool, list(exclude)) if exclude else np.ones(len(pool), bool)
    cand = pool[mask]
    if probs is not None:
        probs = probs[mask]
        probs = probs / probs.sum() if probs.sum() > 0 else None
    n = min(n, len(cand))
    return rng.choice(cand, size=n, replace=False, p=probs) if n else cand[:0]
