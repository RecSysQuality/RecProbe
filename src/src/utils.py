"""General helpers shared by the RecProbe modules."""

from __future__ import annotations

import random

import numpy as np
import pandas as pd

USER, ITEM, TS, RATING, REVIEW = "user_id", "item_id", "timestamp", "rating", "review_text"
CORE_FIELDS = (USER, ITEM, TS, RATING, REVIEW)


# --------------------------------------------------------------------------- #
# Reproducibility                                                             #
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, verbose: bool = True) -> None:
    """Seed Python, NumPy and (if installed) PyTorch/CUDA."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    if verbose:
        print(f"Random seed set to {seed}.")


# --------------------------------------------------------------------------- #
# Timestamps                                                                  #
# --------------------------------------------------------------------------- #
def parse_timestamp(ts):
    """Single value -> pandas Timestamp (seconds/ms/us/ns or date strings)."""
    if ts is None or (isinstance(ts, float) and np.isnan(ts)):
        return pd.NaT
    if isinstance(ts, pd.Timestamp):
        return ts
    if isinstance(ts, str):
        try:
            ts = float(ts)
        except ValueError:
            return pd.to_datetime(ts, errors="raise")
    ts = int(ts)
    if abs(ts) < 1e11:
        return pd.to_datetime(ts, unit="s")
    if abs(ts) < 1e14:
        return pd.to_datetime(ts, unit="ms")
    if abs(ts) < 1e17:
        return pd.to_datetime(ts, unit="us")
    return pd.to_datetime(ts, unit="ns")


def to_unix_seconds(series: pd.Series) -> pd.Series:
    """Vectorised conversion of any timestamp column to Unix seconds (Int64)."""
    if series.empty:
        return series.astype("Int64")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().mean() > 0.95:
        mag = numeric.abs().median()
        div = 1 if mag < 1e11 else 1e3 if mag < 1e14 else 1e6 if mag < 1e17 else 1e9
        return (numeric / div).round().astype("Int64")
    parsed = pd.to_datetime(series, errors="coerce", utc=True)
    out = pd.Series(pd.NA, index=series.index, dtype="Int64")
    mask = parsed.notna()
    # independent of the datetime resolution chosen by pandas (ns, us, s)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    out[mask] = ((parsed[mask] - epoch) // pd.Timedelta(seconds=1)).astype("Int64")
    return out


def date_to_unix(value) -> int | None:
    """'2018-01-01' / 1514764800 / Timestamp -> Unix seconds."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float, np.integer)) or (isinstance(value, str) and value.lstrip("-").isdigit()):
        return int(parse_timestamp(int(value)).timestamp())
    ts = pd.Timestamp(value)          # YAML dates arrive as datetime.date
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    return int(ts.timestamp())


def window_to_seconds(window) -> int:
    """{'days':0,'hours':10,'minutes':0} -> seconds."""
    if window is None:
        return 0
    if isinstance(window, (int, float)):
        return int(window)
    return int(
        window.get("days", 0) * 86400
        + window.get("hours", 0) * 3600
        + window.get("minutes", 0) * 60
        + window.get("seconds", 0)
    )


# --------------------------------------------------------------------------- #
# Ratings                                                                     #
# --------------------------------------------------------------------------- #
def snap_to_scale(values, rmin: float, rmax: float, step: float | None) -> np.ndarray:
    """Clip to the valid range and map to the rating granularity."""
    values = np.asarray(values, dtype=float)
    values = np.clip(values, rmin, rmax)
    if step:
        values = rmin + np.round((values - rmin) / step) * step
        values = np.clip(values, rmin, rmax)
    return values


def sample_ratings(n: int, rmin: float, rmax: float, step: float | None, strategy: str,
                   rng: np.random.Generator, reference=None) -> np.ndarray:
    """
    Sample ``n`` ratings (Section 4.1, defect injection):
      * uniform  -> uniformly from [rmin, rmax] on the rating grid;
      * gaussian -> N(mean, std) of the ``reference`` ratings (target user/item),
                    then clipped and mapped to the original rating scale.
    """
    if n <= 0:
        return np.array([], dtype=float)
    if strategy == "gaussian":
        ref = np.asarray(reference if reference is not None else [], dtype=float)
        ref = ref[np.isfinite(ref)]
        mu = ref.mean() if ref.size else (rmin + rmax) / 2
        sigma = ref.std() if ref.size > 1 and ref.std() > 0 else 1.0
        return snap_to_scale(rng.normal(mu, sigma, size=n), rmin, rmax, step)
    if step:
        grid = np.arange(rmin, rmax + step / 2, step)
        return rng.choice(grid, size=n)
    return rng.uniform(rmin, rmax, size=n)


def infer_rating_scale(ratings: pd.Series) -> tuple[float, float, float | None]:
    r = pd.to_numeric(ratings, errors="coerce").dropna()
    if r.empty:
        return 1.0, 5.0, 1.0
    rmin, rmax = float(r.min()), float(r.max())
    vals = np.unique(r.values)
    if np.allclose(vals, np.round(vals)):
        step = 1.0
    elif np.allclose(vals * 2, np.round(vals * 2)):
        step = 0.5
    else:
        step = None
    return rmin, rmax, step


# --------------------------------------------------------------------------- #
# Misc                                                                        #
# --------------------------------------------------------------------------- #
def token_count(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str).str.split().str.len()


def resolve_budget(budget, n_edges: int) -> int:
    """
    Budget = number of interactions to add or corrupt (Section 4.1).
    Either an absolute number (int >= 1) or a fraction of the training edges
    (float in (0, 1], or a string such as "30%").
    """
    if isinstance(budget, str):
        b = budget.strip()
        if b.endswith("%"):
            return int(round(float(b[:-1]) / 100 * n_edges))
        budget = float(b)
    if isinstance(budget, float) and 0 < budget <= 1:
        return int(round(budget * n_edges))
    return int(budget)


def ordered_nodes(popularity: pd.Series, strategy: str, rng: np.random.Generator,
                  top_k: int | None = None) -> list:
    """
    Order users/items for target selection:
      uniform -> random order;  top -> most popular first;  bottom -> least popular first.
    """
    strategy = {"least": "bottom", "most": "top", "random": "uniform"}.get(strategy, strategy)
    if strategy == "uniform":
        nodes = list(popularity.index)
        rng.shuffle(nodes)
        return nodes
    if strategy not in ("top", "bottom"):
        raise ValueError(f"Unknown selection_strategy '{strategy}' (uniform | top | bottom)")
    # stable sort with random tie-breaking
    shuffled = popularity.sample(frac=1.0, random_state=int(rng.integers(2**31)))
    ordered = shuffled.sort_values(ascending=(strategy == "bottom"), kind="mergesort")
    nodes = list(ordered.index)
    return nodes[:top_k] if top_k else nodes
