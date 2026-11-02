"""
Preprocessing and filtering (Section 4.1, phase 2). Applied to the raw data
before splitting: duplicate removal, rating range filtering, review-length
constraints and iterative k-core extraction.
"""

from __future__ import annotations

import pandas as pd

from utils import ITEM, RATING, REVIEW, TS, USER, token_count


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Keep a single record per (user, item): the most recent one when timestamps exist."""
    if TS in df.columns:
        df = df.sort_values(TS, kind="mergesort", na_position="first")
    return df.drop_duplicates(subset=[USER, ITEM], keep="last")


def filter_rating_range(df, min_rating=None, max_rating=None) -> pd.DataFrame:
    """Exclude interactions whose rating falls outside [min_rating, max_rating]."""
    if RATING not in df.columns:
        return df
    mask = pd.Series(True, index=df.index)
    if min_rating is not None:
        mask &= df[RATING] >= float(min_rating)
    if max_rating is not None:
        mask &= df[RATING] <= float(max_rating)
    return df[mask]


def filter_review_length(df, min_tokens: int) -> pd.DataFrame:
    """Discard reviews shorter than ``min_tokens`` whitespace tokens."""
    if REVIEW not in df.columns or not min_tokens:
        return df
    return df[token_count(df[REVIEW]) >= int(min_tokens)]


def kcore(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively remove users and items with fewer than k interactions."""
    if not k or k <= 1:
        return df
    while True:
        u_cnt = df[USER].map(df[USER].value_counts())
        i_cnt = df[ITEM].map(df[ITEM].value_counts())
        keep = (u_cnt >= k) & (i_cnt >= k)
        if keep.all():
            return df
        df = df[keep]
        if df.empty:
            return df


def preprocess(df: pd.DataFrame, cfg, logger=None) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=[USER, ITEM])
    steps = []
    if cfg.drop_duplicates:
        df = drop_duplicates(df); steps.append(("drop_duplicates", len(df)))
    if cfg.min_rating is not None or cfg.max_rating is not None:
        df = filter_rating_range(df, cfg.min_rating, cfg.max_rating); steps.append(("rating_range", len(df)))
    if cfg.min_review_length:
        df = filter_review_length(df, cfg.min_review_length); steps.append(("review_length", len(df)))
    if cfg.kcore:
        df = kcore(df, int(cfg.kcore)); steps.append((f"{cfg.kcore}-core", len(df)))
    if logger:
        trail = ", ".join(f"{s}: {n}" for s, n in steps)
        logger.info(f"Preprocessing: raw={n0} -> {trail or 'no filters'} | "
                    f"users={df[USER].nunique()} items={df[ITEM].nunique()}")
    return df.reset_index(drop=True)
