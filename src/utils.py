import pandas as pd
import numpy as np
import random

def parse_timestamp(ts):
    # se è già stringa data

    if ts == 0:
        return 0

    if isinstance(ts, str):
        return pd.to_datetime(ts, errors="raise")

    # se è numerico
    ts = int(ts)

    if ts < 1e11:          # ~ seconds (1970–5138)
        return pd.to_datetime(ts, unit="s")
    elif ts < 1e14:        # milliseconds
        return pd.to_datetime(ts, unit="ms")
    elif ts < 1e17:        # microseconds
        return pd.to_datetime(ts, unit="us")
    else:                  # nanoseconds
        return pd.to_datetime(ts, unit="ns")


def per_node_budget(node_size, total_left,min_per_node,max_per_node):
    n = np.random.randint(
        min_per_node,
        max_per_node+1
    )
    return min(n, node_size-1, total_left)

def sample_ratings(base_ratings, n, config):
    if config.rating_behavior.sampling_strategy == 'gaussian':
        mu, sigma = base_ratings.mean(), base_ratings.std() or 1.0
        r = np.random.normal(mu, sigma, size=n)
        return np.clip(np.rint(r), config.rating_behavior.min_rating, config.rating_behavior.max_rating).astype(int)
    return np.random.randint(config.rating_behavior.min_rating, config.rating_behavior.max_rating + 1, size=n)

def sample_reviews(df, config):
    if config.min_length_of_review > 0:
        token_counts = df['review_text'].fillna("").str.split().str.len()
        # Filtra
        df_filtered = df[token_counts >= config.min_length_of_review].copy()
        return df_filtered
    return df

def ordered_nodes(df, target, strategy):
    nodes = df[target].value_counts(ascending=strategy == 'least').index.tolist()
    if strategy == 'uniform':
        random.shuffle(nodes)
    return nodes