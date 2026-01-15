
import numpy as np
import pandas as pd
import random
from NoiseInjector.injectors.base import BaseNoiseInjector


class RatingNoiseInjector(BaseNoiseInjector):

    def __init__(self, logger,config):
        self.config = config.noise_config
        self.budget = self.config.budget
        self.logger = logger

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def apply_noise(self, df):
        df = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()

        ctx = self.config.context
        if ctx == "realistic_noise":
            return self._realistic_noise(df)
        elif ctx == "user_burst_noise":
            return self._burst_noise(df, target='user_id')
        elif ctx == "item_burst_noise":
            return self._burst_noise(df, target='item_id')
        elif ctx == "timestamp_corruption":
            return self._timestamp_corruption(df)

        raise ValueError(f"Unknown rating noise context: {ctx}")

    # ==========================================================
    # COMMON HELPERS
    # ==========================================================
    def _ordered_nodes(self, df, target, strategy):
        nodes = df[target].value_counts(ascending=strategy == 'least').index.tolist()
        if strategy == 'uniform':
            random.shuffle(nodes)
        return nodes

    def _sample_ratings(self, base_ratings, n, config):
        if config.rating_behavior.sampling_strategy == 'gaussian':
            mu, sigma = base_ratings.mean(), base_ratings.std() or 1.0
            r = np.random.normal(mu, sigma, size=n)
            return np.clip(np.rint(r), config.rating_behavior.min_rating, config.rating_behavior.max_rating).astype(int)
        return np.random.randint(config.rating_behavior.min_rating, config.rating_behavior.max_rating + 1, size=n)

    def _per_node_budget(self, node_size, total_left,min_ratings_per_node,max_ratings_per_node):
        n = np.random.randint(
            min_ratings_per_node,
            max_ratings_per_node+1
        )
        return min(n, node_size, total_left)

    # ==========================================================
    # REALISTIC NOISE
    # ==========================================================
    def _realistic_noise(self, df):
        config = self.config
        noise_config = self.config.realistic_noise
        target = 'user_id' if config.realistic_noise.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'

        nodes = self._ordered_nodes(df, target, noise_config.selection_strategy)

        if config.realistic_noise.operation == 'remove':
            return self._remove_ratings(df, nodes, target,other, config,noise_config)

        if config.realistic_noise.operation == 'add':
            return self._add_ratings(df, nodes, target, other, config,noise_config)

        raise ValueError(f"Unknown operation: {config.operation}")

    # ==========================================================
    # REMOVE
    # ==========================================================
    def _remove_ratings(self, df, nodes, target, other, config,noise_config):
        removed_idx = []
        remaining = self.budget
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            n = self._per_node_budget(len(node_df), remaining, noise_config.min_ratings_per_node,noise_config.max_ratings_per_node)
            c = 0
            while n <= 0 or n == len(node_df):
                n = self._per_node_budget(len(node_df), remaining, noise_config.min_ratings_per_node,
                                          noise_config.max_ratings_per_node)
                c+=1
                if c == 20: # too many iterations, change node
                    continue
            if noise_config.preserve_degree_distribution:
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))


            start_ts = noise_config.temporal_interval.start_timestamp
            start_ts = pd.to_datetime(start_ts, unit='s')

            end_ts = noise_config.temporal_interval.end_timestamp
            end_ts = pd.to_datetime(end_ts, unit='s')

            ratings = self._sample_ratings(node_df['rating'], n, noise_config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
                ]

            if start_ts != 0 and end_ts != 0:

                candidates = node_df[
                    (node_df['timestamp'] >= start_ts) &  # timestamp >= start
                    (node_df['timestamp'] <= end_ts)  # timestamp <= end
                    ]

            sampled = candidates.sample(
                n=min(len(candidates), n),
                replace=False
            )
            removed_idx.extend(sampled.index)
            remaining -= len(sampled)

        removed = df.loc[removed_idx]
        return df.drop(removed_idx), removed

    # ==========================================================
    # ADD (shared by realistic + burst)
    # ==========================================================
    def _add_ratings(self, df, nodes, target, other, config,noise_config):
        rows = []
        remaining = self.budget
        all_other = df[other].unique()

        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            used = node_df[other].unique()
            available = all_other
            if config.avoid_duplicates:
                available = np.setdiff1d(all_other, used)

            if len(available) == 0:
                continue

            n = self._per_node_budget(len(node_df), remaining, noise_config.min_ratings_per_node,noise_config.max_ratings_per_node)

            if getattr(noise_config, 'preserve_degree_distribution', None):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            sampled_other = np.random.choice(available, size=n, replace=False)
            sampled_ratings = self._sample_ratings(node_df['rating'], n, noise_config)

            start_ts = noise_config.temporal_interval.start_timestamp
            end_ts = noise_config.temporal_interval.end_timestamp


            if start_ts == 0 and end_ts == 0:
                now_ts = int(pd.Timestamp.now().timestamp())  # ora corrente in secondi
                timestamps = np.random.randint(now_ts - 365 * 24 * 3600, now_ts + 1, size=n)
            else:
                timestamps = np.random.randint(start_ts, end_ts + 1, size=n)

            timestamps = pd.to_datetime(timestamps, unit='s')  # se i timestamp sono in millisecondi

            rows.append(pd.DataFrame({
                target: node,
                other: sampled_other,
                'rating': sampled_ratings,
                'timestamp': timestamps,
                'noise': True
            }))

            remaining -= n
        df['noise'] = False
        added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
        return pd.concat([df, added], ignore_index=True), added

    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df, target):
        config = self.config
        if target == 'item_id':
            noise_config = self.config.item_burst_noise
        else:
            noise_config = self.config.user_burst_noise

        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, noise_config.selection_strategy)
        return self._add_ratings(df, nodes, target, other, config,noise_config)

    # ==========================================================
    # TIMESTAMP CORRUPTION
    # ==========================================================
    def _timestamp_corruption(self, df):
        config = self.config
        noise_config = self.config.timestamp_corruption
        nodes = self._ordered_nodes(
            df,
            'user_id' if noise_config.target == 'user' else 'item_id',
            noise_config.selection_strategy
        )

        remaining = self.budget
        df['noise'] = False
        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df['user_id'] == node]
            n = self._per_node_budget(len(node_df), remaining, noise_config.min_ratings_per_node,noise_config.max_ratings_per_node)
            sampled = node_df.sample(n=n, replace=False)

            df.loc[sampled.index, 'timestamp'] = self._corrupt(sampled,noise_config,config)
            df.loc[sampled.index, 'noise'] = True
            remaining -= n

        return df, None

    def _corrupt(self, df,noise_config,config):
        tb = noise_config.temporal_behavior
        days_map = {"low": 7, "medium": 30, "high": 365}
        max_days = np.random.randint(1, 10) * days_map.get(tb.intensity, 30)

        if tb.corruption_mode == "uniform":
            shifts = np.random.randint(-max_days, max_days + 1, size=len(df))
        elif tb.corruption_mode == "forward":
            shifts = np.random.randint(1, max_days + 1, size=len(df))
        elif tb.corruption_mode == "backward":
            shifts = -np.random.randint(1, max_days + 1, size=len(df))
        else:
            raise ValueError(f"Unknown corruption_mode: {tb.corruption_mode}")

        return pd.to_datetime(df["timestamp"]) + pd.to_timedelta(shifts, unit="D")
