
import numpy as np
import pandas as pd
import random
from src.injectors.base import BaseNoiseInjector

from src.utils import *
class RatingNoiseInjector(BaseNoiseInjector):

    def __init__(self, logger,config):
        self.config = config.noise_config
        self.budget = self.config.budget
        self.logger = logger

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def apply_noise(self, df, df_val, df_test):
        df = df[['user_id', 'item_id', 'rating', 'timestamp']].copy()

        ctx = self.config.context
        if ctx == "realistic_noise":
            return self._realistic_noise(df)
        elif ctx == "user_burst_noise":
            return self._burst_noise(df,df_val, df_test)
        # elif ctx == "item_burst_noise":
        #     return self._burst_noise(df,df_val, df_test, target='item_id')
        elif ctx == "timestamp_corruption":
            return self._timestamp_corruption(df)

        raise ValueError(f"Unknown rating noise context: {ctx}")

    # ==========================================================
    # REALISTIC NOISE
    # ==========================================================
    def _realistic_noise(self, df):
        config = self.config
        noise_config = self.config.realistic_noise
        target = 'user_id' if config.realistic_noise.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'

        nodes = ordered_nodes(df, target, noise_config.selection_strategy)

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
        start_ts = noise_config.temporal_interval.start_timestamp
        end_ts = noise_config.temporal_interval.end_timestamp
        start_ts = parse_timestamp(start_ts)
        end_ts = parse_timestamp(end_ts)

        grouped = df.groupby(target)
        for node in nodes:
            if remaining <= 0 or node not in grouped.groups:
                continue

            #node_df = df[df[target] == node]
            # n = per_node_budget(len(node_df), remaining, noise_config.min_ratings_per_node,noise_config.max_ratings_per_node)
            # if n <= 0:
            #     continue
            node_df = grouped.get_group(node)
            n = per_node_budget(len(node_df), remaining,
                                noise_config.min_ratings_per_node,
                                noise_config.max_ratings_per_node)
            if n <= 0:
                continue

            if noise_config.preserve_degree_distribution:
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))



            ratings = sample_ratings(node_df['rating'], n, noise_config)
            candidates = node_df[node_df['rating'].isin(ratings)]

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
    def _add_ratings_0(self, df, df_val, df_test, nodes, target, other, config, noise_config):
        """
        Fully vectorized version: generates all noisy ratings at once without looping over nodes.
        """
        remaining = self.budget
        all_other = df[other].unique()

        # Precompute used sets for all nodes
        used_dict = {node: set(df[df[target] == node][other].unique()) for node in nodes}
        if df_val is not None:
            for node in nodes:
                used_dict[node].update(df_val[df_val[target] == node][other].unique())
        if df_test is not None:
            for node in nodes:
                used_dict[node].update(df_test[df_test[target] == node][other].unique())

        # Randomly assign number of ratings per node
        n_per_node = np.random.randint(noise_config.min_ratings_per_node,
                                       noise_config.max_ratings_per_node + 1,
                                       size=len(nodes))

        # Adjust by degree distribution if needed
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()
        if getattr(noise_config, 'preserve_degree_distribution', None):
            factors = np.array([(max_degree - degrees.get(node, 0)) / max_degree for node in nodes])
            n_per_node = np.maximum(1, np.ceil(n_per_node * factors).astype(int))

        # Apply remaining budget
        total = n_per_node.sum()
        if total > remaining:
            scale = remaining / total
            n_per_node = np.floor(n_per_node * scale).astype(int)

        # Remove nodes with zero ratings
        valid_idx = n_per_node > 0
        nodes = np.array(nodes)[valid_idx]
        n_per_node = n_per_node[valid_idx]

        # Prepare final arrays
        all_targets, all_others, all_ratings = [], [], []

        # Generate all ratings and other IDs vectorized
        for node, n in zip(nodes, n_per_node):
            # Available "other" nodes
            if getattr(noise_config, 'avoid_duplicates', False):
                available = np.array([x for x in all_other if x not in used_dict[node]])
            else:
                available = all_other

            if len(available) == 0:
                continue

            n = min(n, len(available))
            sampled_other = np.random.choice(available, size=n, replace=False)
            node_ratings = df[df[target] == node]['rating']
            if len(node_ratings) == 0:
                sampled_ratings = np.random.randint(1, 6, size=n)
            else:
                sampled_ratings = sample_ratings(node_ratings, n, noise_config)

            all_targets.extend([node] * n)
            all_others.extend(sampled_other)
            all_ratings.extend(sampled_ratings)

        if len(all_targets) == 0:
            df['noise'] = False
            return df, pd.DataFrame(columns=df.columns)

        # Generate timestamps vectorized
        start_ts = parse_timestamp(noise_config.temporal_interval.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_interval.end_timestamp)

        if start_ts == 0 and end_ts == 0:
            timestamps = np.full(len(all_targets), int(pd.Timestamp.now().timestamp()))
        else:
            timestamps = (
                        (start_ts + (end_ts - start_ts) * np.random.rand(len(all_targets))).astype('int64') // 10 ** 9)

        # Create final DataFrame
        added = pd.DataFrame({
            target: all_targets,
            other: all_others,
            'rating': all_ratings,
            'timestamp': timestamps,
            'noise': True
        })

        df['noise'] = False
        return pd.concat([df, added], ignore_index=True), added

    def _add_ratings(self, df, df_val, df_test, nodes, target, other, config,noise_config):
        rows = []
        remaining = self.budget
        all_other = df[other].unique()
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()
        grouped = df.groupby(target)
        start_ts = noise_config.temporal_interval.start_timestamp
        end_ts = noise_config.temporal_interval.end_timestamp
        start_ts = parse_timestamp(start_ts)
        end_ts = parse_timestamp(end_ts)

        for node in nodes:
            if remaining <= 0:
                break

            node_df = grouped.get_group(node) if node in grouped.groups else pd.DataFrame(columns=df.columns)
            used = node_df[other].unique() if not node_df.empty else np.array([])
            if df_val is not None and df_test is not None:
                node_df_val = df_val.groupby(target).get_group(node) if node in df_val.groupby(target).groups else pd.DataFrame(columns=df.columns)
                node_df_test = df_test.groupby(target).get_group(node) if node in df_test.groupby(target).groups else pd.DataFrame(columns=df.columns)
                used_val = node_df_val[other].unique() if not node_df_val.empty else np.array([])
                used_test = node_df_test[other].unique() if not node_df_test.empty else np.array([])
                used = np.concatenate((used, used_val, used_test))
            available = np.setdiff1d(all_other, used) if getattr(noise_config, 'avoid_duplicates', False) else all_other
            if len(available) == 0:
                continue

            n = np.random.randint(noise_config.min_ratings_per_node,noise_config.max_ratings_per_node + 1)
            if n <= 0:
                continue

            if getattr(noise_config, 'preserve_degree_distribution', None):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            sampled_other = np.random.choice(available, size=n, replace=False)
            sampled_ratings = sample_ratings(node_df['rating'], n, noise_config)

            if start_ts == 0 and end_ts == 0:
                timestamps = [int(pd.Timestamp.now().timestamp()) for _ in range(n)]
            else:
                timestamps = [int((start_ts + (end_ts - start_ts) * random.random()).timestamp()) for _ in range(n)]

            rows.append(pd.DataFrame({
                target: node,
                other: sampled_other,
                'rating': sampled_ratings,
                'timestamp': timestamps,
                'noise': True
            }))

            remaining -= n
            print(remaining)
        df['noise'] = False
        added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
        return pd.concat([df, added], ignore_index=True), added

    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df,df_val, df_test, target):
        noise_config = self.config.rating_burst_noise
        config = self.config
        # if target == 'item_id':
        #     noise_config = self.config.item_burst_noise
        # else:
        #     noise_config = self.config.user_burst_noise
        target = noise_config.target
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        return self._add_ratings(df,df_val,df_test, nodes, target, other, config,noise_config)

    # ==========================================================
    # TIMESTAMP CORRUPTION
    # ==========================================================
    def _timestamp_corruption(self, df):
        noise_config = self.config.timestamp_corruption
        target_col = 'user_id' if noise_config.target == 'user' else 'item_id'
        nodes = ordered_nodes(df, target_col, noise_config.selection_strategy)

        remaining = self.budget
        grouped = df.groupby(target_col)
        df['noise'] = False
        for node in nodes:
            if remaining <= 0 or node not in grouped.groups:
                continue

            node_df = grouped.get_group(node)
            n = per_node_budget(len(node_df), remaining,
                                noise_config.min_ratings_per_node,
                                noise_config.max_ratings_per_node)
            if n <= 0:
                continue

            sampled = node_df.sample(n=n, replace=False)

            df.loc[sampled.index, 'timestamp'] = self._corrupt(sampled,noise_config)
            df.loc[sampled.index, 'noise'] = True
            remaining -= n

        return df, None

    @staticmethod
    def _corrupt(df,noise_config):
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
