
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
        if ctx == "random_inconsistencies":
            return self._realistic_noise(df,df_val, df_test)
        elif ctx == "rating_burst":
            return self._burst_noise(df,df_val, df_test)
        elif ctx == "timestamp_corruption":
            return self._timestamp_corruption(df)

        raise ValueError(f"Unknown rating noise context: {ctx}")

    # ==========================================================
    # REALISTIC NOISE
    # ==========================================================
    def _realistic_noise(self, df,df_val,df_test):
        config = self.config
        noise_config = self.config.random_inconsistencies
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'

        nodes = ordered_nodes(df, target, noise_config.selection_strategy)

        if noise_config.operation == 'remove':
            return self._remove_ratings(df,df_val, df_test, nodes, target,other, config,noise_config)

        if noise_config.operation == 'add':
            return self._add_ratings(df, df_val, df_test, nodes, target, other, config,noise_config)

        raise ValueError(f"Unknown operation: {config.operation}")

    # ==========================================================
    # REMOVE
    # ==========================================================
    # def _remove_ratings(self, df, df_val, df_test, nodes, target, other, config, noise_config):
    #     remaining = self.budget
    #
    #     # ---------- Precompute
    #     degrees = df.groupby(target)[other].nunique().to_dict()
    #     max_degree = max(degrees.values()) if degrees else 1
    #
    #     start_ts = parse_timestamp(noise_config.temporal_behavior.start_timestamp)
    #     end_ts = parse_timestamp(noise_config.temporal_behavior.end_timestamp)
    #
    #     # group indices (velocissimo)
    #     group_indices = df.groupby(target).indices
    #
    #     # rating array per nodo
    #     rating_map = df.groupby(target)['rating'].apply(np.array).to_dict()
    #     timestamp_arr = df['timestamp'].values
    #
    #     removed_idx = []
    #
    #     for node in nodes:
    #         if remaining <= 0:
    #             break
    #
    #         idx = group_indices.get(node)
    #         if idx is None:
    #             continue
    #
    #         node_ratings = rating_map.get(node, np.array([]))
    #         node_timestamps = timestamp_arr[idx]
    #
    #         n = per_node_budget(
    #             len(idx),
    #             remaining,
    #             noise_config.min_ratings_per_node,
    #             noise_config.max_ratings_per_node
    #         )
    #
    #         if n <= 0:
    #             continue
    #
    #         if getattr(noise_config, 'preserve_degree_distribution', None):
    #             factor = (max_degree - degrees.get(node, 0)) / max_degree
    #             n = max(1, int(np.ceil(n * factor)))
    #
    #         n = min(n, len(idx), remaining)
    #         if n <= 0:
    #             continue
    #
    #         # sample ratings (array numpy)
    #         sampled_ratings = sample_ratings(node_ratings, n, noise_config)
    #
    #         # ---------- candidates: boolean mask
    #         mask = np.isin(node_ratings, sampled_ratings)
    #
    #         if start_ts != 0 and end_ts != 0:
    #             mask &= (node_timestamps >= start_ts) & (node_timestamps <= end_ts)
    #
    #         candidate_idx = np.array(idx)[mask]
    #
    #         if len(candidate_idx) == 0:
    #             continue
    #
    #         # campiona n tra i candidati
    #         sampled_idx = np.random.choice(candidate_idx, size=min(len(candidate_idx), n), replace=False)
    #
    #         removed_idx.extend(sampled_idx)
    #         remaining -= len(sampled_idx)
    #
    #     removed_idx = np.array(removed_idx)
    #     removed = df.iloc[removed_idx]
    #     df_new = df.drop(removed_idx)
    #
    #     return df_new, removed

    def _remove_ratings(self, df, df_val, df_test, nodes, target, other, config, noise_config):
        removed_idx = []
        remaining = self.budget

        # Calcolo gradi una sola volta
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()
        degrees_dict = degrees.to_dict()

        # Timestamp una sola volta
        start_ts = parse_timestamp(noise_config.temporal_behavior.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_behavior.end_timestamp)

        # Maschera per timestamp, se presente
        if start_ts != 0 and end_ts != 0:
            time_mask = (df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)
        else:
            time_mask = pd.Series(True, index=df.index)

        # Iterazione sui nodi
        for node in nodes:
            if remaining <= 0 or node not in degrees_dict:
                continue

            node_mask = df[target] == node
            node_df = df[node_mask]

            n = per_node_budget(len(node_df), remaining,
                                noise_config.min_ratings_per_node,
                                noise_config.max_ratings_per_node)
            if n <= 0:
                continue

            if noise_config.preserve_degree_distribution:
                factor = (max_degree - degrees_dict.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            # Selezione delle rating candidate
            ratings = sample_ratings(node_df['rating'], n, noise_config)
            candidates_mask = node_mask & df['rating'].isin(ratings) & time_mask
            candidates_idx = df.index[candidates_mask]

            # Campionamento
            sampled_idx = np.random.choice(
                candidates_idx, size=min(len(candidates_idx), n), replace=False
            )

            removed_idx.extend(sampled_idx)
            remaining -= len(sampled_idx)

        removed = df.loc[removed_idx]
        return df.drop(removed_idx), removed

    # def _remove_ratings(self, df,df_val,df_test, nodes, target, other, config,noise_config):
    #     removed_idx = []
    #     remaining = self.budget
    #     degrees = df.groupby(target)[other].nunique()
    #     max_degree = degrees.max()
    #     start_ts = noise_config.temporal_behavior.start_timestamp
    #     end_ts = noise_config.temporal_behavior.end_timestamp
    #     start_ts = parse_timestamp(start_ts)
    #     end_ts = parse_timestamp(end_ts)
    #
    #     grouped = df.groupby(target)
    #     for node in nodes:
    #         if remaining <= 0 or node not in grouped.groups:
    #             continue
    #
    #         node_df = grouped.get_group(node)
    #         n = per_node_budget(len(node_df), remaining,
    #                             noise_config.min_ratings_per_node,
    #                             noise_config.max_ratings_per_node)
    #         if n <= 0:
    #             continue
    #
    #         if noise_config.preserve_degree_distribution:
    #             factor = (max_degree - degrees.get(node, 0)) / max_degree
    #             n = max(1, int(np.ceil(n * factor)))
    #
    #
    #
    #         ratings = sample_ratings(node_df['rating'], n, noise_config)
    #         candidates = node_df[node_df['rating'].isin(ratings)]
    #
    #         if start_ts != 0 and end_ts != 0:
    #
    #             candidates = node_df[
    #                 (node_df['timestamp'] >= start_ts) &  # timestamp >= start
    #                 (node_df['timestamp'] <= end_ts)  # timestamp <= end
    #                 ]
    #
    #         sampled = candidates.sample(
    #             n=min(len(candidates), n),
    #             replace=False
    #         )
    #         removed_idx.extend(sampled.index)
    #         remaining -= len(sampled)
    #
    #     removed = df.loc[removed_idx]
    #     return df.drop(removed_idx), removed

    def _add_ratings(self, df, df_val, df_test, nodes, target, other, config, noise_config):
        remaining = self.budget

        # ---------- PRECOMPUTE (una sola volta)
        all_other = df[other].unique()
        all_other_set = set(all_other)

        # degrees
        degrees = df.groupby(target)[other].nunique().to_dict()
        max_degree = max(degrees.values()) if degrees else 1

        # Mappa nodo -> set(other)
        used_map = df.groupby(target)[other].agg(set).to_dict()

        if df_val is not None:
            val_map = df_val.groupby(target)[other].agg(set).to_dict()
            for k, v in val_map.items():
                used_map.setdefault(k, set()).update(v)

        if df_test is not None:
            test_map = df_test.groupby(target)[other].agg(set).to_dict()
            for k, v in test_map.items():
                used_map.setdefault(k, set()).update(v)

        # rating history per nodo (serve per sample_ratings)
        rating_map = df.groupby(target)['rating'].apply(np.array).to_dict()

        # timestamp
        start_ts = parse_timestamp(noise_config.temporal_behavior.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_behavior.end_timestamp)
        now_ts = int(pd.Timestamp.now().timestamp())

        # ---------- ACCUMULATORI (no DataFrame nel loop)
        target_col = []
        other_col = []
        rating_col = []
        timestamp_col = []
        noise_col = []

        for node in nodes:
            if remaining <= 0:
                break

            used = used_map.get(node, set())

            if getattr(config, 'avoid_duplicates', False):
                available = list(all_other_set - used)
            else:
                available = all_other

            if not available:
                continue

            n = np.random.randint(
                noise_config.min_ratings_per_node,
                noise_config.max_ratings_per_node + 1
            )

            if getattr(config, 'preserve_degree_distribution', None):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            n = min(n, len(available), remaining)

            if n <= 0:
                continue

            sampled_other = np.random.choice(available, size=n, replace=False)

            sampled_ratings = sample_ratings(
                rating_map.get(node, np.array([])),
                n,
                noise_config
            )

            if start_ts == 0 and end_ts == 0:
                timestamps = np.full(n, now_ts, dtype=np.int64)
            else:
                rand = np.random.random(n)
                timestamps = (
                        start_ts + (end_ts - start_ts) * rand
                ).astype(np.int64)

            # append vettoriale
            target_col.extend([node] * n)
            other_col.extend(sampled_other)
            rating_col.extend(sampled_ratings)
            timestamp_col.extend(timestamps)
            noise_col.extend([True] * n)

            remaining -= n

        # ---------- COSTRUZIONE FINALE
        df['noise'] = False

        if target_col:
            added = pd.DataFrame({
                target: target_col,
                other: other_col,
                'rating': rating_col,
                'timestamp': timestamp_col,
                'noise': noise_col
            })
            result = pd.concat([df, added], ignore_index=True)
        else:
            added = pd.DataFrame(columns=df.columns)
            result = df

        return result, added

    # def _add_ratings0(self, df, df_val, df_test, nodes, target, other, config,noise_config):
    #     rows = []
    #     remaining = self.budget
    #     all_other = df[other].unique()
    #     degrees = df.groupby(target)[other].nunique()
    #     max_degree = degrees.max()
    #     grouped = df.groupby(target)
    #     start_ts = noise_config.temporal_behavior.start_timestamp
    #     end_ts = noise_config.temporal_behavior.end_timestamp
    #     start_ts = parse_timestamp(start_ts)
    #     end_ts = parse_timestamp(end_ts)
    #
    #     for node in nodes:
    #         if remaining <= 0:
    #             break
    #
    #         node_df = grouped.get_group(node) if node in grouped.groups else pd.DataFrame(columns=df.columns)
    #         used = node_df[other].unique() if not node_df.empty else np.array([])
    #         if df_val is not None and df_test is not None:
    #             node_df_val = df_val.groupby(target).get_group(node) if node in df_val.groupby(target).groups else pd.DataFrame(columns=df.columns)
    #             node_df_test = df_test.groupby(target).get_group(node) if node in df_test.groupby(target).groups else pd.DataFrame(columns=df.columns)
    #             used_val = node_df_val[other].unique() if not node_df_val.empty else np.array([])
    #             used_test = node_df_test[other].unique() if not node_df_test.empty else np.array([])
    #             used = np.concatenate((used, used_val, used_test))
    #         available = np.setdiff1d(all_other, used) if getattr(config, 'avoid_duplicates', False) else all_other
    #         if len(available) == 0:
    #             continue
    #
    #         n = np.random.randint(noise_config.min_ratings_per_node,noise_config.max_ratings_per_node + 1)
    #         if n <= 0:
    #             continue
    #
    #         if getattr(config, 'preserve_degree_distribution', None):
    #             factor = (max_degree - degrees.get(node, 0)) / max_degree
    #             n = max(1, int(np.ceil(n * factor)))
    #
    #         sampled_other = np.random.choice(available, size=n, replace=False)
    #         sampled_ratings = sample_ratings(node_df['rating'], n, noise_config)
    #
    #         if start_ts == 0 and end_ts == 0:
    #             timestamps = [int(pd.Timestamp.now().timestamp()) for _ in range(n)]
    #         else:
    #             timestamps = [int((start_ts + (end_ts - start_ts) * random.random()).timestamp()) for _ in range(n)]
    #
    #         rows.append(pd.DataFrame({
    #             target: node,
    #             other: sampled_other,
    #             'rating': sampled_ratings,
    #             'timestamp': timestamps,
    #             'noise': True
    #         }))
    #
    #         remaining -= n
    #         print(remaining)
    #     df['noise'] = False
    #     added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
    #     return pd.concat([df, added], ignore_index=True), added

    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df,df_val, df_test):
        noise_config = self.config.rating_burst
        config = self.config

        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        return self._add_ratings(df,df_val,df_test, nodes, target, other, config,noise_config)

    # # ==========================================================
    # # TIMESTAMP CORRUPTION
    # # ==========================================================
    # def _timestamp_corruption(self, df):
    #     noise_config = self.config.timestamp_corruption
    #     target_col = 'user_id' if noise_config.target == 'user' else 'item_id'
    #
    #     df = df.copy()
    #     df['noise'] = False
    #
    #     nodes = ordered_nodes(df, target_col, noise_config.selection_strategy)
    #
    #     remaining = self.budget
    #
    #     # ---------- PRECOMPUTE GROUP INDICES (velocissimo)
    #     group_indices = df.groupby(target_col).indices
    #
    #     # ---------- PRECONVERT TIMESTAMP UNA VOLTA
    #     timestamps = pd.to_datetime(df["timestamp"]).values
    #
    #     # accumulatori
    #     corrupted_indices = []
    #
    #     for node in nodes:
    #         if remaining <= 0:
    #             break
    #
    #         idx = group_indices.get(node)
    #         if idx is None:
    #             continue
    #
    #         group_size = len(idx)
    #
    #         n = per_node_budget(
    #             group_size,
    #             remaining,
    #             noise_config.min_ratings_per_node,
    #             noise_config.max_ratings_per_node
    #         )
    #
    #         if n <= 0:
    #             continue
    #
    #         # 🔥 numpy sampling invece di pandas.sample
    #         sampled_idx = np.random.choice(idx, size=n, replace=False)
    #
    #         corrupted_indices.append(sampled_idx)
    #         remaining -= n
    #
    #     if not corrupted_indices:
    #         return df, df.iloc[[]]
    #
    #     # ---------- CONCATENA UNA VOLTA SOLA
    #     corrupted_indices = np.concatenate(corrupted_indices)
    #
    #     # ---------- SHIFT VETTORIALE
    #     shifts = self._corrupt_vectorized(len(corrupted_indices), noise_config)
    #
    #     timestamps[corrupted_indices] = (
    #             timestamps[corrupted_indices] + shifts
    #     )
    #
    #     df.loc[corrupted_indices, "timestamp"] = timestamps[corrupted_indices]
    #     df.loc[corrupted_indices, "noise"] = True
    #
    #     return df, df.loc[corrupted_indices]
    #
    # def _corrupt_vectorized(self, size, noise_config):
    #     tb = noise_config.temporal_behavior
    #     days_map = {"low": 7, "medium": 30, "high": 365}
    #     max_days = np.random.randint(1, 10) * days_map.get(tb.intensity, 30)
    #
    #     if tb.corruption_mode == "uniform":
    #         shifts = np.random.randint(-max_days, max_days + 1, size=size)
    #     elif tb.corruption_mode == "forward":
    #         shifts = np.random.randint(1, max_days + 1, size=size)
    #     elif tb.corruption_mode == "backward":
    #         shifts = -np.random.randint(1, max_days + 1, size=size)
    #     else:
    #         raise ValueError(f"Unknown corruption_mode: {tb.corruption_mode}")
    #
    #     return pd.to_timedelta(shifts, unit="D").values

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

        return df, df[df['noise'] == True]

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
