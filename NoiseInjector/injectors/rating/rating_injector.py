# import numpy as np
# import pandas as pd
# from ..base import BaseNoiseInjector
# import random
# class RatingNoiseInjector(BaseNoiseInjector):
#     def __init__(self, config, logger):
#         """
#         config: dict parsed from YAML
#         """
#         self.config = config.noise_config
#         self.budget = self.config.budget
#         self.logger = logger
#
#     def apply_noise(self, df):
#         df = df[['user_id','item_id','rating','timestamp']]
#         if self.config.context == "realistic_noise":
#             df_noisy = self._realistic_noise(df)
#         elif self.config.context == "user_burst_noise":
#             df_noisy = self._user_burst_noise(df)
#         elif self.config.context == "item_burst_noise":
#             df_noisy = self._item_burst_noise(df)
#         elif self.config.context == "timestamp_corruption":
#             df_noisy = self._timestamp_corruption(df)
#         else:
#             self.logger.error(f"Unknown rating noise context: {self.config.noise_config.context}")
#             raise ValueError(f"Unknown rating noise context: {self.config.noise_config.context}")
#
#         return df_noisy
#
#     def _realistic_noise(self,df):
#         config = self.config.realistic_noise
#         target = 'user_id' if config.target == 'user' else 'item_id'
#         other = 'item_id' if config.target == 'user' else 'user_id'
#         strategy = config.selection_strategy
#         # if config.rating_behavior.sampling_strategy == 'gaussian':
#         #     self.logger.info("Using gaussian sampling strategy for ratings.")
#         #     ratings = np.random.normal(loc=df['rating'].mean(), scale=df['rating'].std(), size=self.budget)
#         #     ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#         # else:
#         #     self.logger.info("Using uniform sampling strategy for ratings.")
#         #     ratings = np.random.randint(config.min_rating, config.max_rating+1, size=self.budget)
#
#         # sort in descending order based on user_id or item_id -- in general, target
#         ordered_list = df[target].value_counts(ascending=strategy == 'least').index.tolist()
#         selected_nodes = []
#         if strategy == 'uniform':
#             self.logger.info("Using uniform selection strategy.")
#             # select randomly k users or items
#             random.shuffle(ordered_list)
#             #selected_nodes = np.random.choice(df[target].unique(), size=k, replace=False)
#         node_degrees = df.groupby(target)[other].nunique()
#         max_degree = node_degrees.max()
#         #elif strategy == 'popularity_based':
#         #     self.logger.info("Using popularity-based selection strategy.")
#         #     selected_nodes = ordered_list[:k]
#         # detect all the ratings associated to the selected users or items
#         #df_filtered = df[df[target].isin(selected_nodes)]
#         # find the rating behaviour: check the rating column in df and compute its mean and std and
#         if config.operation == 'remove':
#             self.logger.info("Applying realistic noise: remove ratings")
#             # from df_filtered, for each user or item, remove a certain number of ratings, from config.min_ratings_per_node to config.max_ratings_per_node
#             # find ratings to assign basing a gaussian distribution
#
#             ratings_to_remove_df = pd.DataFrame(columns=df.columns)  # lista vuota come DataFrame
#             total_removed = 0
#
#             while total_removed < self.budget:
#                 for node in ordered_list:
#                     node_ratings = df[df[target] == node]
#                     # possible_ratings = node_ratings[node_ratings['rating'].isin(ratings)]
#                     # n_ratings = len(possible_ratings)
#
#                     # if n_ratings == 0:
#                     #     continue
#
#                     min_remove = min(config.per_node_limits.min_ratings_per_node, len(node_ratings))
#                     max_remove = min(config.per_node_limits.max_ratings_per_node, len(node_ratings))
#                     if min_remove > max_remove:
#                         n_remove = np.random.randint(config.per_node_limits.min_ratings_per_node, config.per_node_limits.max_ratings_per_node + 1)
#                     elif min_remove == max_remove:
#                         n_remove = min_remove
#                     else:
#                         raise ValueError("min_remove cannot be greater than max_remove")
#
#                     if config.rating_behavior.sampling_strategy == 'gaussian':
#                         self.logger.info("Using gaussian sampling strategy for ratings.")
#                         ratings = np.random.normal(loc=node_ratings['rating'].mean(), scale=node_ratings['rating'].std(), size=n_remove)
#                         ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#                     else:
#                         self.logger.info("Using uniform sampling strategy for ratings.")
#                         ratings = np.random.randint(config.min_rating, config.max_rating+1, size=n_remove)
#
#                     # non superare il budget residuo
#                     n_remove = min(n_remove, self.budget - total_removed)
#                     if n_remove <= 0:
#                         break
#
#                     node_ratings = node_ratings[node_ratings['rating'].isin(ratings)]
#                     sampled = node_ratings.sample(n=n_remove, replace=False)
#                     # get the ratings values to exclude from possible ratings
#                     sampled_ratings = sampled['rating'].unique()
#                     # exclude the sampled ratings from possible ratings
#                     # ratings = ratings.tolist()
#                     # for r in sampled_ratings:
#                     #     if r in ratings:
#                     #         ratings.remove(r)
#                     # ratings = np.array(ratings)
#                     ratings_to_remove_df = pd.concat([ratings_to_remove_df, sampled])
#                     total_removed += n_remove
#
#                     if total_removed >= self.budget:
#                         break
#
#             df_noisy = df.drop(ratings_to_remove_df.index)
#             return df_noisy, ratings_to_remove_df
#
#         elif config.operation == 'add':
#             self.logger.info("Applying realistic noise: add ratings")
#             # Implement addition logic here
#             # For now, just return the original df
#
#             ratings_to_add_df = pd.DataFrame(columns=df.columns)  # lista vuota come DataFrame
#             total_added = 0
#             all_other = df[other].unique() # tutti gli item_id se target è user_id e viceversa
#
#             while total_added < self.budget:
#                 for node in ordered_list:
#                     # select
#
#                     node_ratings = df[df[target] == node]
#                     node_other = node_ratings[other].unique()
#                     all_other_available = np.setdiff1d(all_other, node_other)
#                     if len(all_other_available) == 0:
#                         continue
#                     n_add = np.random.randint(config.per_node_limits.min_ratings_per_node,
#                                               config.per_node_limits.max_ratings_per_node + 1)
#
#                     if config.preserve_degree_distribution:
#                         degree_factor = (max_degree - node_degrees.get(node, 0)) / max_degree
#                         n_add = int(np.ceil(n_add * degree_factor))
#
#                     n_add = min(n_add, self.budget - total_added, len(all_other_available))
#
#                     # non superare il budget residuo
#                     n_add = min(n_add, self.budget - total_added)
#                     if n_add <= 0:
#                         break
#
#                     sampled_other = np.random.choice(all_other_available, size=n_add, replace=False)
#                     if config.rating_behavior.sampling_strategy == 'gaussian':
#                         self.logger.info("Using gaussian sampling strategy for ratings.")
#                         ratings = np.random.normal(loc=node_ratings['rating'].mean(),
#                                                    scale=node_ratings['rating'].std(), size=n_add)
#                         sampled_ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#                     else:
#                         self.logger.info("Using uniform sampling strategy for ratings.")
#                         sampled_ratings = np.random.randint(config.min_rating, config.max_rating + 1, size=n_add)
#
#                     # exclude the sampled ratings from possible ratings
#                     # ratings = ratings.tolist()
#                     # for r in sampled_ratings:
#                     #     if r in ratings:
#                     #         ratings.remove(r)
#                     # ratings = np.array(ratings)
#                     new_ratings = pd.DataFrame({
#                         target: [node]*n_add,
#                         other: sampled_other,
#                         'rating': sampled_ratings,
#                         'timestamp': pd.Timestamp.now().timestamp()  # or any other logic for timestamp
#                     })
#                     ratings_to_add_df = pd.concat([ratings_to_add_df, new_ratings])
#                     total_added += n_add
#
#                     if total_added >= self.budget:
#                         break
#
#             df_noisy = pd.concat([ratings_to_add_df, df], ignore_index=True)
#             return df_noisy, ratings_to_add_df
#         else:
#             self.logger.error(f"Unknown operation: {config.operation}")
#             raise ValueError(f"Unknown operation: {config.operation}")
#
#
#     def _user_burst_noise(self,df):
#
#         config = self.config.realistic_noise
#         config.operation = 'add'
#         target = 'user_id'
#         other = 'item_id'
#         strategy = config.selection_strategy
#
#
#         # sort in descending order based on user_id or item_id -- in general, target
#         ordered_list = df[target].value_counts(ascending=strategy == 'least').index.tolist()
#         selected_nodes = []
#         if strategy == 'uniform':
#             self.logger.info("Using uniform selection strategy.")
#             # select randomly k users or items
#             random.shuffle(ordered_list)
#             # selected_nodes = np.random.choice(df[target].unique(), size=k, replace=False)
#         node_degrees = df.groupby(target)[other].nunique()
#         max_degree = node_degrees.max()
#
#         self.logger.info("Applying realistic noise: add ratings")
#             # Implement addition logic here
#             # For now, just return the original df
#
#         ratings_to_add_df = pd.DataFrame(columns=df.columns)  # lista vuota come DataFrame
#         total_added = 0
#         all_other = df[other].unique()  # tutti gli item_id se target è user_id e viceversa
#
#         while total_added < self.budget:
#             for node in ordered_list:
#                 node_ratings = df[df[target] == node]
#                 node_other = node_ratings[other].unique()
#                 all_other_available = np.setdiff1d(all_other, node_other)
#                 if len(all_other_available) == 0:
#                     continue
#                 n_add = np.random.randint(config.per_node_limits.min_ratings_per_node,
#                                           config.per_node_limits.max_ratings_per_node + 1)
#
#
#                 n_add = min(n_add, self.budget - total_added, len(all_other_available))
#
#                 # non superare il budget residuo
#                 n_add = min(n_add, self.budget - total_added)
#                 if n_add <= 0:
#                     break
#
#                 sampled_other = np.random.choice(all_other_available, size=n_add, replace=False)
#                 #sampled_ratings = np.random.choice(ratings, size=n_add, replace=True)
#                 sampled_other = np.random.choice(all_other_available, size=n_add, replace=False)
#                 if config.rating_behavior.sampling_strategy == 'gaussian':
#                     self.logger.info("Using gaussian sampling strategy for ratings.")
#                     ratings = np.random.normal(loc=node_ratings['rating'].mean(),
#                                                scale=node_ratings['rating'].std(), size=n_add)
#                     sampled_ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#                 else:
#                     self.logger.info("Using uniform sampling strategy for ratings.")
#                     sampled_ratings = np.random.randint(config.min_rating, config.max_rating + 1, size=n_add)
#
#                 # exclude the sampled ratings from possible ratings
#                 ratings = ratings.tolist()
#                 for r in sampled_ratings:
#                     if r in ratings:
#                         ratings.remove(r)
#                 ratings = np.array(ratings)
#                 new_ratings = pd.DataFrame({
#                     target: [node] * n_add,
#                     other: sampled_other,
#                     'rating': sampled_ratings,
#                     'timestamp': pd.Timestamp.now().timestamp()  # or any other logic for timestamp
#                 })
#                 ratings_to_add_df = pd.concat([ratings_to_add_df, new_ratings])
#                 total_added += n_add
#
#                 if total_added >= self.budget:
#                     break
#
#             df_noisy = pd.concat([ratings_to_add_df, df], ignore_index=True)
#             return df_noisy, ratings_to_add_df
#         else:
#             self.logger.error(f"Unknown operation: {config.operation}")
#             raise ValueError(f"Unknown operation: {config.operation}")
#
#     def _item_burst_noise(self,df):
#         config = self.config.realistic_noise
#         config.operation = 'add'
#         target = 'item_id'
#         other = 'user_id'
#         strategy = config.selection_strategy
#
#
#         # sort in descending order based on user_id or item_id -- in general, target
#         ordered_list = df[target].value_counts(ascending=strategy == 'least').index.tolist()
#         selected_nodes = []
#         if strategy == 'uniform':
#             self.logger.info("Using uniform selection strategy.")
#             # select randomly k users or items
#             random.shuffle(ordered_list)
#             # selected_nodes = np.random.choice(df[target].unique(), size=k, replace=False)
#         node_degrees = df.groupby(target)[other].nunique()
#         max_degree = node_degrees.max()
#
#         self.logger.info("Applying realistic noise: add ratings")
#             # Implement addition logic here
#             # For now, just return the original df
#
#         ratings_to_add_df = pd.DataFrame(columns=df.columns)  # lista vuota come DataFrame
#         total_added = 0
#         all_other = df[other].unique()  # tutti gli item_id se target è user_id e viceversa
#
#         while total_added < self.budget:
#             for node in ordered_list:
#                 node_ratings = df[df[target] == node]
#                 node_other = node_ratings[other].unique()
#                 all_other_available = np.setdiff1d(all_other, node_other)
#                 if len(all_other_available) == 0:
#                     continue
#                 n_add = np.random.randint(config.per_node_limits.min_ratings_per_node,
#                                           config.per_node_limits.max_ratings_per_node + 1)
#
#
#                 n_add = min(n_add, self.budget - total_added, len(all_other_available))
#
#                 # non superare il budget residuo
#                 n_add = min(n_add, self.budget - total_added)
#                 if n_add <= 0:
#                     break
#
#                 sampled_other = np.random.choice(all_other_available, size=n_add, replace=False)
#                 #sampled_ratings = np.random.choice(ratings, size=n_add, replace=True)
#                 # exclude the sampled ratings from possible ratings
#                 # ratings = ratings.tolist()
#                 # for r in sampled_ratings:
#                 #     if r in ratings:
#                 #         ratings.remove(r)
#                 # ratings = np.array(ratings)
#                 sampled_other = np.random.choice(all_other_available, size=n_add, replace=False)
#                 if config.rating_behavior.sampling_strategy == 'gaussian':
#                     self.logger.info("Using gaussian sampling strategy for ratings.")
#                     ratings = np.random.normal(loc=node_ratings['rating'].mean(),
#                                                scale=node_ratings['rating'].std(), size=n_add)
#                     sampled_ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#                 else:
#                     self.logger.info("Using uniform sampling strategy for ratings.")
#                     sampled_ratings = np.random.randint(config.min_rating, config.max_rating + 1, size=n_add)
#
#                 new_ratings = pd.DataFrame({
#                     target: [node] * n_add,
#                     other: sampled_other,
#                     'rating': sampled_ratings,
#                     'timestamp': pd.Timestamp.now().timestamp()  # or any other logic for timestamp
#                 })
#                 ratings_to_add_df = pd.concat([ratings_to_add_df, new_ratings])
#                 total_added += n_add
#
#                 if total_added >= self.budget:
#                     break
#
#             df_noisy = pd.concat([ratings_to_add_df, df], ignore_index=True)
#             return df_noisy, ratings_to_add_df
#         else:
#             self.logger.error(f"Unknown operation: {config.operation}")
#             raise ValueError(f"Unknown operation: {config.operation}")
#
#
#     def _corrupt(self, df):
#         tb = self.config.realistic_noise.temporal_behavior
#
#         INTENSITY_TO_DAYS = {
#             "low": 7,
#             "medium": 30,
#             "high": 365
#         }
#
#         max_days = np.random.randint(1, 10, size=1)[0] * INTENSITY_TO_DAYS.get(tb.intensity, 30)
#
#         if tb.corruption_mode == "uniform_shift":
#             shifts = np.random.randint(-max_days, max_days + 1, size=len(df))
#         elif tb.corruption_mode == "forward":
#             shifts = np.random.randint(1, max_days + 1, size=len(df))
#         elif tb.corruption_mode == "backward":
#             shifts = -np.random.randint(1, max_days + 1, size=len(df))
#         else:
#             raise ValueError(f"Unknown corruption_mode: {tb.corruption_mode}")
#
#         return pd.to_datetime(df["timestamp"]) + pd.to_timedelta(shifts, unit="D")
#
#     def _timestamp_corruption(self,df):
#
#
#
#
#
#         config = self.config.realistic_noise
#         config.operation = 'corrupt'
#         target = 'user_id' if config.target == 'user' else 'item_id'
#         other = 'item_id' if config.target == 'user' else 'user_id'
#         strategy = config.selection_strategy
#         # if config.rating_behavior.sampling_strategy == 'gaussian':
#         #     self.logger.info("Using gaussian sampling strategy for ratings.")
#         #     ratings = np.random.normal(loc=df['rating'].mean(), scale=df['rating'].std(), size=self.budget)
#         #     ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#         # else:
#         #     self.logger.info("Using uniform sampling strategy for ratings.")
#         #     ratings = np.random.randint(config.min_rating, config.max_rating+1, size=self.budget)
#
#         # sort in descending order based on user_id or item_id -- in general, target
#         ordered_list = df[target].value_counts(ascending=strategy == 'least').index.tolist()
#         selected_nodes = []
#         if strategy == 'uniform':
#             self.logger.info("Using uniform selection strategy.")
#             # select randomly k users or items
#             random.shuffle(ordered_list)
#             # selected_nodes = np.random.choice(df[target].unique(), size=k, replace=False)
#
#         self.logger.info("Applying realistic noise: remove ratings")
#         # from df_filtered, for each user or item, remove a certain number of ratings, from config.min_ratings_per_node to config.max_ratings_per_node
#         # find ratings to assign basing a gaussian distribution
#
#         ratings_to_remove_df = pd.DataFrame(columns=df.columns)  # lista vuota come DataFrame
#         total_corrupted = 0
#
#
#         while total_corrupted < self.budget:
#             for node in ordered_list:
#                 node_ratings = df[df[target] == node]
#                 # possible_ratings = node_ratings[node_ratings['rating'].isin(ratings)]
#                 # n_ratings = len(possible_ratings)
#
#                 # if n_ratings == 0:
#                 #     continue
#
#                 min_corrupt = min(config.per_node_limits.min_ratings_per_node, len(node_ratings))
#                 max_corrupt = min(config.per_node_limits.max_ratings_per_node, len(node_ratings))
#                 if min_corrupt > max_corrupt:
#                     n_corrupt = np.random.randint(config.per_node_limits.min_ratings_per_node,
#                                                  config.per_node_limits.max_ratings_per_node + 1)
#                 elif min_corrupt == max_corrupt:
#                     n_corrupt = max_corrupt
#                 else:
#                     raise ValueError("min_remove cannot be greater than max_remove")
#
#                 if config.rating_behavior.sampling_strategy == 'gaussian':
#                     self.logger.info("Using gaussian sampling strategy for ratings.")
#                     ratings = np.random.normal(loc=node_ratings['rating'].mean(),
#                                                scale=node_ratings['rating'].std(), size=n_corrupt)
#                     ratings = np.clip(np.rint(ratings), config.min_rating, config.max_rating).astype(int)
#                 else:
#                     self.logger.info("Using uniform sampling strategy for ratings.")
#                     ratings = np.random.randint(config.min_rating, config.max_rating + 1, size=n_corrupt)
#
#                 # non superare il budget residuo
#                 n_corrupt = min(n_corrupt, self.budget - total_corrupted)
#                 if n_corrupt <= 0:
#                     break
#
#                 node_ratings = node_ratings[node_ratings['rating'].isin(ratings)]
#                 sampled = node_ratings.sample(n=n_corrupt, replace=False)
#                 sampled_timestamps = self._corrupt(sampled)["timestamp"]
#                 df.loc[sampled.index, "timestamp"] = sampled_timestamps.values
#
#
#                 ratings_to_remove_df = pd.concat([ratings_to_remove_df, sampled])
#                 total_corrupted += n_corrupt
#
#                 if total_corrupted >= self.budget:
#                     break
#
#             df_noisy = df.drop(ratings_to_remove_df.index)
#             return df_noisy, ratings_to_remove_df
#
#
#         else:
#             self.logger.error(f"Unknown operation: {config.operation}")
#             raise ValueError(f"Unknown operation: {config.operation}")
#
import numpy as np
import pandas as pd
import random
from ..base import BaseNoiseInjector


class RatingNoiseInjector(BaseNoiseInjector):

    def __init__(self, config, logger):
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
        if ctx == "user_burst_noise":
            return self._burst_noise(df, target='user_id')
        if ctx == "item_burst_noise":
            return self._burst_noise(df, target='item_id')
        if ctx == "timestamp_corruption":
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
            return np.clip(np.rint(r), config.min_rating, config.max_rating).astype(int)
        return np.random.randint(config.min_rating, config.max_rating + 1, size=n)

    def _per_node_budget(self, node_size, total_left, limits):
        n = np.random.randint(
            limits.min_ratings_per_node,
            limits.max_ratings_per_node + 1
        )
        return min(n, node_size, total_left)

    # ==========================================================
    # REALISTIC NOISE
    # ==========================================================
    def _realistic_noise(self, df):
        config = self.config.realistic_noise
        target = 'user_id' if config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'

        nodes = self._ordered_nodes(df, target, config.selection_strategy)

        if config.operation == 'remove':
            return self._remove_ratings(df, nodes, target,other, config)

        if config.operation == 'add':
            return self._add_ratings(df, nodes, target, other, config)

        raise ValueError(f"Unknown operation: {config.operation}")

    # ==========================================================
    # REMOVE
    # ==========================================================
    def _remove_ratings(self, df, nodes, target, other, config):
        removed_idx = []
        remaining = self.budget
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            n = self._per_node_budget(len(node_df), remaining, config.per_node_limits)
            if n <= 0 or n == len(node_df):
                continue

            if getattr(config, "preserve_degree_distribution", False):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            ratings = self._sample_ratings(node_df['rating'], n, config)

            start_ts = config.temporal_interval.start_timestamp
            end_ts = config.temporal_interval.end_timestamp

            candidates = node_df[
                node_df['rating'].isin(ratings)
                ]

            if start_ts != 0 and end_ts != 0:

                candidates = node_df[
                    node_df['rating'].isin(ratings) &  # rating desiderati
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
    def _add_ratings(self, df, nodes, target, other, config):
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

            n = self._per_node_budget(len(available), remaining, config.per_node_limits)

            if getattr(config, "preserve_degree_distribution", False):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            sampled_other = np.random.choice(available, size=n, replace=False)
            sampled_ratings = self._sample_ratings(node_df['rating'], n, config)

            start_ts = config.temporal_interval.start_timestamp
            end_ts = config.temporal_interval.end_timestamp


            if start_ts == 0 and end_ts == 0:
                timestamps = pd.Timestamp.now().timestamp()
            else:
                timestamps = np.random.randint(start_ts, end_ts + 1, size=n)

            rows.append(pd.DataFrame({
                target: node,
                other: sampled_other,
                'rating': sampled_ratings,
                'timestamp': timestamps
            }))

            remaining -= n

        added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
        return pd.concat([df, added], ignore_index=True), added

    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df, target):
        config = self.config.realistic_noise
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        return self._add_ratings(df, nodes, target, other, config)

    # ==========================================================
    # TIMESTAMP CORRUPTION
    # ==========================================================
    def _timestamp_corruption(self, df):
        config = self.config.realistic_noise
        nodes = self._ordered_nodes(
            df,
            'user_id' if config.target == 'user' else 'item_id',
            config.selection_strategy
        )

        remaining = self.budget

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df['user_id'] == node]
            n = self._per_node_budget(len(node_df), remaining, config.per_node_limits)
            sampled = node_df.sample(n=n, replace=False)

            df.loc[sampled.index, 'timestamp'] = self._corrupt(sampled)
            remaining -= n

        return df, None

    def _corrupt(self, df):
        tb = self.config.realistic_noise.temporal_behavior
        days_map = {"low": 7, "medium": 30, "high": 365}
        max_days = np.random.randint(1, 10) * days_map.get(tb.intensity, 30)

        if tb.corruption_mode == "uniform_shift":
            shifts = np.random.randint(-max_days, max_days + 1, size=len(df))
        elif tb.corruption_mode == "forward":
            shifts = np.random.randint(1, max_days + 1, size=len(df))
        elif tb.corruption_mode == "backward":
            shifts = -np.random.randint(1, max_days + 1, size=len(df))
        else:
            raise ValueError(f"Unknown corruption_mode: {tb.corruption_mode}")

        return pd.to_datetime(df["timestamp"]) + pd.to_timedelta(shifts, unit="D")
