
import numpy as np
import pandas as pd
import random
from NoiseInjector.injectors.base import BaseNoiseInjector
import random
import nltk
from nltk.corpus import wordnet
import json
# scarica WordNet se non già presente
nltk.download('wordnet')
nltk.download('omw-1.4')

from transformers import pipeline
sentiment_inverter = pipeline("text2text-generation", model="t5-base",device = 0)  # esempio

class CombinedNoiseInjector(BaseNoiseInjector):

    def __init__(self, config, logger):
        self.config = config.noise_config
        self.budget = self.config.budget
        self.logger = logger

    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def apply_noise(self, df):
        df = df[['user_id', 'item_id', 'rating','review_text','title', 'timestamp']].copy()

        ctx = self.config.context
        if ctx == "rating_review_burst":
            return self._corrupt(df, target=ctx.target)
        elif ctx == 'semantic_drift':
            return self._drift(df, target=ctx.target)






        raise ValueError(f"Unknown rating noise context: {ctx}")


    def _corrupt(self, df, target):
        config = self.config.rating_review_burst
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        df = self._sample_reviews(df, config)
        if config.modify == 'rating':
            return self._corrupt_ratings(df, nodes, target, other, config)
        else:
            return self._corrupt_reviews(df, nodes, target, other, config)

    def _drift(self, df, target):
        config = self.config.rating_review_burst
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        df = self._sample_reviews(df, config)
        return self._semantic_drift(df, nodes, target, other, config)


    def _corrupt_ratings(self, df, nodes, target, other, config):
        """
        Aggiunge review a un set di nodi per simulare review burst.

        - df: dataframe originale
        - nodes: lista di target nodes (item o user)
        - target: colonna target ('item' o 'user')
        - other: colonna complementare ('user' o 'item')
        - config: oggetto config con parametri del burst
        """
        remaining = self.budget
        start_ts = getattr(config, 'temporal_interval', {}).get('start_timestamp', 0)
        end_ts = getattr(config, 'temporal_interval', {}).get('end_timestamp', 2 ** 31 - 1)

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]

            # Filtra recensioni nel range temporale
            candidates = node_df.copy()
            ratings = self._sample_ratings(node_df['rating'], n, config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]
            if start_ts != 0 and end_ts != 0:
                candidates = candidates[
                    (node_df['timestamp'] >= start_ts) &
                    (node_df['timestamp'] <= end_ts)
                    ]
            if candidates.empty:
                continue

            # Numero di recensioni da corrompere
            n = self._per_node_budget(len(candidates), remaining, config.per_node_limits)
            if n <= 0:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            remaining -= len(sampled)

            for idx in sampled.index:
                rating = sampled.at[idx, 'rating']
                if rating in [config.rating_behavior.max_rating-1,config.rating_behavior.max_rating]:
                    rating = config.rating_behavior.min_rating
                elif rating in [config.rating_behavior.min_rating,config.rating_behavior.min_rating+1]:
                    rating = config.rating_behavior.max_rating

                df.at[idx, 'rating'] = rating

        return df

    def _corrupt_reviews(self, df, nodes, target, other, config, batch_size=32):
        """
        Corrupt reviews by flipping ratings and adjusting sentiment.
        Optimized with GPU and batch processing.
        """
        remaining = self.budget
        start_ts = getattr(config, 'temporal_interval', {}).get('start_timestamp', 0)
        end_ts = getattr(config, 'temporal_interval', {}).get('end_timestamp', 2 ** 31 - 1)

        # GPU pipeline
        global sentiment_inverter
        if 'sentiment_inverter' not in globals():
            from transformers import pipeline
            sentiment_inverter = pipeline("text2text-generation", model="t5-base", device=0)

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            ratings = self._sample_ratings(node_df['rating'], n, config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]
            if start_ts != 0 and end_ts != 0:
                candidates = candidates[
                    (node_df['timestamp'] >= start_ts) &
                    (node_df['timestamp'] <= end_ts)
                    ]
            if candidates.empty:
                continue

            n = self._per_node_budget(len(candidates), remaining, config.per_node_limits)
            if n <= 0:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            remaining -= len(sampled)

            # Prepariamo batch di testi da trasformare
            review_texts = []
            titles = []
            for idx in sampled.index:
                rating = sampled.at[idx, 'rating']
                review_text = sampled.at[idx, 'review_text']
                title = sampled.at[idx, 'title']

                if rating in [config.rating_behavior.max_rating - 1, config.rating_behavior.max_rating]:
                    # vogliamo recensione negativa
                    review_texts.append(f"Turn into negative review: {review_text}")
                    titles.append(f"Turn into negative review: {title}")
                    # Aggiorniamo il rating
                    df.at[idx, 'rating'] = config.rating_behavior.min_rating
                elif rating in [config.rating_behavior.min_rating, config.rating_behavior.min_rating + 1]:
                    # vogliamo recensione positiva
                    review_texts.append(f"Turn into positive review: {review_text}")
                    titles.append(f"Turn into positive review: {title}")
                    # Aggiorniamo il rating
                    df.at[idx, 'rating'] = config.rating_behavior.max_rating
                else:
                    # lasciamo invariato, non aggiungiamo al batch
                    review_texts.append(None)
                    titles.append(None)

            # Funzione di batch processing
            def batched_transform(text_list):
                outputs = []
                batch_indices = [i for i, t in enumerate(text_list) if t is not None]
                filtered_texts = [t for t in text_list if t is not None]

                for i in range(0, len(filtered_texts), batch_size):
                    batch = filtered_texts[i:i + batch_size]
                    res = sentiment_inverter(batch, max_length=100)
                    outputs.extend([r['generated_text'] for r in res])

                # reinseriamo None dove non abbiamo trasformazioni
                full_output = []
                j = 0
                for t in text_list:
                    if t is None:
                        full_output.append(None)
                    else:
                        full_output.append(outputs[j])
                        j += 1
                return full_output

            # Trasformiamo in batch
            new_review_texts = batched_transform(review_texts)
            new_titles = batched_transform(titles)

            # Scriviamo in blocco nel DataFrame
            for idx, new_r, new_t in zip(sampled.index, new_review_texts, new_titles):
                if new_r is not None:
                    df.at[idx, 'review_text'] = new_r
                if new_t is not None:
                    df.at[idx, 'title'] = new_t

        return df

    def _semantic_drift(self, df, nodes, target, other, config, batch_size=32):
        """
        Apply semantic drift to reviews for a set of nodes.
        Optimized with batching for GPU processing.
        """
        remaining = self.budget
        start_ts = getattr(config, 'temporal_interval', {}).get('start_timestamp', 0)
        end_ts = getattr(config, 'temporal_interval', {}).get('end_timestamp', 2 ** 31 - 1)

        # Create a GPU pipeline if not already
        global sentiment_inverter
        if 'sentiment_inverter' not in globals():
            from transformers import pipeline
            sentiment_inverter = pipeline("text2text-generation", model="t5-base", device=0)

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]

            ratings = self._sample_ratings(node_df['rating'], n, config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]
            if start_ts != 0 and end_ts != 0:
                candidates = candidates[
                    (node_df['timestamp'] >= start_ts) &
                    (node_df['timestamp'] <= end_ts)
                    ]
            if candidates.empty:
                continue

            n = self._per_node_budget(len(candidates), remaining, config.per_node_limits)
            if n <= 0:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            remaining -= len(sampled)

            # Batch the review_text and title for semantic drift
            review_texts = sampled['review_text'].tolist()
            titles = sampled['title'].tolist()

            # Process in batches
            def batched_transform(text_list):
                outputs = []
                for i in range(0, len(text_list), batch_size):
                    batch = text_list[i:i + batch_size]
                    batch_prompts = [f"Change the main context while keeping style and grammar: {t}" for t in batch]
                    res = sentiment_inverter(batch_prompts, max_length=100)
                    outputs.extend([r['generated_text'] for r in res])
                return outputs

            new_review_texts = batched_transform(review_texts)
            new_titles = batched_transform(titles)

            # Write back in bulk using .loc
            df.loc[sampled.index, 'review_text'] = new_review_texts
            df.loc[sampled.index, 'title'] = new_titles

        return df

    def _sample_ratings(self, base_ratings, n, config):
        if config.rating_behavior.sampling_strategy == 'gaussian':
            mu, sigma = base_ratings.mean(), base_ratings.std() or 1.0
            r = np.random.normal(mu, sigma, size=n)
            return np.clip(np.rint(r), config.min_rating, config.max_rating).astype(int)
        return np.random.randint(config.min_rating, config.max_rating + 1, size=n)

    def _sample_reviews(self, df, config):
        if config.min_length_of_review > 0:
            token_counts = df['review_text'].fillna("").str.split().str.len()
            # Filtra
            df_filtered = df[token_counts >= config.min_length_of_review].copy()
            return df_filtered
        return df

    def _ordered_nodes(self, df, target, strategy):
        nodes = df[target].value_counts(ascending=strategy == 'least').index.tolist()
        if strategy == 'uniform':
            random.shuffle(nodes)
        return nodes