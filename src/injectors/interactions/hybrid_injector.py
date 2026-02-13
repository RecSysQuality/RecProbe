
import numpy as np
import pandas as pd
import random


from src.injectors.base import BaseNoiseInjector
import random
import nltk
from nltk.corpus import wordnet
import json
import torch
from src.utils import *
# scarica WordNet se non già presente
# nltk.download('wordnet')
# nltk.download('omw-1.4')

from transformers import pipeline
#sentiment_inverter = pipeline("text2text-generation", model="t5-base",device = 0)  # esempio

class HybridNoiseInjector(BaseNoiseInjector):
    def __init__(self, config, logger):
        self.config = config.noise_config
        self.budget = self.config.budget
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # ==========================================================
        # PUBLIC API
        # ==========================================================
    def apply_noise(self, df, df_val,df_test):
        df = df[['user_id', 'item_id', 'rating', 'review_text', 'title', 'timestamp']].copy()
        ctx = self.config.context
        if ctx == "rating_review_burst":
            return self._corrupt(df)
        if ctx == "semantic_drift":
            return self._drift(df)


        raise ValueError(f"Unknown rating noise context: {ctx}")
    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def _drift(self, df):
        config = self.config
        noise_config = self.config.semantic_drift
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        df = sample_reviews(df, noise_config)
        return self._semantic_drift(df, nodes, target, other, config,noise_config)


    def _corrupt(self, df):
        config = self.config
        noise_config = self.config.rating_review_burst
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        df = sample_reviews(df, noise_config)
        if noise_config.modify == 'rating':
            return self._corrupt_ratings(df, nodes, target, other, config,noise_config)
        else:
            return self._corrupt_reviews(df, nodes, target, other, config,noise_config)


    def _corrupt_ratings(self, df, nodes, target, other, config,noise_config):
        mod_idx = []
        remaining = self.budget

        df['noise'] = False
        start_ts = parse_timestamp(noise_config.temporal_interval.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_interval.end_timestamp)


        review_to_convert, title_to_convert = [], []
        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            n = per_node_budget(len(node_df), remaining, noise_config.min_reviews_per_node,
                                noise_config.max_reviews_per_node)
            if n <= 0:
                continue

            ratings = sample_ratings(node_df['rating'], n, noise_config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]

            # if start_ts != 0 and end_ts != 0:
            #     candidates = node_df[
            #         node_df['rating'].isin(ratings) &  # rating desiderati
            #         (node_df['timestamp'] >= start_ts) &  # timestamp >= start
            #         (node_df['timestamp'] <= end_ts)  # timestamp <= end
            #         ]

            if candidates.empty:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            mod_idx.extend(sampled.index)

            remaining -= len(sampled)
            for idx in sampled.index:
                rating = sampled.at[idx, 'rating']
                mid = [i for i in range(noise_config.rating_behavior.min_rating,noise_config.rating_behavior.max_rating+1)]
                middle_index = len(mid) // 2

                # Ottenere il valore centrale
                middle_value = mid[middle_index]
                if rating > middle_value:
                    rating = noise_config.rating_behavior.min_rating
                elif rating < middle_value:
                    rating = noise_config.rating_behavior.max_rating

                df.at[idx, 'rating'] = rating
                df.at[idx, 'noise'] = True
        removed = df.loc[mod_idx]

        return df,removed

    def _corrupt_reviews(self, df, nodes, target, other, config, noise_config):
        """
        Corrupt reviews by flipping ratings and adjusting sentiment.
        Optimized with GPU and batch processing.
        """
        mod_idx = []
        remaining = self.budget

        df['noise'] = False
        start_ts = noise_config.temporal_interval.start_timestamp
        end_ts = noise_config.temporal_interval.end_timestamp
        start_ts = parse_timestamp(start_ts)
        end_ts = parse_timestamp(end_ts)

        review_to_convert, title_to_convert = [], []
        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            n = per_node_budget(len(node_df), remaining, noise_config.min_reviews_per_node,
                                noise_config.max_reviews_per_node)
            if n <= 0:
                continue

            ratings = sample_ratings(node_df['rating'], n, noise_config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]

            # if start_ts != 0 and end_ts != 0:
            #     candidates = node_df[
            #         node_df['rating'].isin(ratings) &  # rating desiderati
            #         (node_df['timestamp'] >= start_ts) &  # timestamp >= start
            #         (node_df['timestamp'] <= end_ts)  # timestamp <= end
            #         ]

            if candidates.empty:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            mod_idx.extend(sampled.index)
            remaining -= len(sampled)


            # Trasformiamo in batch
            new_review_texts,new_titles = self._invert_sentiment(config,noise_config)
           # new_titles = self._invert_sentiment( config,noise_config)

            # Scriviamo in blocco nel DataFrame
            for idx, new_r, new_t in zip(sampled.index, new_review_texts, new_titles):
                if new_r is not None:
                    df.at[idx, 'review_text'] = new_r
                if new_t is not None:
                    df.at[idx, 'title'] = new_t

        removed = df.loc[mod_idx]
        df['noise'] = False
        df.loc[mod_idx,'noise'] = True
        return df,removed

    def _semantic_drift(self, df, nodes, target, other, config, noise_config):
        mod_idx = []
        remaining = self.budget

        df['noise'] = False
        start_ts = noise_config.temporal_interval.start_timestamp
        end_ts = noise_config.temporal_interval.end_timestamp
        start_ts = parse_timestamp(start_ts)
        end_ts = parse_timestamp(end_ts)

        review_to_convert,title_to_convert = [],[]
        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]
            n = per_node_budget(len(node_df), remaining, noise_config.min_reviews_per_node,
                                      noise_config.max_reviews_per_node)
            if n <= 0:
                continue

            ratings = sample_ratings(node_df['rating'], n, noise_config)
            candidates = node_df[
                node_df['rating'].isin(ratings)
            ]

            if start_ts != 0 and end_ts != 0:
                candidates = node_df[
                    node_df['rating'].isin(ratings) &  # rating desiderati
                    (node_df['timestamp'] >= start_ts) &  # timestamp >= start
                    (node_df['timestamp'] <= end_ts)  # timestamp <= end
                    ]

            if candidates.empty:
                continue

            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            mod_idx.extend(sampled.index)

            remaining -= len(sampled)
            df.loc[sampled.index, 'noise'] = True
            review_texts = sampled['review_text'].tolist()
            title_text = sampled['title'].tolist()
            review_to_convert.extend(review_texts)
            title_to_convert.extend(title_text)

        generator = pipeline(
            "text-generation",
            model=noise_config.model,
            device=-1,  # CPU
            torch_dtype="auto"
        )

        text_list = review_to_convert + title_to_convert
        new_reviews, new_titles = self._change_context(generator,text_list,len(review_to_convert))
        df.loc[mod_idx.index, 'review_text'] = new_reviews
        df.loc[mod_idx.index, 'title'] = new_titles
        df['noise'] = False
        df.loc[mod_idx.index, 'noise'] = True
        mod = df.loc[mod_idx]
        return df, mod


    @staticmethod
    def _change_context(generator,text_list,mid):


        prompts = [f"Rewrite the sentence by:\n- keeping the same product\n- keeping the same sentiment\n- changing the context completely\n- avoiding references to the original context\n-- return only the revised text without any explanation\nSentence: {s} \nReturned sentence: "
         for s in text_list]

        outp = generator(
            prompts,
            batch_size=128,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8
        )
        outputs = [out["generated_text"].split('Returned sentence: \n')[-1].split('\n')[0].replace('"', '') for out in outp]

        return outputs[0:mid],outputs[mid:]

    @staticmethod
    def _invert_sentiment(df,noise_config):

        # Prepariamo batch di testi da trasformare
        review_texts = []
        titles = []
        for idx in df.index:
            review_text = df.at[idx, 'review_text']
            title = df.at[idx, 'title']
            rating = df.at[idx, 'rating']
            mid = [i for i in
                   range(noise_config.rating_behavior.min_rating, noise_config.rating_behavior.max_rating + 1)]
            middle_index = len(mid) // 2

            # Ottenere il valore centrale
            middle_value = mid[middle_index]
            if rating >= middle_value:
                review_texts.append(
                    f"Change the sentiment of the review, turning it into a negative review: {review_text} \nReturned sentence: ")
                if title != '':
                    titles.append(
                        f"Change the sentiment of the review, turning it into a negative review: {title} \nReturned sentence: ")
                # df.at[idx, 'rating'] = config.rating_behavior.min_rating
            elif rating < middle_value:
                review_texts.append(
                    f"Change the sentiment of the review, turning it into a positive review: {review_text} \nReturned sentence: ")
                if title != '':
                    titles.append(
                        f"Change the sentiment of the review, turning it into a positive review: {title} \nReturned sentence: ")
                # df.at[idx, 'rating'] = config.rating_behavior.max_rating


        # Funzione di batch processing
        generator = pipeline("text2text-generation", model=noise_config.model, device=0)

        outputs = []

        full_title,full_output = [],[]

        res = generator(
            review_texts,
            batch_size=128,
            max_new_tokens=120,
            do_sample=True,
            temperature=0.8
        )

        #outputs.extend([r['generated_text'] for r in res])
        outputs.extend([out["generated_text"].split('Returned sentence: \n')[-1].split('\n')[0].replace('"', '') for out in res])

        # reinseriamo None dove non abbiamo trasformazioni
        full_output = []
        j = 0
        for t in review_texts:
            if t is None:
                full_output.append(None)
            else:
                full_output.append(outputs[j])
                j += 1

        if len(titles) > 0:
            res = generator(
                titles,
                batch_size=128,
                max_new_tokens=120,
                do_sample=True,
                temperature=0.8
            )

            #outputs.extend([r['generated_text'] for r in res])
            outputs.extend([out["generated_text"].split('Returned sentence: \n')[-1].split('\n')[0].replace('"', '') for out in res])

            # reinseriamo None dove non abbiamo trasformazioni
            full_output = []
            j = 0
            for t in review_texts:
                if t is None:
                    full_output.append(None)
                else:
                    full_output.append(outputs[j])
                    j += 1
        else:
            full_title = [None for _ in range(len(full_output))]

        return full_output, full_title


