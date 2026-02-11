
import numpy as np
import pandas as pd
import random
from transformers import T5ForConditionalGeneration,T5Tokenizer
from NoiseInjector.injectors.base import BaseNoiseInjector
import random
import nltk
import torch
import re
from nltk.corpus import words as nltk_words


from nltk.corpus import wordnet
import json
from NoiseInjector.utils import *
from transformers import pipeline


class ReviewNoiseInjector(BaseNoiseInjector):

    def __init__(self, config, logger):
        self.config = config.noise_config
        self.budget = self.config.budget
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # ==========================================================
    # PUBLIC API
    # ==========================================================
    def apply_noise(self, df):
        df = df[['user_id', 'item_id', 'rating','review_text','title', 'timestamp']].copy()
        ctx = self.config.context
        if ctx == "remove_reviews":
            return self._remove_noise(df)
        if ctx == "review_burst_noise":
            return self._burst_noise(df)
        if ctx == "sentence_noise":
            return self._sentence_noise(df)

        raise ValueError(f"Unknown rating noise context: {ctx}")

    # ==========================================================
    # REMOVE NOISE
    # ==========================================================
    def _remove_noise(self, df):
        config = self.config
        noise_config = self.config.remove_reviews
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        df = sample_reviews(df, noise_config)
        return self._remove_reviews(df, nodes, target, other, config,noise_config)


    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df):
        config = self.config
        noise_config = self.config.review_burst_noise
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        df = sample_reviews(df, noise_config)
        return self._add_reviews(df, nodes, target, other, config, noise_config)

    # ==========================================================
    # SENTENCE NOISE
    # ==========================================================
    def _sentence_noise(self, df):
        config = self.config
        noise_config = self.config.sentence_noise
        target = 'user_id' if noise_config.target == 'user' else 'item_id'
        #self.global_vocab = json.load(open(config.vocab_path, 'r'))
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = ordered_nodes(df, target, noise_config.selection_strategy)
        df = sample_reviews(df, noise_config)
        return self._add_sentence_noise(df, nodes, target, other, config,noise_config)



    def _remove_reviews(self, df, nodes, target,other, config,noise_config):
        remaining = self.budget

        start_ts = parse_timestamp(noise_config.temporal_interval.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_interval.end_timestamp)

        df['noise'] = False

        # Pre-calcolo dei gradi dei nodi
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()

        # Indicizza per target e rating per filtraggi più rapidi
        df_grouped = df.groupby(target)

        removed_idx = []
        for node in nodes:
            if remaining <= 0:
                break

            if node not in df_grouped.groups:
                continue

            node_idx = df_grouped.groups[node]
            node_df = df.loc[node_idx]

            n = per_node_budget(len(node_df), remaining,
                                      noise_config.min_reviews_per_node,
                                      noise_config.max_reviews_per_node)
            if n <= 0:
                continue

            ratings = sample_ratings(node_df['rating'], n, noise_config)
            mask = node_df['rating'].isin(ratings)
            if start_ts != 0 and end_ts != 0:
                mask &= (node_df['timestamp'] >= start_ts) & (node_df['timestamp'] <= end_ts)

            candidates_idx = node_df.index[mask]

            if len(candidates_idx) == 0:
                continue

            sampled_idx = np.random.choice(candidates_idx, size=min(len(candidates_idx), n), replace=False)

            # Campiona casualmente le recensioni da svuotare
            #sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            #removed_idx.extend(sampled.index)
            removed_idx.extend(sampled_idx)
            remaining -= len(sampled_idx)
            df.loc[sampled_idx, 'noise'] = True

        # Svuota il testo delle recensioni selezionate
        df.loc[removed_idx, ['review_text', 'title']] = ""
        removed = df.loc[removed_idx]
        return df, removed

    # ==========================================================
        # ADD (shared by realistic + burst)
    # ==========================================================


    def _add_reviews(self, df, nodes, target, other, config, noise_config):

        rows = []
        remaining = self.budget
        all_other = df[other].unique()
        model = T5ForConditionalGeneration.from_pretrained(
            noise_config.near_duplicates_configuration.model,
            use_safetensors=True
        ).to(self.device)
        # model = T5ForConditionalGeneration.from_pretrained(noise_config.near_duplicates_configuration.model).to(
        #     self.device)
        tokenizer = T5Tokenizer.from_pretrained(noise_config.near_duplicates_configuration.model)

        seed_review, seed_title, seed_rating = '','',4

        seed_review = noise_config.near_duplicates_configuration.review
        seed_title = noise_config.near_duplicates_configuration.title
        seed_rating = noise_config.near_duplicates_configuration.rating

        if noise_config.near_duplicates_configuration.review is not None:
            new_reviews = self._paraphrase(model, tokenizer, text_list=[seed_review], num_seq=noise_config.max_reviews_per_node*2)
        if noise_config.near_duplicates_configuration.title is not None:
            new_titles = self._paraphrase(model, tokenizer, text_list=[seed_title], num_seq=noise_config.max_reviews_per_node*2)

        df_grouped = df.groupby(target)

        start_ts = parse_timestamp(noise_config.temporal_interval.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_interval.end_timestamp)

        for node in nodes:
            if remaining <= 0:
                break
            seed_review = noise_config.near_duplicates_configuration.review
            seed_title = noise_config.near_duplicates_configuration.title
            seed_rating = noise_config.near_duplicates_configuration.rating
            if node not in df_grouped.groups:
                continue

            node_idx = df_grouped.groups[node]
            node_df = df.loc[node_idx]
            used = node_df[other].unique()

            # Evita duplicati se richiesto
            available = np.setdiff1d(all_other, used) if getattr(config, "avoid_duplicates", False) else all_other
            if len(available) == 0:
                continue

            # Numero di review da aggiungere
            # n = per_node_budget(len(node_df), remaining, noise_config.min_reviews_per_node,
            #                     noise_config.max_reviews_per_node)

            n = np.random.randint(noise_config.min_reviews_per_node,noise_config.max_reviews_per_node+1)
            n = min(n,remaining)

            if n <= 0:
                continue

            # selezione degli altri (user o item)
            sampled_other = np.random.choice(available, size=n, replace=False)

            # SEED: scegli la review con rating mediano

            if not seed_review and not seed_title:
                seed_rating = noise_config.near_duplicates_configuration.rating

                median_review_candidates = node_df[
                    (node_df['rating'] == seed_rating)
                    ]['review_text'].tolist()

                median_title_candidates = node_df[node_df['rating'] == seed_rating]['title'].tolist()

                if median_review_candidates:
                    sel = random.choice(median_review_candidates)
                    seed_review = sel
                    idx = median_review_candidates.index(sel)
                    seed_title = median_title_candidates[idx] if median_title_candidates else ''
                else:
                    continue
                new_reviews = self._paraphrase(model, tokenizer, [seed_review], num_seq=n)
                new_titles = self._paraphrase(model, tokenizer, [seed_title], num_seq=n) if seed_title else [''] * n

            if not seed_title:
                new_titles = [''] * n

                # Genera timestamps in batch
            if start_ts == 0 and end_ts == 0:
                timestamps = np.full(n, int(pd.Timestamp.now().timestamp()), dtype=int)
            else:
                timestamps = np.array([
                    int((start_ts + (end_ts - start_ts) * random.random()).timestamp())
                    for _ in range(n)
                ], dtype=int)

            np.random.shuffle(new_reviews)
            np.random.shuffle(new_titles)
            # print(node)
            # print(n)
            # print(seed_review)
            # for p in new_reviews:
            #     print(p)
            # print(len(new_reviews))
            # print(len(new_reviews[0:n]))
            # print(len(new_titles[0:n]))
            # print(len(sampled_other))
            # print(len(timestamps))
            # print('\n\n')
            sampled_other = sampled_other[0:min(n,len(new_reviews))]
            timestamps = timestamps[0:min(n,len(new_reviews))]
            rows.append(pd.DataFrame({
                target: node,
                other: sampled_other,
                'rating': seed_rating,
                'timestamp': timestamps,
                'review_text': new_reviews[:n],
                'title': new_titles[:n],
                'noise': True
            }))
            remaining -= n
        df['noise'] = False
        added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
        return pd.concat([df, added], ignore_index=True), added




    def _add_sentence_noise(self, df, nodes, target, other, config, noise_config):
        df['noise'] = False
        remaining = self.budget
        mod_idx = []

        start_ts = parse_timestamp(noise_config.temporal_interval.start_timestamp)
        end_ts = parse_timestamp(noise_config.temporal_interval.end_timestamp)

        # Prepara pool di parole
        nltk.download('words', quiet=True)
        word_list_all = nltk_words.words()
        random.shuffle(word_list_all)
        pool = [p for p in random.sample(word_list_all, 300) if len(p) > 5]

        if noise_config.noise_type == 'sentence_noise':
            generator = pipeline("text-generation", model=noise_config.model, device=0)  # usa GPU se disponibile
            pool = self._generate_pool(generator, depth=self.budget*2, batch_size=128,topics = pool)

        df_grouped = df.groupby(target)

        for node in nodes:
            if remaining <= 0:
                break

            node_idx = df_grouped.groups[node]
            node_df = df.loc[node_idx]

            n = per_node_budget(len(node_df), remaining, noise_config.min_reviews_per_node,
                                noise_config.max_reviews_per_node)
            if n <= 0:
                continue

            ratings = sample_ratings(node_df['rating'], n, noise_config)

            # Candidati filtrati
            mask = node_df['rating'].isin(ratings)
            if start_ts != 0 and end_ts != 0:
                mask &= (node_df['timestamp'] >= start_ts) & (node_df['timestamp'] <= end_ts)

            candidates_idx = node_df.index[mask]
            if len(candidates_idx) == 0:
                continue
            # Campiona indici
            sampled_idx = np.random.choice(candidates_idx, size=min(len(candidates_idx), n), replace=False)
            mod_idx.extend(sampled_idx)
            df.loc[sampled_idx, 'noise'] = True
            remaining -= len(sampled_idx)

            # Vettorizza le review
            review_texts = df.loc[sampled_idx, 'review_text'].tolist()
            titles = df.loc[sampled_idx, 'title'].tolist() if 'title' in df.columns else [''] * len(sampled_idx)

            if noise_config.noise_type == 'shuffle':
                # Shuffle words vettoriale
                corrupted_reviews = [' '.join(np.random.permutation(r.split())) if len(r.split()) > 1 else r
                                     for r in review_texts]
                corrupted_titles = [' '.join(np.random.permutation(t.split())) if len(t.split()) > 1 else t
                                    for t in titles]

            elif noise_config.noise_type == 'sentence_noise':
                corrupted_reviews = []
                for t in review_texts:
                    parts = [p for p in re.split(r'[.!?]', t) if p]
                    new_text = []
                    for p in parts:
                        new_text.append(p)
                        new_text.append('.')
                        new_text.append(random.choice(pool))
                    corrupted_reviews.append(' '.join(new_text))
                for t in titles:
                    parts = [p for p in re.split(r'[.!?]', t) if p]
                    new_text = []
                    for p in parts:
                        new_text.append(p)
                        new_text.append('.')
                        new_text.append(random.choice(pool))
                    corrupted_titles.append(' '.join(new_text))
                #corrupted_titles = titles  # mantieni i titoli
            else:
                corrupted_reviews = []
                intensity_map = {'low': 7, 'medium': 5, 'high': 2}
                up = intensity_map.get(noise_config.intensity, 5)
                for t in review_texts:
                    parts = [p for p in t.split() if p]
                    final_rev = []
                    c = 0
                    for p in parts:
                        if c < up:
                            final_rev.append(p)
                            c += 1
                        else:
                            final_rev.append(random.choice(pool))
                            c = 0
                    corrupted_reviews.append(' '.join(final_rev))
                for t in titles:
                    parts = [p for p in t.split() if p]
                    final_rev = []
                    c = 0
                    for p in parts:
                        if c < up:
                            final_rev.append(p)
                            c += 1
                        else:
                            final_rev.append(random.choice(pool))
                            c = 0
                    corrupted_titles.append(' '.join(final_rev))
                #corrupted_titles = titles
            df.loc[sampled_idx, 'review_text'] = corrupted_reviews
            if 'title' in df.columns:
                df.loc[sampled_idx, 'title'] = corrupted_titles

        df['noise'] = False
        mod = df.loc[mod_idx]
        df.loc[mod_idx,'noise'] = True
        return df, mod



    def _generate_pool(self,generator,depth,batch_size=64,topics = None):
        # Process in batches
        if self.device == 'cpu':
            depth = 10
        if topics is None:
            topics = [
                "breakfast", "lunch", "dinner", "snack", "dessert",
                "coffee", "tea", "water", "juice", "milk",
                "bread", "cheese", "eggs", "bacon", "pancakes",
                "sandwich", "salad", "soup", "pizza", "pasta",
                "rice", "beans", "chicken", "beef", "fish",
                "fruits", "vegetables", "yogurt", "butter", "honey",
                "jam", "cereal", "oatmeal", "nuts", "chocolate",
                "cookies", "cake", "pie", "muffin", "croissant",
                "sauce", "spices", "salt", "pepper", "oil",
                "buttermilk", "smoothie", "granola", "tea-time", "doughnut"
            ]

        num_seq = depth / len(topics)

        batch_prompts = []
        for topic in topics:
            prompt = f"""Write a short sentence of about 15 words about {topic}.
            Returned sentence:
            """
            batch_prompts.append(prompt)
        # for topic in topics:
        #     messages = [
        #         {"role": "system", "content": "You generate short coherent sentences."},
        #         {"role": "user", "content": f"Write a short sentence of 15 words about {topic}."}
        #     ]
        #     prompt = generator.tokenizer.apply_chat_template(
        #         messages, tokenize=False, add_generation_prompt=True
        #     )
        #     batch_prompts.append(prompt)
        res = generator(
            batch_prompts,
            do_sample=True,
            max_new_tokens=50,
            num_return_sequences=num_seq,
            temperature=1.0,
            top_k=50,
            top_p=0.95
        )
        #results = [r[0]['generated_text'].split('<|assistant|>\n')[1] for r in res]
        #results = [out["generated_text"].split('Returned sentence:\n')[-1].replace('"', '') for out in res]
        results = [out["generated_text"].split('Returned sentence: \n')[-1].split('\n')[0].replace('"', '') for out in res]

        return results

    def _paraphrase(self,model,tokenizer,text_list,batch_size=64,num_seq = 2):
        outputs = []

        self.logger.info("Paraphrasing...")

        text = "paraphrase: " + text_list[0]
        encoding = tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        beam_outputs = model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            do_sample=True,
            temperature=0.9,
            max_length=256,
            top_k=120,
            top_p=0.98,
            early_stopping=True,
            num_return_sequences=num_seq
        )


        final_outputs = []
        for beam_output in beam_outputs:
            sent = tokenizer.decode(beam_output, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            final_outputs.append(sent)
        return final_outputs

