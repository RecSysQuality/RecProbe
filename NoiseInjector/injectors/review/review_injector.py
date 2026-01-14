
import numpy as np
import pandas as pd
import random
from ..base import BaseNoiseInjector
import random
import nltk
from nltk.corpus import wordnet
import json
# scarica WordNet se non già presente
nltk.download('wordnet')
nltk.download('omw-1.4')


class ReviewNoiseInjector(BaseNoiseInjector):

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
        if ctx == "remove_reviews":
            return self._remove_noise(df,target=ctx.target)
        if ctx == "review_burst_noise":
            return self._burst_noise(df, target='user_id')
        if ctx == "sentence_noise":
            return self._sentence_noise(df, target='item_id')

        raise ValueError(f"Unknown rating noise context: {ctx}")

    # ==========================================================
    # REMOVE NOISE
    # ==========================================================
    def _remove_noise(self, df, target):
        config = self.config.realistic_noise
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        df = self._sample_reviews(df, config)
        return self._remove_reviews(df, nodes, target, other, config)


    # ==========================================================
    # BURST NOISE
    # ==========================================================
    def _burst_noise(self, df, target):
        config = self.config.realistic_noise
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        df = self._sample_reviews(df, config)
        return self._add_reviews(df, nodes, target, other, config)

    # ==========================================================
    # SENTENCE NOISE
    # ==========================================================
    def _sentence_noise(self, df, target):
        config = self.config.realistic_noise
        self.global_vocab = json.load(open(config.vocab_path, 'r'))
        other = 'item_id' if target == 'user_id' else 'user_id'
        nodes = self._ordered_nodes(df, target, config.selection_strategy)
        df = self._sample_reviews(df, config)
        return self._add_sentence_noise(df, nodes, target, other, config,global_vocab)



    def _remove_reviews(self, df, nodes, target,other, config):
        removed_idx = []
        remaining = self.budget

        start_ts = getattr(config, 'temporal_interval', {}).get('start_timestamp', 0)
        end_ts = getattr(config, 'temporal_interval', {}).get('end_timestamp', 2 ** 31 - 1)
        degrees = df.groupby(target)[other].nunique()
        max_degree = degrees.max()

        for node in nodes:
            if remaining <= 0:
                break

            node_df = df[df[target] == node]

            # Seleziona solo le recensioni nel range temporale
            candidates = node_df[
                (node_df['timestamp'] >= start_ts) &
                (node_df['timestamp'] <= end_ts)
                ]

            if candidates.empty:
                continue

            # Decidi quante recensioni "rimuovere" per questo nodo
            n = self._per_node_budget(len(candidates), remaining, config.per_node_limits)
            if n <= 0:
                continue

            if getattr(config, "preserve_degree_distribution", False):
                factor = (max_degree - degrees.get(node, 0)) / max_degree
                n = max(1, int(np.ceil(n * factor)))

            # Campiona casualmente le recensioni da svuotare
            sampled = candidates.sample(n=min(len(candidates), n), replace=False)
            removed_idx.extend(sampled.index)
            remaining -= len(sampled)

        # Svuota il testo delle recensioni selezionate
        df.loc[removed_idx, 'review_text'] = ""
        df.loc[removed_idx, 'title'] = ""

        return df

        # ==========================================================
        # ADD (shared by realistic + burst)
        # ==========================================================

    def _add_reviews(self, df, nodes, target, other, config):
        """
        Aggiunge review a un set di nodi per simulare review burst.

        - df: dataframe originale
        - nodes: lista di target nodes (item o user)
        - target: colonna target ('item' o 'user')
        - other: colonna complementare ('user' o 'item')
        - config: oggetto config con parametri del burst
        """
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
            if getattr(config, "avoid_duplicates", False):
                available = np.setdiff1d(all_other, used)

            if len(available) == 0:
                continue

            # determina numero di review da aggiungere
            n = self._per_node_budget(
                len(available),
                remaining,
                limits=(config.min_reviews_per_node, config.max_reviews_per_node)
            )

            # selezione degli altri (user o item)
            sampled_other = np.random.choice(available, size=n, replace=False)

            # SEED: scegli la review con rating mediano
            median_rating = node_df['rating'].median()
            median_review_candidates = node_df[node_df['rating'] == median_rating]['review'].tolist()
            if median_review_candidates.empty:
                seed_review = node_df['review'].sample(1).iloc[0]
                seed_title = node_df['title'].iloc[0] if 'title' in df.columns else ""
            else:
                selected = median_review_candidates.sample(1).iloc[0]
                seed_review = selected['review']
                seed_title = selected['title'] if 'title' in df.columns else ""

            # genera near-duplicate reviews
            new_reviews = []
            new_titles = []
            for _ in range(n):
                # Review
                dup_review = self._paraphrase_review(seed_review, replace_prob=0.35)
                while len(dup_review.split()) < config.min_length_of_review:
                    dup_review += " " + self._paraphrase_review(seed_review, replace_prob=0.35)
                new_reviews.append(dup_review)

                # Title (solo se esiste)
                if seed_title:
                    dup_title = self._paraphrase_review(seed_title, replace_prob=0.35)
                    new_titles.append(dup_title)
                else:
                    new_titles.append("")

            # genera ratings
            if config.rating_behavior.sampling_strategy == "gaussian":
                mu = median_rating
                sigma = 1
                sampled_ratings = np.clip(np.random.normal(mu, sigma, size=n),
                                          config.rating_behavior.min_rating,
                                          config.rating_behavior.max_rating)
                sampled_ratings = sampled_ratings.round().astype(int)
            else:  # uniform
                sampled_ratings = np.random.randint(config.rating_behavior.min_rating,
                                                    config.rating_behavior.max_rating + 1,
                                                    size=n)

            # genera timestamps
            start_ts = config.temporal_interval.start_timestamp
            end_ts = config.temporal_interval.end_timestamp
            if start_ts == 0 and end_ts == 0:
                timestamps = [pd.Timestamp.now().timestamp()] * n
            else:
                timestamps = np.random.randint(start_ts, end_ts + 1, size=n)

            # aggiungi al dataframe
            rows.append(pd.DataFrame({
                target: node,
                other: sampled_other,
                'rating': sampled_ratings,
                'timestamp': timestamps,
                'review': new_reviews,
                'title': new_titles
            }))

            remaining -= n

        added = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=df.columns)
        return pd.concat([df, added], ignore_index=True), added

    def _add_sentence_noise(self, df, nodes, target, other, config, global_vocab):
        """
        Applica rumore testuale alle recensioni:
        - shuffle: mescola le parole
        - sentence_noise: aggiunge frasi casuali dai template
        - token_noise: sostituisce parole con token casuali

        Parametri:
        - df: dataframe delle recensioni
        - nodes: nodi target (user_id o item_id)
        - target, other: colonne target e complementare
        - config: configurazione del noise
        - global_vocab: vocabolario globale per sentence/token noise
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
            if start_ts != 0 and end_ts != 0:
                candidates = node_df[
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
                review = sampled.at[idx, 'review_text']
                title = sampled.at[idx, 'title'] if 'title' in df.columns else ""

                if config.noise_type == 'shuffle':
                    # shuffle parole
                    words = review.split()
                    if len(words) > 1:
                        np.random.shuffle(words)
                    corrupted_review = " ".join(words)
                    while len(corrupted_review.split()) < config.min_length_of_review:
                        corrupted_review += " " + review
                    df.at[idx, 'review_text'] = corrupted_review

                elif config.noise_type == 'sentence_noise':
                    sentence_vocab = global_vocab.get('sentence_fragments', {})
                    templates = global_vocab.get('sentence_templates', [])
                    topics = [k for k in sentence_vocab.keys() if k != "sentence_templates"]

                    sentences = review.split(". ")  # recensione originale
                    title_sentences = title.split(". ") if title else []

                    n_sent = 2  # numero di frasi da aggiungere
                    for _ in range(n_sent):
                        topic = np.random.choice(topics)
                        fragments = sentence_vocab[topic]
                        template = np.random.choice(templates)
                        n_slots = template.count("{}")
                        fillers = np.random.choice(fragments, size=min(n_slots, len(fragments)), replace=False)
                        new_sentence = template.format(*fillers)

                        # inserisci casualmente
                        insert_idx = np.random.randint(0, len(sentences) + 1)
                        sentences.insert(insert_idx, new_sentence)
                        if title_sentences:
                            insert_idx_t = np.random.randint(0, len(title_sentences) + 1)
                            title_sentences.insert(insert_idx_t, new_sentence)

                    corrupted_review = ". ".join(sentences)
                    corrupted_title = ". ".join(title_sentences) if title_sentences else title

                    while len(corrupted_review.split()) < config.min_length_of_review:
                        corrupted_review += " " + review

                    df.at[idx, 'review_text'] = corrupted_review
                    if 'title' in df.columns:
                        df.at[idx, 'title'] = corrupted_title

                elif config.noise_type == 'token_noise':
                    token_vocab = global_vocab.get('token_fragments', [])
                    replace_prob = getattr(config, 'replace_prob', 0.2)

                    words = review.split()
                    words_title = title.split() if title else []
                    new_words = [np.random.choice(token_vocab) if np.random.rand() < replace_prob else w for w in words]
                    new_words_title = [np.random.choice(token_vocab) if np.random.rand() < replace_prob else w for w in
                                       words_title]

                    corrupted_review = " ".join(new_words)
                    corrupted_title = " ".join(new_words_title)

                    while len(corrupted_review.split()) < config.min_length_of_review:
                        corrupted_review += " " + review

                    df.at[idx, 'review_text'] = corrupted_review
                    if 'title' in df.columns:
                        df.at[idx, 'title'] = corrupted_title

        return df

    # ==========================================================
    # COMMON HELPERS
    # ==========================================================
    def _ordered_nodes(self, df, target, strategy):
        nodes = df[target].value_counts(ascending=strategy == 'least').index.tolist()
        if strategy == 'uniform':
            random.shuffle(nodes)
        return nodes

    def _sample_reviews(self, df, config):
        if config.min_length_of_review > 0:
            token_counts = df['review_text'].fillna("").str.split().str.len()
            # Filtra
            df_filtered = df[token_counts >= config.min_length_of_review].copy()
            return df_filtered
        return df

    def _per_node_budget(self, node_size, total_left, limits):
        n = np.random.randint(
            limits.min_reviews_per_node,
            limits.max_reviews_per_node + 1
        )
        return min(n, node_size, total_left)

    # ==========================================================
    # HELPERS FOR NEAR-DUPLICATE GENERATION
    # ==========================================================

    def _get_synonyms(self,word):
        """
        Restituisce una lista di sinonimi di una parola usando WordNet.
        """
        syns = set()
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                # rimuove eventuali underscore
                syns.add(lemma.name().replace("_", " "))
        return list(syns)


    def _paraphrase_review(self,text, replace_prob=0.2):
        """
        Genera una versione leggermente variata di `text` sostituendo alcune parole con sinonimi.

        Parametri:
        - text: stringa (review o title)
        - replace_prob: probabilità di sostituire ogni parola con un sinonimo

        Restituisce:
        - stringa paraphrased
        """
        tokens = text.split()
        new_tokens = []

        for tok in tokens:
            if random.random() < replace_prob:
                syns = self._get_synonyms(tok)
                if syns:
                    # scegli un sinonimo casuale
                    new_tokens.append(random.choice(syns))
                else:
                    new_tokens.append(tok)
            else:
                new_tokens.append(tok)

        return " ".join(new_tokens)

