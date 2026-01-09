# Copyright 2018 The Cornac Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Example for Convolutional Matrix Factorization"""
import os
import argparse
import pandas as pd
import cornac
from cornac.data import Reader
from cornac.datasets import movielens
from cornac.eval_methods import RatioSplit
from cornac.data import TextModality
from cornac.data.text import BaseTokenizer
import ssl
from cornac.data import ReviewModality

from tensorflow.python.client import device_lib

from cornac.data import Dataset
import certifi
from cornac.models import BPR, ConvMF, MostPop, LightGCN,GRU4Rec,NARRE
from cornac.metrics import Recall, NDCG, RMSE
from ray.tune.examples.pbt_dcgan_mnist.common import batch_size
from cornac.eval_methods import NextItemEvaluation
import torch
from torch.cuda import device
import tensorflow as tf
base_dir = os.path.dirname(os.path.abspath(__file__))
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
gpus = tf.config.list_physical_devices('GPU')
print("Num GPUs Available:", len(gpus))
print("Devices:", gpus)
print(device_lib.list_local_devices())

VALID_MODELS = ['Pop','BPR','LightGCN','GRU4Rec']
    #,'CTR','HFT']

# ConvMF extends matrix factorization to leverage item textual information
# The necessary data can be loaded as follows
# Definisci le metriche
recall_10 = Recall(k=10)
recall_50 = Recall(k=50)
ndcg_10 = NDCG(k=10)
ndcg_50 = NDCG(k=50)

metrics_list = [recall_10, recall_50, ndcg_10, ndcg_50]


def load_vectors():
    import numpy as np

    embedding_path = os.path.join(base_dir, "cornac_data","wiki_giga_2024_100_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05.050_combined.txt")
    embedding_dim = 100  # controlla che siano 100, come nel nome del file

    pretrained_word_embeddings = {}

    with open(embedding_path, "r", encoding="utf-8") as f:
        first_line = f.readline()

        # Controlla se è header (tipo: "2000000 100")
        if len(first_line.split()) == 2:
            pass  # header presente, ignora prima riga
        else:
            # prima riga è embedding reale
            f.seek(0)  # torna all'inizio del file

        # Leggi tutte le righe
        for line in f:
            parts = line.rstrip().split(" ")
            word = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            if len(vector) == embedding_dim:
                pretrained_word_embeddings[word] = vector
        return pretrained_word_embeddings

def create_dataset(df_path,dataset):
    if not os.path.isfile(os.path.join(df_path, "inter_rating.csv")):
        inter_csv = os.path.join(df_path, "inter_full.csv")
        inter_df = pd.read_csv(inter_csv)
        inter_df_rating = inter_df[["user_id", "item_id", "rating"]]
        print(inter_df_rating.shape)

        # Rimuovi tutte le righe con valori mancanti in colonne chiave
        df_clean = inter_df_rating.dropna(subset=['user_id', 'item_id', 'rating'])

        # Assicurati che 'rating' sia float
        df_clean['rating'] = df_clean['rating'].astype(float)
        print(df_clean.shape)
        df_clean = df_clean.drop_duplicates()
        print(df_clean.shape)
        df_clean.to_csv(os.path.join(df_path, "inter_rating.csv"), index=False)
        print(df_clean.shape)


def run_ctr(df_path):
    items_path = os.path.join(df_path, "items_full.csv")
    items_df = pd.read_csv(items_path, sep=',', quotechar='"')
    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)

    items_df = items_df.sort_values("item_id").reset_index(drop=True)
    # Lista di descrizioni e lista di item_ids in corrispondenza
    items_descriptions = items_df['description'].tolist()
    items_ids = items_df['item_id'].tolist()
    # ---- Step 2: crea TextModality ----
    item_text_modality = TextModality(
        corpus=items_descriptions,
        ids=items_ids,
        tokenizer=BaseTokenizer(sep=' ', stop_words='english'),
        max_vocab=8000,
        max_doc_freq=0.5
    )



    # ---- Step 5: definisci metodo di valutazione ----
    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        item_text=item_text_modality,
        verbose=True,
        seed=123
    )

    # ---- Step 6: istanzia modello ConvMF ----
    ctr = cornac.models.CTR(k=50, max_iter=50, lambda_v=1)
    if dataset == 'amazon_beauty':
        # CTR – piccolo dataset
        ctr = cornac.models.CTR(
            k=30,
            max_iter=25,
            lambda_v=1.0,
            seed=123,
        )

    elif dataset == 'amazon_baby':
        # CTR – dataset medio
        ctr = cornac.models.CTR(
            k=30,
            max_iter=25,
            lambda_v=1.0,
            seed=123,
        )

    elif dataset == 'yelp':
        # CTR – dataset grande
        ctr = cornac.models.CTR(
            k=30,
            max_iter=25,
            lambda_v=1.0,
            seed=123,
        )



    # ---- Step 8: esperimento Cornac ----
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[ctr],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()


def run_narre(df_path):
    # --- Step 1: leggi le review user-item ---
    reviews_path = os.path.join(df_path, "inter_full.csv")
    reviews_df = pd.read_csv(reviews_path, sep=',', quotechar='"')



    vocab = 5000
    if dataset == 'yelp':
        vocab = 8000



    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)
    reviews_tuples = list(zip(reviews_df["user_id"], reviews_df["item_id"], reviews_df["review_text"].astype(str)))

    review_modality = ReviewModality(
        data=reviews_tuples,
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=vocab,
        max_doc_freq=0.5,
    )

    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        review_text=review_modality,
        verbose=True,
        seed=123,
    )
    pretrained_word_embeddings = load_vectors()
    if dataset == 'amazon_beauty':  # ~10k–50k review
        model = NARRE(
            embedding_size=100,
            id_embedding_size=16,
            n_factors=16,
            attention_size=8,
            kernel_sizes=[3],
            n_filters=32,
            dropout_rate=0.5,
            max_text_length=50,
            max_num_review=10,  # 🔥 FONDAMENTALE
            batch_size=32,
            max_iter=20,  # 15 è inutile all'inizio
            init_params={'pretrained_word_embeddings': pretrained_word_embeddings},
            verbose=True,
            seed=123,
        )

    elif dataset == 'amazon_baby':  # ~50k–200k review
        model = NARRE(
            embedding_size=50,
            id_embedding_size=16,
            n_factors=16,
            attention_size=8,
            kernel_sizes=[3],
            n_filters=32,
            dropout_rate=0.5,
            max_text_length=40,
            max_num_review=5,  # 🔥 ANCORA PIÙ IMPORTANTE
            batch_size=32,
            max_iter=8,
            verbose=True,
            seed=123,
        )

    else:  # >200k review (1M incluso)
        model = NARRE(
            embedding_size=50,
            id_embedding_size=16,
            n_factors=16,
            attention_size=8,
            kernel_sizes=[3],
            n_filters=32,
            dropout_rate=0.5,
            max_text_length=30,
            max_num_review=3,  # 🔥 O MUORE
            batch_size=32,
            max_iter=5,
            verbose=True,
            seed=123,
        )

    # ---- Step 8: esperimento Cornac ----
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[model],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()


def run_hft(df_path):
    # --- Step 1: leggi le review user-item ---
    reviews_path = os.path.join(df_path, "inter_full.csv")
    reviews_df = pd.read_csv(reviews_path, sep=',', quotechar='"')


    reviews_df["review_text"] = reviews_df["review_text"].astype(str)
    # --- Step 2: aggrega tutte le review per item ---
    aggregated_reviews = (
        reviews_df
        .groupby("item_id")["review_text"]
        .apply(lambda texts: " ".join(str(t) for t in texts))
        .reset_index()
    )
    vocab = 3000
    if dataset == 'yelp':
        vocab = 5000
    elif dataset == 'amazon_baby':
        vocab = 5000

    # --- Step 3: crea TextModality dalle review aggregate ---
    item_text_modality = TextModality(
        corpus=aggregated_reviews["review_text"].tolist(),
        ids=aggregated_reviews["item_id"].tolist(),
        tokenizer=BaseTokenizer(stop_words="english"),
        max_vocab=vocab,
        max_doc_freq=0.5
    )

    # item_text_modality = TextModality(
    #     corpus=reviews_df["review_text"].tolist(),
    #     ids=list(zip(reviews_df["user_id"], reviews_df["item_id"])),
    #     tokenizer=BaseTokenizer(stop_words="english"),
    #     max_vocab=vocab,
    #     max_doc_freq=0.3
    # )


    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)

    # ---- Step 5: definisci metodo di valutazione ----
    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        item_text=item_text_modality,
        verbose=True,
        seed=123
    )

    # reviews_path = os.path.join(df_path, "inter_full.csv")
    # reviews_df = pd.read_csv(reviews_path, sep=',', quotechar='"')
    #
    #
    #
    # vocab = 5000
    # if dataset == 'yelp':
    #     vocab = 8000
    #
    #
    #
    # data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)
    # reviews_tuples = list(zip(reviews_df["user_id"], reviews_df["item_id"], reviews_df["review_text"].astype(str)))
    #
    # review_modality = ReviewModality(
    #     data=reviews_tuples,
    #     tokenizer=BaseTokenizer(stop_words="english"),
    #     max_vocab=vocab,
    #     max_doc_freq=0.5,
    # )
    #
    # ratio_split = RatioSplit(
    #     data=data,
    #     test_size=0.2,
    #     #val_size=0.1,
    #     exclude_unknowns=True,
    #     review_text=review_modality,
    #     verbose=True,
    #     seed=123,
    # )

    if dataset == 'amazon_beauty':
        hft = cornac.models.HFT(
            k=5,  # pochi fattori per velocità
            max_iter=20,  # iterazioni brevi
            grad_iter=2,  # pochi aggiornamenti per batch
            l2_reg=0.001,
            lambda_text=0.01,  # testo moderato
            vocab_size=min(3000, vocab),  # ridotto
            seed=123,
        )

    elif dataset == 'amazon_baby':
        hft = cornac.models.HFT(
            k=10,  # più fattori
            max_iter=40,  # iterazioni medie
            grad_iter=5,  # aggiornamenti batch
            l2_reg=0.001,
            lambda_text=0.01,
            vocab_size=min(5000, vocab),  # vocabolario moderato
            seed=123,
        )

    elif dataset == 'yelp':
        hft = cornac.models.HFT(
            k=10,  # più fattori per dataset grande
            max_iter=40,  # più iterazioni
            grad_iter=5,  # batch grad più numerosi
            l2_reg=0.001,
            lambda_text=0.01,  # testo rilevante ma non eccessivo
            vocab_size=min(5000, vocab),  # vocabolario grande
            seed=123,
        )

    # if dataset == 'amazon_beauty':
    #     # HFT per amazon_beauty (piccolo dataset)
    #     hft = cornac.models.HFT(
    #         k=8,
    #         max_iter=100,
    #         grad_iter=5,
    #         l2_reg=0.01,
    #         lambda_text=0.02,
    #         vocab_size=3000,
    #         seed=123,
    #     )
    #
    # elif dataset == 'amazon_baby':
    #     # HFT per amazon_baby (dataset medio)
    #     hft = cornac.models.HFT(
    #         k=18,
    #         max_iter=100,
    #         grad_iter=7,
    #         l2_reg=0.001,
    #         lambda_text=0.03,
    #         vocab_size=7000,
    #         seed=123,
    #     )
    #
    # elif dataset == 'yelp':
    #     # HFT per yelp (dataset grande)
    #     hft = cornac.models.HFT(
    #         k=25,
    #         max_iter=100,
    #         grad_iter=10,
    #         l2_reg=0.0005,
    #         lambda_text=0.015,
    #         vocab_size=10000,
    #         seed=123,
    #     )



    # ctr = cornac.models.CTR(k=50, max_iter=50, lambda_v=1)
    # if dataset == 'amazon_beauty':
    #     # CTR – piccolo dataset
    #     ctr = cornac.models.CTR(
    #         k=30,
    #         max_iter=25,
    #         lambda_v=1.0,
    #         seed=123,
    #     )
    #
    # elif dataset == 'amazon_baby':
    #     # CTR – dataset medio
    #     ctr = cornac.models.CTR(
    #         k=30,
    #         max_iter=25,
    #         lambda_v=1.0,
    #         seed=123,
    #     )
    #
    # elif dataset == 'yelp':
    #     # CTR – dataset grande
    #     ctr = cornac.models.CTR(
    #         k=30,
    #         max_iter=25,
    #         lambda_v=1.0,
    #         seed=123,
    #     )



    # ---- Step 8: esperimento Cornac ----
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[hft],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()


def run_bpr(df_path,dataset):
    create_dataset(df_path,dataset)
    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)

    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        verbose=True,
        seed=123
    )

    if dataset == 'amazon_beauty':
        bpr = BPR(
            k=50,
            max_iter=100,
            learning_rate=0.01,
            lambda_reg=0.001,
            seed=123,
            verbose=True
        )

    elif dataset == 'amazon_baby':
        bpr = BPR(
            k=80,
            max_iter=200,
            learning_rate=0.001,
            lambda_reg=0.0001,
            seed=123,
            verbose=True
        )

    elif dataset == 'yelp':
        bpr = BPR(
            k=100,
            max_iter=300,
            learning_rate=0.0001,
            lambda_reg=0.0001,
            seed=123,
            verbose=True
        )

    # ---- Step 4: istanzia modello BPR ----
    # if dataset == 'amazon_beauty':
    #     bpr = BPR(
    #         k=100,
    #         max_iter=100,
    #         learning_rate=0.01,
    #         lambda_reg=0.0001,
    #         verbose=True,
    #         seed=123
    #     )
    #
    # elif dataset == 'yelp':
    #     bpr = BPR(
    #         k=100,
    #         max_iter=100,
    #         learning_rate=0.01,
    #         lambda_reg=0.005,
    #         verbose=True,
    #         seed=123
    #     )
    # else:
    #     bpr = BPR(
    #         k=80,
    #         max_iter=100,
    #         learning_rate=0.005,
    #         lambda_reg=0.001,
    #         verbose=True,
    #         seed=123
    #     )
    # ---- Step 5: metriche ranking ----
    metrics_list = [Recall(k=10), Recall(k=50), NDCG(k=10), NDCG(k=50)]

    # ---- Step 6: esperimento Cornac ----
    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[bpr],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()


def run_pop(df_path, dataset):
    create_dataset(df_path,dataset)
    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)

    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        verbose=True,
        seed=123
    )

    pop_model = MostPop()
    metrics_list = [Recall(k=10), Recall(k=50), NDCG(k=10), NDCG(k=50)]

    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[pop_model],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()

def run_gru4rec(df_path, dataset):
    # Carica il dataset
    if os.path.exists(os.path.join(df_path, "inter_sequential.csv")):
        df = pd.read_csv(os.path.join(df_path, "inter_full.csv"))

        SESSION_GAP = 60 * 60 * 24 # 30 minuti (secondi)

        # carica CSV
        # tieni solo colonne utili
        df = df[["user_id", "item_id", "timestamp"]]
        df["timestamp"] = df["timestamp"].fillna(0)
        df["timestamp"] = df["timestamp"].round(0).astype(int)

        # ms → secondi
        df["timestamp"] = (df["timestamp"] // 1000).astype(int)

        # ordina per utente + tempo
        df = df.sort_values(["user_id", "timestamp"])
        df = df[['user_id', 'item_id', 'timestamp']]
        max_len = 20
        df = df.groupby('user_id').tail(10)

        df.to_csv(os.path.join(df_path, "inter_sequential.csv"), index=False)
        print('saved')
        # salva

        df = pd.read_csv(os.path.join(df_path, "inter_sequential.csv"))[['user_id','item_id','timestamp']]

        df = df.sort_values(["user_id", "timestamp"])

        # prendi l'ultimo evento di ogni sessione come test
        if dataset == 'amazon_beauty':
            k = 2
        elif dataset == 'amazon_baby':
            k = 2
        else:
            k = 2
        test_data = df.groupby("user_id").tail(k)

        # tutto il resto è train
        train_data = df.drop(test_data.index)

        # opzionale: salva su file
        train_data.to_csv(os.path.join(df_path, "train_data.csv"), index=False, header=False)
        test_data.to_csv(os.path.join(df_path, "test_data.csv"), index=False, header=False)
        print('saved')
    train_data = Reader().read(os.path.join(df_path, "train_data.csv"),fmt="SIT", sep=',',skip_lines=1)
    test_data = Reader().read(os.path.join(df_path, "test_data.csv"),fmt="SIT", sep=',',skip_lines=1)

    next_item_eval = NextItemEvaluation.from_splits(
        train_data=train_data,
        test_data=test_data,
        exclude_unknowns=True,
        verbose=True,
        seed=123
    )
    # print("Num users train:", next_item_eval.train_set.num_users)
    # print("Num items train:", next_item_eval.train_set.num_items)
    # print("Num ratings train:", next_item_eval.train_set.num_ratings)
    # print("Num users test:", next_item_eval.test_set.num_users)
    # print("Num ratings test:", next_item_eval.test_set.num_ratings)

    # Parametri GRU4Rec adattati per dataset size
    if dataset == 'amazon_beauty':
        gru_model = GRU4Rec(
            layers=[100],
            loss="bpr-max",
            n_sample=1024,
            batch_size=64,
            n_epochs=10,
            seed=123,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=True
        )
        gru_model = GRU4Rec(
            layers=[64],
            loss="cross-entropy",
            batch_size=64,
            n_epochs=50,  # prova anche 30
            learning_rate=0.01,  # prova anche 0.005
            dropout_p_hidden=0.0,  # o 0.1
            logq=1.0,
            constrained_embedding=True,
            seed=123,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    elif dataset == 'amazon_baby':
        gru_model = GRU4Rec(
            layers=[100],
            loss="bpr-max",
            n_sample=2048,
            batch_size=512,
            n_epochs=50,
            seed=123,
            learning_rate=0.001,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        gru_model = GRU4Rec(
            layers=[50],  # ↓
            loss="cross-entropy",  # CAMBIO CRITICO
            batch_size=256,
            n_epochs=100,
            learning_rate=0.003,  # ↑
            seed=123,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        gru_model = GRU4Rec(
            layers=[100],
            loss="cross-entropy",
            batch_size=128,
            n_epochs=80,  # puoi provare 50 / 80 / 100
            learning_rate=0.005,  # leggermente più alto di 0.003
            dropout_p_hidden=0.1,
            logq=1.0,
            constrained_embedding=True,
            seed=123,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    elif dataset == 'yelp':
        gru_model = GRU4Rec(
            layers=[128],
            loss="bpr-max",
            n_sample=4096,
            batch_size=2048,
            n_epochs=100,
            learning_rate=0.0005,
            seed=123,
            verbose=True,device="cuda" if torch.cuda.is_available() else "cpu",
        )
        gru_model = GRU4Rec(
            layers=[100],
            loss="cross-entropy",  # ← fondamentale
            batch_size=1024,
            n_epochs=150,
            learning_rate=0.001,
            seed=123,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )
        gru_model = GRU4Rec(
            layers=[100],
            loss="cross-entropy",
            batch_size=1024,
            n_epochs=80,  # puoi provare 50 / 80 / 100
            learning_rate=0.0005,  # leggermente più alto di 0.003
            dropout_p_hidden=0.1,
            logq=1.0,
            constrained_embedding=True,
            seed=123,
            verbose=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

    # if dataset == 'amazon_beauty':  # piccolo
    #     gru_model = GRU4Rec(
    #         layers=[100],
    #         loss="bpr-max",
    #         n_sample=1024,
    #         dropout_p_embed=0.0,
    #         dropout_p_hidden=0.5,
    #         sample_alpha=0.75,
    #         batch_size=64,
    #         n_epochs=2,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         verbose=True,
    #         seed=123
    #     )
    # elif dataset == 'amazon_baby':  # medio
    #     gru_model = GRU4Rec(
    #         layers=[100, 100],
    #         loss="bpr-max",
    #         n_sample=2048,
    #         dropout_p_embed=0.0,
    #         dropout_p_hidden=0.5,
    #         sample_alpha=0.75,
    #         batch_size=512,
    #         n_epochs=30,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         verbose=True,
    #         seed=123
    #     )
    # else:  # grande
    #     gru_model = GRU4Rec(
    #         layers=[200, 200],
    #         loss="bpr-max",
    #         n_sample=4096,
    #         dropout_p_embed=0.0,
    #         dropout_p_hidden=0.5,
    #         sample_alpha=0.75,
    #         batch_size=1024,
    #         n_epochs=50,
    #         device="cuda" if torch.cuda.is_available() else "cpu",
    #         verbose=True,
    #         seed=123
    #     )

    # Metriche per next-item/ranking
    metrics_list = [Recall(k=10), Recall(k=50), NDCG(k=10), NDCG(k=50)]

    exp = cornac.Experiment(
        eval_method=next_item_eval,
        models=[gru_model],
        metrics=metrics_list,
    )


    exp.run()




def run_lightgcn(df_path, dataset):
    create_dataset(df_path,dataset)
    # inter_df = inter_df.rename(columns={"user_id": "u_id", "item_id": "i_id", "rating": "rating"})
    # inter_df['rating'] = 1.0


    data = Reader().read(os.path.join(df_path, "inter_rating.csv"), sep=',',skip_lines=1)

    ratio_split = RatioSplit(
        data=data,
        test_size=0.2,
        exclude_unknowns=True,
        verbose=True,
        seed=123
    )

    # Parametri LightGCN (puoi personalizzare k/epochs se vuoi)
    if dataset == 'amazon_beauty':
        lgn_model = LightGCN(
            emb_size=64,
            num_layers=2,
            num_epochs=30,
            learning_rate=0.01,
            lambda_reg=0.002,
            batch_size=2048,
            seed=123,
            verbose=True
        )

    elif dataset == 'amazon_baby':
        lgn_model = LightGCN(
            emb_size=64,
            num_layers=2,
            num_epochs=40,
            learning_rate=0.005,
            lambda_reg=1e-4,
            batch_size=4096,
            seed=123,
            verbose=True
        )
        lgn_model = LightGCN(
            emb_size=64,
            num_layers=3,
            num_epochs=50,
            learning_rate=0.001,
            lambda_reg=1e-4,
            batch_size=4096,
            seed=123,
            verbose=True
        )

    elif dataset == 'yelp':
        lgn_model = LightGCN(
            emb_size=64,
            num_layers=3,
            num_epochs=50,
            learning_rate=0.001,
            lambda_reg=1e-4,
            batch_size=4096,
            seed=123,
            verbose=True
        )

    # if dataset == 'amazon_beauty':  # piccolo dataset
    #     lgn_model = LightGCN(
    #         emb_size=32,
    #         num_layers=1,
    #         num_epochs=50,
    #         learning_rate=0.01,
    #         lambda_reg=0.02,
    #         batch_size=2048,
    #         verbose=True,
    #         seed=123
    #     )
    # elif dataset == 'amazon_baby':  # medio dataset
    #     lgn_model = LightGCN(
    #         emb_size=64,
    #         num_layers=2,
    #         num_epochs=100,
    #         learning_rate=0.005,
    #         lambda_reg=1e-4,
    #         batch_size=4096,
    #         verbose=True,
    #         seed=123
    #     )
    # else:  # grande dataset
    #     lgn_model = LightGCN(
    #         emb_size=64,
    #         num_layers=3,
    #         num_epochs=100,
    #         learning_rate=0.001,
    #         lambda_reg=1e-4,
    #         batch_size=4096,
    #         verbose=True,
    #         seed=123
    #     )

    metrics_list = [Recall(k=10), Recall(k=50), NDCG(k=10), NDCG(k=50)]

    exp = cornac.Experiment(
        eval_method=ratio_split,
        models=[lgn_model],
        metrics=metrics_list,
        user_based=True
    )

    exp.run()




if __name__ == "__main__":
    # POP: alla me del futuro: se runno amazon_beauty_5 con [0,inf] sul yaml, dà lo stesso risultato di [5,inf] con amazon_beauty

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        nargs='?',  # <--- rende l'argomento posizionale opzionale
        default='BPR',  # usato se non viene passato da CLI
    )

    parser.add_argument(
        'dataset',
        nargs='?',  # <--- anche questo opzionale
        default='amazon_beauty',
        help='Dataset'
    )
    parser.add_argument(
        'type',
        nargs='?',  # <--- anche questo opzionale
        default='all',
        help='Type'
    )
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    if dataset == 'all':
        datasets = ['amazon_beauty', 'amazon_baby', 'yelp']
    else:
        datasets = [dataset]

    if model == 'all':
        models = VALID_MODELS
    else:
        models = [model]

    for model in models:
        for dataset in  datasets:  # per ora fisso qui
            if model == 'Pop':
                run_pop(os.path.join(base_dir, "cornac_data", dataset), dataset)
            elif model ==  'BPR':
                run_bpr(os.path.join(base_dir, "cornac_data", dataset), dataset)
            elif model == 'LightGCN':
                run_lightgcn(os.path.join(base_dir, "cornac_data", dataset), dataset)
            elif model == 'GRU4Rec':
                run_gru4rec(os.path.join(base_dir, "cornac_data", dataset), dataset)
            elif model == 'CTR':
                run_ctr(os.path.join(base_dir, "cornac_data", dataset))
            elif model == 'HFT':
                run_hft(os.path.join(base_dir, "cornac_data", dataset))
            elif model == 'NARRE':
                run_narre(os.path.join(base_dir, "cornac_data", dataset))