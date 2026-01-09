import os.path
import time
import pandas as pd
import torch
from torch.utils.data import Dataset

import numpy as np
def date(f='%Y-%m-%d %H:%M:%S'):
    return time.strftime(f, time.localtime())


def load_optimal_embedding(data_paths, glove_path, target_vocab_size, word_dim=50):
    """
    Vocabolario da MULTIPLI CSV (train+val+test)
    """
    from collections import Counter
    import pandas as pd
    import torch.nn.init as init

    print("🔍 1. Vocabolario da tutti i dataset...")
    all_words = []

    for i, path in enumerate(data_paths):
        df = pd.read_csv(path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        for review in df['review'].dropna():
            if isinstance(review, str):
                all_words.extend(review.split())
        print(f"({len(df)} righe)")

    word_freq = Counter(all_words)
    total_words = sum(word_freq.values())
    dataset_vocab = [w for w, _ in word_freq.most_common(target_vocab_size)]
    coverage = sum(word_freq[w] for w in dataset_vocab) / total_words
    print(f"📊 {len(dataset_vocab)} parole → {coverage:.1%} coverage")

    print("📥 2. Caricamento GloVe...")
    glove_emb, glove_dict = load_embedding(glove_path)

    print("🎯 3. Embedding ottimale...")
    word_dict = {'<PAD>': 0, '<UNK>': 1}
    word_emb = [[0.0] * word_dim, [0.0] * word_dim]

    glove_hits = 0
    for word in dataset_vocab:
        if word in glove_dict:
            word_emb.append(glove_emb[glove_dict[word]])
        else:
            emb = torch.empty(word_dim)
            word_emb.append(emb.tolist())


        word_dict[word] = len(word_dict)

    print(f"✅ Vocabolario: {len(word_dict)} | GloVe: {glove_hits / len(dataset_vocab):.1%}")
    return word_emb, word_dict


from collections import Counter
import pandas as pd


def load_top_embeddings(data_paths,glove_path, top_k=1291150):
    """
    Carica embedding da file word2vec/GloVe
    e conserva solo le top_k parole più frequenti nei dataset.

    Args:
        word2vec_file: path al file di embedding
        data_paths: lista di CSV contenenti colonne ['userID','itemID','review','rating']
        top_k: numero di parole più frequenti da mantenere

    Returns:
        word_emb: lista di embedding
        word_dict: dizionario {parola: indice}
    """
    # 1️⃣ Costruisci vocabolario dai dataset
    all_words = []
    for path in data_paths:
        df = pd.read_csv(path, header=None, names=['userID', 'itemID', 'review', 'rating'])
        for review in df['review'].dropna():
            if isinstance(review, str):
                all_words.extend(review.split())
    word_freq = Counter(all_words)
    most_common_words = set([w for w, _ in word_freq.most_common(top_k)])
    print(len(most_common_words))
    # 2️⃣ Carica embedding da file e filtra
    word_emb = list()
    word_dict = dict()
    word_emb.append([0])
    word_dict['<UNK>'] = 0

    with open(glove_path, encoding='utf-8') as f:
        for line in f:
            tokens = line.split(' ')
            word = tokens[0]
            if word in most_common_words:
                #print(word)
                #print(list(most_common_words)[0:100])
            #     continue  # salta parole fuori dalle top_k
                word_emb.append([float(i) for i in tokens[1:]])
                word_dict[tokens[0]] = len(word_dict)

    # Correggi UNK embedding dimension
    word_emb[0] = [0] * len(word_emb[1])

    print(f"✅ Caricate {len(word_dict)} parole su {top_k} richieste")
    return word_emb, word_dict


def load_embedding(word2vec_file):
    with open(word2vec_file, encoding='utf-8') as f:
        word_emb = list()
        word_dict = dict()
        word_emb.append([0])
        word_dict['<UNK>'] = 0
        c = 0
        for line in f.readlines():
            tokens = line.split(' ')
            word_emb.append([float(i) for i in tokens[1:]])
            word_dict[tokens[0]] = len(word_dict)

        word_emb[0] = [0] * len(word_emb[1])
    return word_emb, word_dict


def predict_mse(model, dataloader, device,model_name='DeepCoNN'):
    mse, sample_count = 0, 0
    with torch.no_grad():
        for batch in dataloader:
            if model_name == 'DeepCoNN':
                user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
                predict = model(user_reviews, item_reviews)
                predict = torch.sigmoid(predict) * 4 + 1  # Da ~[0,5] -> [1,5]

            else:  # NARRE
                user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, ratings = \
                    map(lambda x: x.to(device), batch)
                datas = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
                predict = model(datas)

            # user_reviews, item_reviews, ratings = map(lambda x: x.to(device), batch)
            # predict = model(user_reviews, item_reviews)
            # scaled_predict = torch.sigmoid(predict) * 4 + 1  # Da ~[0,5] -> [1,5]


            mse += torch.nn.functional.mse_loss(predict, ratings, reduction='sum').item()
            sample_count += len(ratings)
    return mse / sample_count  # dataloader上的均方误差

from collections import defaultdict

class DeepCoNNDataset(Dataset):
    def __init__(self, data_path, word_dict, config, retain_rui=True,save_path = False):
        self.word_dict = word_dict
        self.config = config
        self.retain_rui = retain_rui  # 是否在最终样本中，保留user和item的公共review
        self.PAD_WORD_idx = self.word_dict[config.PAD_WORD]
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.lowest_r_count = config.lowest_review_count  # lowest amount of reviews wrote by exactly one user/item

        if os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            self.user_reviews = checkpoint["user_reviews"]
            self.item_reviews = checkpoint["item_reviews"]
            self.rating = checkpoint["rating"]
        else:
            df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])

            print('df shape: ',df.shape)
            df['review'] = df['review'].apply(self._review2id)  # 分词->数字
            print('reviews computed')
            self.sparse_idx = set()  # 暂存稀疏样本的下标，最后删除他们
            user_reviews = self._get_reviews(df)  # 收集每个user的评论列表
            item_reviews = self._get_reviews(df, 'itemID', 'userID')
            rating = torch.Tensor(df['rating'].to_list()).view(-1, 1)

            self.user_reviews = user_reviews[[idx for idx in range(user_reviews.shape[0]) if idx not in self.sparse_idx]]
            self.item_reviews = item_reviews[[idx for idx in range(item_reviews.shape[0]) if idx not in self.sparse_idx]]
            self.rating = rating[[idx for idx in range(rating.shape[0]) if idx not in self.sparse_idx]]

            torch.save({
                "user_reviews": self.user_reviews,
                "item_reviews": self.item_reviews,
                "rating": self.rating,
                "word_dict": self.word_dict,  # se vuoi salvare anche il vocabolario
                "config": self.config,  # se config è pickleable
            }, save_path)

    def __getitem__(self, idx):
        return self.user_reviews[idx], self.item_reviews[idx], self.rating[idx]

    def __len__(self):
        return self.rating.shape[0]

    def _get_reviews(self, df, lead='userID', costar='itemID'):
        # 对于每条训练数据，生成用户的所有评论汇总
        reviews_by_lead = dict(list(df[[costar, 'review']].groupby(df[lead])))  # 每个user/item评论汇总
        lead_reviews = []
        print('in get reviews')
        print(df[lead].shape)
        for idx, (lead_id, costar_id) in enumerate(zip(df[lead], df[costar])):
            if idx % 1000 == 0:
                print('idx cur', idx)
            df_data = reviews_by_lead[lead_id]  # 取出lead的所有评论：DataFrame
            if self.retain_rui:
                reviews = df_data['review'].to_list()  # 取lead所有评论：列表
            else:
                reviews = df_data['review'][df_data[costar] != costar_id].to_list()
            if len(reviews) < self.lowest_r_count:
                self.sparse_idx.add(idx)
            reviews = self._adjust_review_list(reviews, self.review_length, self.review_count)
            lead_reviews.append(reviews)
        return torch.LongTensor(lead_reviews)

    def _adjust_review_list(self, reviews, r_length, r_count):
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx] * r_length] * (r_count - len(reviews))
        reviews = [r[:r_length] + [0] * (r_length - len(r)) for r in reviews]
        return reviews

    def _review2id(self, review):
        if not isinstance(review, str):
            return []
        wids = []
        for word in review.split():
            if word in self.word_dict:
                wids.append(self.word_dict[word])
            else:
                wids.append(self.PAD_WORD_idx)
        return wids



import os
import torch
from torch.utils.data import Dataset
import pandas as pd
from collections import defaultdict

import os
import torch
from torch.utils.data import Dataset
import pandas as pd

class NARREDataset(Dataset):
    """
    Dataset per NARRE originale.
    Restituisce:
    user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, rating
    """
    def __init__(self, data_path, word_dict, config, save_path=None):
        self.word_dict = word_dict
        self.config = config
        self.PAD_WORD_idx = self.word_dict.get(config.PAD_WORD, 0)
        self.review_length = config.review_length
        self.review_count = config.review_count
        self.max_item_per_user = getattr(config, 'max_item_per_user', 50)
        self.max_user_per_item = getattr(config, 'max_user_per_item', 50)
        self.doc_dim = getattr(config, 'doc_dim', 10)

        if save_path and os.path.exists(save_path):
            checkpoint = torch.load(save_path)
            self.user_reviews = checkpoint["user_reviews"]
            self.item_reviews = checkpoint["item_reviews"]
            self.uids = checkpoint["uids"]
            self.iids = checkpoint["iids"]
            self.user_item2id = checkpoint["user_item2id"]
            self.item_user2id = checkpoint["item_user2id"]
            self.user_doc = checkpoint["user_doc"]
            self.item_doc = checkpoint["item_doc"]
            self.rating = checkpoint["rating"]
        else:
            df = pd.read_csv(data_path, header=None, names=['userID', 'itemID', 'review', 'rating'])
            df['review'] = df['review'].apply(self._review2id)

            # raccolta recensioni per user/item
            user_reviews_dict = self._build_reviews_dict(df, 'userID')
            item_reviews_dict = self._build_reviews_dict(df, 'itemID')

            self.user_reviews, self.item_reviews = [], []
            self.uids, self.iids = [], []
            self.user_item2id, self.item_user2id = [], []
            self.user_doc, self.item_doc = [], []
            self.rating = []
            c = 0
            user_to_items = df.groupby('userID')['itemID'].apply(list).to_dict()
            item_to_users = df.groupby('itemID')['userID'].apply(list).to_dict()
            # Funzioni vettorizzabili
            adjust_review = np.vectorize(lambda x: self._adjust_review_list(x))
            pad_list = lambda lsts, max_len: np.array([self._pad_list(l, max_len) for l in lsts])
            # Funzioni vettorizzabili
            adjust_review = np.vectorize(lambda x: self._adjust_review_list(x))
            pad_list = lambda lsts, max_len: np.array([self._pad_list(l, max_len) for l in lsts])
            uids = df['userID'].to_numpy()
            iids = df['itemID'].to_numpy()


            # --- 1️⃣ Pre-calcolare mapping user->items e item->users
            user_to_items = df.groupby('userID')['itemID'].apply(list).to_dict()
            item_to_users = df.groupby('itemID')['userID'].apply(list).to_dict()

            # --- 2️⃣ Estrarre colonne come array
            uids = df['userID'].to_numpy()
            iids = df['itemID'].to_numpy()
            ratings = df['rating'].to_numpy() if 'rating' in df.columns else None
            num_rows = len(df)

            # --- 3️⃣ Preparare reviews usando list comprehension (più veloce di iterrows)
            user_reviews_arr = [self._adjust_review_list(user_reviews_dict[u]) for u in uids]
            item_reviews_arr = [self._adjust_review_list(item_reviews_dict[i]) for i in iids]

            # --- 4️⃣ Pre-calcolare user->item e item->user padded lists
            user_item2id_arr = [self._pad_list(user_to_items[u], self.max_item_per_user) for u in uids]
            item_user2id_arr = [self._pad_list(item_to_users[i], self.max_user_per_item) for i in iids]

            # --- 5️⃣ Placeholder per documenti
            user_doc_arr = np.zeros((num_rows, self.doc_dim), dtype=np.float32)
            item_doc_arr = np.zeros((num_rows, self.doc_dim), dtype=np.float32)

            # --- 6️⃣ Assegnare direttamente alle proprietà dell'oggetto
            self.uids = uids.tolist()
            self.iids = iids.tolist()
            self.user_reviews = user_reviews_arr
            self.item_reviews = item_reviews_arr
            self.user_item2id = user_item2id_arr
            self.item_user2id = item_user2id_arr
            self.user_doc = user_doc_arr.tolist()
            self.item_doc = item_doc_arr.tolist()
            if ratings is not None:
                self.rating = ratings.tolist()

            # for idx, row in df.iterrows():
            #     c+=1
            #     if c % 10000 == 0:
            #         print(c)
            #     u, i = row['userID'], row['itemID']
            #
            #     self.user_reviews.append(self._adjust_review_list(user_reviews_dict[u]))
            #     self.item_reviews.append(self._adjust_review_list(item_reviews_dict[i]))
            #
            #     self.uids.append(u)
            #     self.iids.append(i)
            #
            #     # mapping user->item / item->user
            #     user_item_ids = list(df[df['userID'] == u]['itemID'])
            #     item_user_ids = list(df[df['itemID'] == i]['userID'])
            #     self.user_item2id.append(self._pad_list(user_item_ids, self.max_item_per_user))
            #     self.item_user2id.append(self._pad_list(item_user_ids, self.max_user_per_item))

                # doc placeholder
                # self.user_doc.append([0]*self.doc_dim)
                # self.item_doc.append([0]*self.doc_dim)
                #
                # self.rating.append(row['rating'])

            # converti tutto in tensori
            self.user_reviews = torch.LongTensor(self.user_reviews)
            self.item_reviews = torch.LongTensor(self.item_reviews)
            self.uids = torch.LongTensor(self.uids)
            self.iids = torch.LongTensor(self.iids)
            self.user_item2id = torch.LongTensor(self.user_item2id)
            self.item_user2id = torch.LongTensor(self.item_user2id)
            self.user_doc = torch.LongTensor(self.user_doc)
            self.item_doc = torch.LongTensor(self.item_doc)
            self.rating = torch.FloatTensor(self.rating).view(-1, 1)



            # salva i tensori su disco per riutilizzo
            if save_path:
                torch.save({
                    "user_reviews": self.user_reviews,
                    "item_reviews": self.item_reviews,
                    "uids": self.uids,
                    "iids": self.iids,
                    "user_item2id": self.user_item2id,
                    "item_user2id": self.item_user2id,
                    "user_doc": self.user_doc,
                    "item_doc": self.item_doc,
                    "rating": self.rating
                }, save_path)

    def __len__(self):
        return len(self.rating)

    def __getitem__(self, idx):
        return (
            self.user_reviews[idx],
            self.item_reviews[idx],
            self.uids[idx],
            self.iids[idx],
            self.user_item2id[idx],
            self.item_user2id[idx],
            self.user_doc[idx],
            self.item_doc[idx],
            self.rating[idx]
        )

    def _review2id(self, review):
        if not isinstance(review, str):
            return []
        return [self.word_dict.get(w, self.PAD_WORD_idx) for w in review.split()]

    def _build_reviews_dict(self, df, key='userID'):
        grouped = df.groupby(key)['review'].apply(list).to_dict()
        return grouped

    def _adjust_review_list(self, reviews):
        r_count = self.review_count
        r_len = self.review_length
        reviews = reviews[:r_count] + [[self.PAD_WORD_idx]*r_len]*(r_count - len(reviews))
        reviews = [r[:r_len] + [0]*(r_len - len(r)) for r in reviews]
        return reviews

    def _pad_list(self, lst, max_len):
        lst = lst[:max_len] + [0]*(max_len - len(lst))
        return lst
