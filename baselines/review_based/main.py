import os
import time

from pyarrow.dataset import dataset
from tqdm import tqdm
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
import numpy as np
import json
from config import Config
from model import DeepCoNN
from narre import NARRE
from utils import load_embedding, DeepCoNNDataset, predict_mse, date,NARREDataset,load_optimal_embedding,load_top_embeddings
import random
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # Forza determinismo su GPU
    torch.backends.cudnn.benchmark = False

set_seed(42)

def train(train_dataloader, valid_dataloader, model, config, model_path):
    print(f'{date()}## Start the training!')
    #train_mse = predict_mse(model, train_dataloader, config.device)
    #valid_mse = predict_mse(model, valid_dataloader, config.device)
    #print(f'{date()}#### Initial train mse {train_mse:.6f}, validation mse {valid_mse:.6f}')
    start_time = time.perf_counter()

    opt = torch.optim.Adam(model.parameters(), config.learning_rate, weight_decay=config.l2_regularization)
    lr_sch = torch.optim.lr_scheduler.ExponentialLR(opt, config.learning_rate_decay)
    count_epochs = 0
    best_loss, best_epoch = 100, 0
    for epoch in tqdm(range(config.train_epochs), desc="Epochs", leave=False):
        model.train()  # 将模型设置为训练状态
        total_loss, total_samples = 0, 0
        pbar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        for batch in pbar:
            if config.model == 'DeepCoNN':
                user_reviews, item_reviews, ratings = map(lambda x: x.to(config.device), batch)
                predict = model(user_reviews, item_reviews)
            else:
                user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc, ratings = \
                    [x.to(config.device) for x in batch]

                datas = [user_reviews, item_reviews, uids, iids, user_item2id, item_user2id, user_doc, item_doc]
                predict = model(datas)

            loss = F.mse_loss(predict, ratings, reduction='sum')  # 平方和误差
            opt.zero_grad()  # 梯度清零
            loss.backward()  # 反向传播计算梯度
            opt.step()  # 根据梯度信息更新所有可训练参数

            total_loss += loss.item()
            total_samples += len(predict)

        lr_sch.step()
        model.eval()  # 停止训练状态
        valid_mse = predict_mse(model, valid_dataloader, config.device,config.model)
        train_loss = total_loss / total_samples
        print(f"{date()}#### Epoch {epoch:3d}; train mse {train_loss:.6f}; validation mse {valid_mse:.6f}")

        if valid_mse < best_loss:
            best_loss = valid_mse
            print('saving')
            torch.save(model, model_path)
            count_epochs = 0  # resetta il contatore se miglioriamo
        elif valid_mse < best_loss + 1e-2:
            count_epochs = 0
        else:
            count_epochs += 1  # incremento solo se non miglioriamo

        if count_epochs >= config.early_stop:
            break
    end_time = time.perf_counter()
    print(f'{date()}## End of training! Time used {end_time - start_time:.0f} seconds.')


def test(dataloader, model,save_path,config):
    print(f'{date()}## Start the testing!')

    def config_to_dict(config):
        """
        Converte la config in un dizionario serializzabile per JSON
        """
        d = {}
        for k, v in vars(config).items():
            # Se non serializzabile, converte in stringa
            try:
                json.dumps({k: v})
                d[k] = v
            except TypeError:
                d[k] = str(v)
        return d

    start_time = time.perf_counter()
    test_loss = predict_mse(model, dataloader, config.device,config.model)
    end_time = time.perf_counter()
    print(f"{date()}## Test end, test mse is {test_loss:.6f}, time used {end_time - start_time:.0f} seconds.")
    # Prepara i dati da salvare
    results = {
        "dataset": getattr(config, "dataset", "unknown"),
        "test_mse": test_loss,
        "config": config_to_dict(config)  # salva tutte le variabili della config
    }

    # Salva in JSON
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"{date()}## Results and config saved to {save_path}")
    return results

import numpy as np
import torch


def create_ranking_dataloader(test_df, train_val_seen, all_items, num_neg=100, batch_size=256):
    """Crea dataloader con positivi + negatives"""
    candidates_data = []

    for user in test_df['userID'].unique():
        test_pos = test_df[test_df['userID'] == user]
        seen = train_val_seen.get(user, set())
        unseen = set(all_items) - seen
        neg_pool = unseen - set(test_pos['itemID'])

        # Positivi (label=1 se >=4)
        for _, row in test_pos.iterrows():
            if row['rating'] >= 4:
                candidates_data.append((user, row['itemID'], 1.0))  # binary label

        # Negatives (label=0)
        neg_items = np.random.choice(list(neg_pool), min(num_neg, len(neg_pool)))
        for item in neg_items:
            candidates_data.append((user, item, 0.0))

    # Crea tensors urs, irs, labels (adatta al tuo preprocess embedding)
    # Assumi hai funzioni get_user_reviews(user), get_item_reviews(item)
    urs, irs, labels = [], [], []
    for user, item, label in candidates_data:
        urs.append(get_user_reviews(user))  # torch.tensor
        irs.append(get_item_reviews(item))
        labels.append(label)

    dataset = TensorDataset(torch.stack(urs), torch.stack(irs), torch.tensor(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def recall_at_k(scores, labels, k):
    top_k_idx = np.argsort(scores)[-k:][::-1]  # decrescente
    hits = np.sum(labels[top_k_idx])
    return hits / np.sum(labels) if np.sum(labels) > 0 else 0

def ndcg_at_k(scores, labels, k):
    top_k_idx = np.argsort(scores)[-k:][::-1]
    dcg = np.sum(labels[top_k_idx] / np.log2(np.arange(2, k+2)))
    idcg = np.sum(np.sort(labels[-k:])[::-1] / np.log2(np.arange(2, k+2)))
    return dcg / idcg if idcg > 0 else 0


def evaluate_ranking(dataloader, model, device):
    model.eval()
    k_list = [10, 50]
    recalls_per_k = {k: [] for k in k_list}
    ndcgs_per_k = {k: [] for k in k_list}

    with torch.no_grad():
        for batch in dataloader:
            urs, irs, labels_batch = [x.to(device) for x in batch]  # labels invece di rats

            scores = model(urs, irs).cpu().numpy()
            labels = labels_batch.cpu().numpy()

            # Per OGNI utente nel batch (assumi 1 utente per riga)
            for u_scores, u_labels in zip(scores, labels):
                for k in k_list:
                    recalls_per_k[k].append(recall_at_k(u_scores, u_labels, k))
                    ndcgs_per_k[k].append(ndcg_at_k(u_scores, u_labels, k))

    results = {f"Recall@{k}": np.mean(recalls_per_k[k]) for k in k_list}
    results.update({f"NDCG@{k}": np.mean(ndcgs_per_k[k]) for k in k_list})
    return results


# Usage:
from collections import defaultdict
import pandas as pd

if __name__ == '__main__':
    config = Config()
    print(config)
    print(f'{date()}## Load embedding and data...')

    if config.model == 'DeepCoNN':
        word_emb, word_dict = load_embedding(config.word2vec_file)

        print('loading training...')
        train_dataset = DeepCoNNDataset(config.train_file, word_dict, config,save_path=config.path_inter_train)
        print('loading validation...')
        valid_dataset = DeepCoNNDataset(config.valid_file, word_dict, config, retain_rui=False,save_path=config.path_inter_vali)
        print('loading test...')
        test_dataset = DeepCoNNDataset(config.test_file, word_dict, config, retain_rui=False,save_path=config.path_inter_test)
        print('loaded all')
    else:
        if config.vocab_size < 1291148:  # Vocabolario ridotto
            all_data_paths = [
                config.train_file,  # train.csv
                config.valid_file,  # val.csv
                config.test_file  # test.csv
            ]


            print("🚀 Modalità VOCAB OTTIMALE attivata!")
            word_emb, word_dict = load_top_embeddings(
                all_data_paths,  # ← Train+Val+Test!
                config.word2vec_file,
                config.vocab_size  # 5000
            )
            print(len(word_emb))
            config.vocab_size = len(word_emb)
            # dimensione di ogni embedding (dovrebbe essere costante)
            print(len(word_emb[0]))
            print(len(word_emb[1]))
 # Auto-update!
            print(f"✅ Vocabolario: {config.vocab_size}")
        else:
            print("📚 Vocabolario globale completo")
            word_emb, word_dict = load_embedding(config.word2vec_file)

        print('loading training...')
        train_dataset = NARREDataset(config.train_file, word_dict, config,save_path=config.path_inter_train_narre)
        print('loading validation...')
        valid_dataset = NARREDataset(config.valid_file, word_dict, config,save_path=config.path_inter_vali_narre)
        print('loading test...')
        test_dataset = NARREDataset(config.test_file, word_dict, config,save_path=config.path_inter_test_narre)
        print('loaded all')

    train_dlr = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    valid_dlr = DataLoader(valid_dataset, batch_size=config.batch_size)
    test_dlr = DataLoader(test_dataset, batch_size=config.batch_size)
    if config.model == 'DeepCoNN':
        model = DeepCoNN(config, word_emb).to(config.device)
    else:
        # Unisci tutti gli ID per calcolare numero totale utenti/item
        all_uids = torch.cat([train_dataset.uids, valid_dataset.uids, test_dataset.uids])
        all_iids = torch.cat([train_dataset.iids, valid_dataset.iids, test_dataset.iids])

        config.user_num = len(torch.unique(all_uids))
        config.item_num = len(torch.unique(all_iids))

        print(f"Utenti totali: {config.user_num}, Item totali: {config.item_num}")
        word_emb = torch.tensor(word_emb, dtype=torch.float)
        model = NARRE(config, word_emb).to(config.device)

    del train_dataset, valid_dataset, test_dataset, word_emb, word_dict

    os.makedirs(os.path.dirname(config.model_file), exist_ok=True)  # 文件夹不存在则创建
    train(train_dlr, valid_dlr, model, config, config.model_file)
    test(test_dlr, torch.load(config.model_file),config.save_path,config)

    # # 1. all_items: TUTTI gli item unici nei tuoi dati
    # val_df = pd.read_csv(config.valid_file)
    # train_df = pd.read_csv(config.train_file)
    # test_df = pd.read_csv(config.test_file)
    #
    # all_items = set(pd.concat([train_df,val_df,test_df])['itemID'].unique())
    #
    # # 2. user_seen: item visti in train+val per utente
    # user_seen = defaultdict(set)
    # for df in [train_df, val_df]:
    #     for _, row in df.iterrows():
    #         user_seen[row['userID']].add(row['itemID'])
    #
    # user_seen = dict(user_seen)  # {user_id: {item1, item2, ...}}
    #
    # ranking_dl = create_ranking_dataloader(test_df, user_seen, all_items)
    # results = evaluate_ranking(ranking_dl, model, device)
    # #print("Recall@10: {:.4f}, Recall@50: {:.4f}".format(metrics["Recall@10"], metrics["Recall@50"]))
    # #print("NDCG@10: {:.4f}, NDCG@50: {:.4f}".format(metrics["NDCG@10"], metrics["NDCG@50"]))
    #
    # #metrics = evaluate_ranking(test_dlr, model, config.device)
    # #print("Recall@10: {:.4f}, Recall@50: {:.4f}".format(metrics["Recall@10"], metrics["Recall@50"]))
    # #print("NDCG@10: {:.4f}, NDCG@50: {:.4f}".format(metrics["NDCG@10"], metrics["NDCG@50"]))