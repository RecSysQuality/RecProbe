#
#
# import json
# import pandas as pd
# from pathlib import Path
# from sentence_transformers import SentenceTransformer
# import torch
#
# # --- Inizializzare il modello ---
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print("Using device:", device)
# model = SentenceTransformer('all-MiniLM-L6-v2',device=device)
#
# def parse_details(js):
#     d = js
#     tokens = []
#     for k, v in d.items():
#         if isinstance(v, str):
#             tokens.append(v.replace(" ", "_"))
#         elif isinstance(v, list):
#             tokens.extend([str(x).replace(" ", "_") for x in v])
#         # puoi aggiungere float_seq se necessario
#     return " ".join(tokens)
#
# def list_to_token_seq(s):
#     lst = ast.literal_eval(s)
#     tokens = [t.replace(" ", "_") for t in lst]
#     return " ".join(tokens)
#
# def get_embeddings_batch(texts, batch_size=64):
#     """Restituisce embeddings normalizzati per una lista di testi usando batch processing."""
#     return model.encode([str(t) for t in texts], show_progress_bar=True, normalize_embeddings=True)
#
# def load_jsonl(file_path):
#     """Carica un file JSONL in una lista di dizionari."""
#     with open(file_path, 'r') as f:
#         return [json.loads(line) for line in f]
#
# def create_interactions_df(reviews, batch_size=64):
#     """Crea DataFrame delle interazioni con embeddings in batch."""
#     texts = [f"{r.get('title', '')} {r.get('text', '')}" for r in reviews]
#     embeddings = get_embeddings_batch(texts)
#
#     inter_data = []
#     for r, emb in zip(reviews, embeddings):
#         inter_data.append({
#             'user_id': r.get('user_id'),
#             'item_id': r.get('asin'),
#             'rating': r.get('rating'),
#             'timestamp': r.get('timestamp'),
#             #'images': r.get('images', []),
#             #'title': r.get('title',''),
#             #'text': r.get('text',''),
#             'verified_purchase': float(r.get('verified_purchase', False)),
#             'helpful_vote': r.get('helpful_vote', 0),
#             'inter_emb': ' '.join(map(str, emb))
#         })
#
#     df = pd.DataFrame(inter_data)
#     df.columns = [
#         'user_id:token', 'item_id:token', 'rating:float', 'timestamp:float',
#           'verified_purchase:float',
#         'helpful_vote:float', 'inter_emb:float_seq'
#     ]
#     return df
#
# def create_items_df(metadata, main_category='All Beauty', batch_size=64):
#     """Crea DataFrame dei prodotti con embeddings in batch."""
#     texts = [f"{item.get('title','')} {item.get('description','')}" for item in metadata]
#     embeddings = get_embeddings_batch(texts, batch_size=batch_size)
#
#     item_data = []
#     for item, emb in zip(metadata, embeddings):
#         item_data.append({
#             'item_id': item.get('parent_asin',''),
#             #'title': item.get('title',''),
#             #'description': item.get('description', ''),
#             'average_rating': item.get('average_rating',None),
#             'rating_number': item.get('rating_number',None),
#             'features': " ".join([c.replace(" ", "_") for c in item.get('features', [])]),
#             'price': item.get('price',0.0),
#             #'images': item.get('images',''),
#             #'videos': item.get('videos',''),
#             'store': item.get('store',''),
#             'categories': " ".join([c.replace(" ", "_") for c in item.get('categories', [])]),
#             'details': parse_details(item.get('details',{})),
#             'bought_together': item.get('bought_together',[]),
#             'main_category': item.get('main_category', main_category),
#             'item_emb': ' '.join(map(str, emb))
#         })
#
#     df = pd.DataFrame(item_data).drop_duplicates(subset=['item_id'])
#     df.columns = [
#         'item_id:token',  'average_rating:float', 'rating_number:float',
#         'features:token_seq',  'price:float',
#          'store:token', 'categories:token_seq', 'details:token_seq',
#         'bought_together:token_seq', 'main_category:token_seq', 'item_emb:float_seq'
#     ]
#     return df
#
# def convert_amazon_to_recbole(input_data, input_metadata, output_inter_file, output_item_file, main_category='All Beauty', batch_size=64):
#     reviews = load_jsonl(input_data)
#     metadata = load_jsonl(input_metadata)
#
#     inter_df = create_interactions_df(reviews, batch_size=batch_size)
#     item_df = create_items_df(metadata, main_category=main_category, batch_size=batch_size)
#
#     inter_df.to_csv(output_inter_file, sep='\t', index=False)
#     item_df.to_csv(output_item_file, sep='\t', index=False)
#
# if __name__ == "__main__":
#     datasets2 = [
#         ('amazon_beauty','All_Beauty.jsonl', 'meta_All_Beauty.jsonl', 'amazon_beauty.inter', 'amazon_beauty.item', 'All Beauty'),
#         ('amazon_baby','Baby_Products.jsonl', 'meta_Baby_Products.jsonl', 'amazon_baby.inter', 'amazon_baby.item', 'Baby Products'),
#         ('amazon_fashion','Amazon_Fashion.jsonl', 'meta_Amazon_Fashion.jsonl', 'amazon_fashion.inter', 'amazon_fashion.item', 'Amazon Fashion'),
#     ]
#     datasets = [('amazon_toys','Toys_and_Games.jsonl', 'meta_Toys_and_Games.jsonl', 'amazon_toys.inter', 'amazon_toys.item', 'Toys_and_Games'),
#         ]
#     print('eccoci qua updated 3')
#     for folder,data_file, meta_file, inter_file, item_file, category in datasets:
#         print(f"Processing {category}...")
#         convert_amazon_to_recbole(
#             input_data=f'./data/{data_file}',
#             input_metadata=f'./data/{meta_file}',
#             output_inter_file=f'./dataset/{folder}/{inter_file}',
#             output_item_file=f'./dataset/{folder}/{item_file}',
#             main_category=category,
#             batch_size=128  # puoi aumentare se hai abbastanza VRAM
#         )
import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# -----------------------------
# Config e modello
# -----------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
if device == "cuda":
    model = model.half()  # mixed precision per risparmiare VRAM

EMB_DIM = 384

# -----------------------------
# Utility
# -----------------------------
def jsonl_stream(path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def parse_details(js):
    if not js:
        return "NONE"
    tokens = []
    for v in js.values():
        if isinstance(v, str):
            tokens.append(v.replace(" ", "_"))
        elif isinstance(v, list):
            tokens.extend(str(x).replace(" ", "_") for x in v)
    return " ".join(tokens)

# -----------------------------
# INTERACTIONS
# -----------------------------
def create_interactions_df_stream(jsonl_path, output_path, batch_size=128):
    rows = []
    batch_texts = []
    batch_meta = []

    for r in tqdm(jsonl_stream(jsonl_path)):
        text = f"{r.get('title','')} {r.get('text','')}".strip()
        batch_texts.append(text)
        batch_meta.append(r)

        if len(batch_texts) == batch_size:
            embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            for meta, emb in zip(batch_meta, embs):
                rows.append({
                    "user_id:token": meta.get("user_id"),
                    "item_id:token": meta.get("asin"),
                    "rating:float": meta.get("rating"),
                    "timestamp:float": meta.get("timestamp"),
                    "verified_purchase:float": float(meta.get("verified_purchase", False)),
                    "helpful_vote:float": meta.get("helpful_vote", 0),
                    "inter_emb:float_seq": " ".join(map(str, emb))
                })
            batch_texts.clear()
            batch_meta.clear()

    # ultimo batch
    if batch_texts:
        embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        for meta, emb in zip(batch_meta, embs):
            rows.append({
                "user_id:token": meta.get("user_id"),
                "item_id:token": meta.get("asin"),
                "rating:float": meta.get("rating"),
                "timestamp:float": meta.get("timestamp"),
                "verified_purchase:float": float(meta.get("verified_purchase")) or 0.0,
                "helpful_vote:float": meta.get("helpful_vote", 0),
                "inter_emb:float_seq": " ".join(map(str, emb))
            })

    pd.DataFrame(rows).to_csv(output_path, sep="\t", index=False)
    print(f"Saved interactions to {output_path}")

# -----------------------------
# ITEMS
# -----------------------------
def create_items_df_stream(jsonl_path, output_path, main_category, batch_size=128):
    rows = []
    batch_texts = []
    batch_meta = []

    for item in tqdm(jsonl_stream(jsonl_path)):
        text = f"{item.get('title','')} {item.get('description','')}".strip()
        batch_texts.append(text)
        batch_meta.append(item)

        if len(batch_texts) == batch_size:
            embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
            for meta, emb in zip(batch_meta, embs):
                rows.append({
                    "item_id:token": meta.get("parent_asin"),
                    "average_rating:float": meta.get("average_rating"),
                    "rating_number:float": meta.get("rating_number"),
                    "features:token_seq": " ".join(c.replace(" ", "_") for c in meta.get("features")) or "<NONE>",
                    "price:float": meta.get("price") or 0.0,
                    "store:token": meta.get("store") or "None",
                    "categories:token_seq": " ".join(c.replace(" ", "_") for c in meta.get("categories")) or "<NONE>",
                    "details:token_seq": parse_details(meta.get("details")),
                    #"bought_together:token_seq": " ".join(meta.get("bought_together", ["<NONE>"])),
                    "main_category:token_seq": main_category,
                    "item_emb:float_seq": " ".join(map(str, emb))
                })
            batch_texts.clear()
            batch_meta.clear()

    if batch_texts:
        embs = model.encode(batch_texts, convert_to_numpy=True, normalize_embeddings=True)
        for meta, emb in zip(batch_meta, embs):
            rows.append({
                "item_id:token": meta.get("parent_asin"),
                "average_rating:float": meta.get("average_rating"),
                "rating_number:float": meta.get("rating_number"),
                "features:token_seq": " ".join(c.replace(" ", "_") for c in meta.get("features", ["<NONE>"])),
                "price:float": meta.get("price", 0.0),
                "store:token": meta.get("store", "NONE"),
                "categories:token_seq": " ".join(c.replace(" ", "_") for c in meta.get("categories", ["<NONE>"])),
                "details:token_seq": parse_details(meta.get("details", "NONE")),
                #"bought_together:token_seq": " ".join(meta.get("bought_together", ["<NONE>"])),
                "main_category:token_seq": main_category,
                "item_emb:float_seq": " ".join(map(str, emb))
            })

    pd.DataFrame(rows).drop_duplicates("item_id:token").to_csv(output_path, sep="\t", index=False)
    print(f"Saved items to {output_path}")



def fillna_with_defaults(path_in,path_out,type='item'):
    df = pd.read_csv(path_in, sep="\t")  # cambia sep="\t" se necessario
    if type == 'item':
        float_cols = ["average_rating:float", "rating_number:float", "price:float"]
        token_cols = [
            "item_id:token", "title:token_seq", "description:token_seq",
            "features:token_seq", "store:token", "categories:token_seq",
            "details:token_seq", "bought_together:token_seq",
            "main_category:token_seq", "item_emb:float_seq"
        ]
    else:

        float_cols = ["helpful_vote:float", "inter_emb:float_seq"]
        token_cols = [
            "title:token_seq", "text:token_seq",
            "verified_purchase:token"
        ]
    # Riempie valori mancanti nelle colonne float con 0.0
    df[float_cols] = df[float_cols].fillna(0.0)

    # Riempie valori mancanti o stringhe vuote nelle colonne token/token_seq con "NONE"
    df[token_cols] = df[token_cols].fillna("NONE").replace("", "NONE").replace("[]","NONE").replace([],"NONE")
    # Salva il CSV aggiornato
    
    df.to_csv(path_out, sep="\t", index=False)
# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":

    datasets = [
        ('amazon_beauty','All_Beauty.jsonl', 'meta_All_Beauty.jsonl', 'amazon_beauty.inter', 'amazon_beauty.item', 'All Beauty'),
        ('amazon_baby','Baby_Products.jsonl', 'meta_Baby_Products.jsonl', 'amazon_baby.inter', 'amazon_baby.item', 'Baby Products')
    ]

    for folder, data_file, meta_file, inter_file, item_file, category in datasets:
        print(f"Processing {category}...")
        fillna_with_defaults(f"./dataset/{folder}/{item_file}",f"./dataset/{folder}/2_{item_file}")
        fillna_with_defaults(f"./dataset/{folder}/{inter_file}",f"./dataset/{folder}/2_{inter_file}", type='inter')
        #fillna_with_defaults(f"./dataset/{folder}/{inter_file}",f"./dataset/{folder}/2_{inter_file}")
        # create_items_df_stream(
        #     f"./dataset/{folder}/{item_file}",
        #     f"./dataset/{folder}/{item_file}",
        #     main_category=category,
        #     batch_size=1024
        # )
        #
        # create_interactions_df_stream(
        #     f"./data/{data_file}",
        #     f"./dataset/{folder}/{inter_file}",
        #     batch_size=1024
        # )


