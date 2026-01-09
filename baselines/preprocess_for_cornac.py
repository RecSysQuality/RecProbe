import pandas as pd

import json
import os
import pandas as pd
from collections import Counter

def replace_underscores_in_chunk(df, cols):
    for col in cols:
        if col in df.columns:
            df[col] = df[col].str.replace("_", " ", regex=False)
    return df

def build_review_lookup(jsonl_path):
    """
    Costruisce un dizionario review_id -> review_text dal JSONL.
    Se il file è troppo grande per la RAM, usare SQLite o Parquet.
    """
    review_lookup = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            rid = data["review_id"]
            text = data["text"]
            review_lookup[rid] = text
    return review_lookup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def process_items_yelp_full(items_path, output_path):
    import os
    import pandas as pd

    os.makedirs(output_path, exist_ok=True)
    items_out = os.path.join(output_path, "items_full.csv")
    print("Processing:", items_path)

    # Leggi tutto in RAM
    df = pd.read_csv(
        items_path,
        sep="\t",
        usecols=["item_id:token", "item_name:token_seq", "categories:token_seq"]
    )

    # Rinomina colonna
    df = df.rename(columns={"item_id:token": "item_id"})
    # Sostituisci NaN con stringa vuota
    df = df.fillna("")
    df = df.drop_duplicates()

    # Costruisci description = item_name + categories
    df["description"] = (
        df["item_name:token_seq"].str.strip() + " " +
        df["categories:token_seq"].str.strip()
    ).str.strip()

    # Sostituisci underscore con spazio
    df = replace_underscores_in_chunk(df, ["description"])

    # Se vuoi, rimuovi duplicati globali su item_id

    # Salva CSV finale
    df[["item_id", "description"]].to_csv(items_out, index=False)
    print("Done! Saved:", items_out)



def process_items_yelp(items_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(output_path)
    items_out = os.path.join(output_path, "items.csv")

    item_text_lookup = {}

    reader = pd.read_csv(
        items_path,
        sep="\t",
        chunksize=200_000,
        usecols=["item_id:token", "item_name:token_seq", "categories:token_seq"]
    )

    first = True
    for chunk in reader:
        chunk = chunk.rename(columns={"item_id:token": "item_id"})
        chunk = chunk.fillna("")

        # description = title + description + categories
        chunk["description"] = (
            chunk["item_name:token_seq"].str.strip() + " " +
            chunk["categories:token_seq"].str.strip()).str.strip()

        chunk = replace_underscores_in_chunk(chunk, ["description"])

        # Salva items.csv
        chunk[["item_id", "description"]].to_csv(
            items_out,
            mode="w" if first else "a",
            index=False,
            header=first
        )

def process_items_full(items_path, output_path):
    import os
    os.makedirs(output_path, exist_ok=True)
    print(output_path)

    items_out = os.path.join(output_path, "items_full.csv")

    # Leggi tutto in RAM
    df = pd.read_csv(
        items_path,
        sep="\t",
        usecols=["item_id:token", "title:token_seq", "description:token_seq", "categories:token_seq"]
    )
    df = df.drop_duplicates()
    # Rinomina colonna
    df = df.rename(columns={"item_id:token": "item_id"})

    # Sostituisci NaN con stringa vuota
    df = df.fillna("")

    # Costruisci description = title + description + categories
    df["description"] = (
        df["title:token_seq"].str.strip() + " " +
        df["description:token_seq"].str.strip() + " " +
        df["categories:token_seq"].str.strip()
    ).str.strip()

    # Sostituisci underscore con spazio
    df = replace_underscores_in_chunk(df, ["description"])

    # Salva CSV finale
    df[["item_id", "description"]].to_csv(items_out, index=False)
    print("Done! Saved:", items_out)



def process_items(items_path, output_path):
    os.makedirs(output_path, exist_ok=True)
    print(output_path)
    items_out = os.path.join(output_path, "items.csv")

    item_text_lookup = {}

    reader = pd.read_csv(
        items_path,
        sep="\t",
        chunksize=200_000,
        usecols=["item_id:token", "title:token_seq", "description:token_seq", "categories:token_seq"]
    )




    first = True
    for chunk in reader:
        chunk = chunk.rename(columns={"item_id:token": "item_id"})
        chunk = chunk.fillna("")

        # description = title + description + categories
        chunk["description"] = (
            chunk["title:token_seq"].str.strip() + " " +
            chunk["description:token_seq"].str.strip() + " " +
            chunk["categories:token_seq"].str.strip()
        ).str.strip()

        chunk = replace_underscores_in_chunk(chunk, ["description"])

        # Salva items.csv
        chunk[["item_id", "description"]].to_csv(
            items_out,
            mode="w" if first else "a",
            index=False,
            header=first
        )


def process_ratings_yelp_full(ratings_path, output_path,k):
    print(output_path)

    df = pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "rating:float",'review_id:token','timestamp:float']
    )

    items_out = os.path.join(output_path, "inter_full.csv")
    df = df.drop_duplicates()

    print("Counting user/item frequencies for k-core...")


    while True:
        user_cnt = Counter(df["user_id:token"])
        item_cnt = Counter(df["item_id:token"])

        # Step 2: set validi per k-core
        valid_users = {u for u, c in user_cnt.items() if c >= k}
        valid_items = {i for i, c in item_cnt.items() if c >= k}

        prev_shape = df.shape
        df = df[df["user_id:token"].isin(valid_users) & df["item_id:token"].isin(valid_items)]
        print("After k-core filtering:", df.shape)
        if df.shape == prev_shape:
            break



    # Costruisci lookup review_id -> review_text una volta sola
    jsonl_path = os.path.join(BASE_DIR, "dataset", "yelp", "yelp_academic_dataset_review.json")

    review_lookup = build_review_lookup(jsonl_path)
    df = df.rename(columns={
        "user_id:token": "user_id",
        "item_id:token": "item_id",
        "review_id:token": "review_id",
        "rating:float": "rating",
        "timestamp:float": "timestamp"
    })
    # Mappa review_text
    df["review_text"] = df["review_id"].map(review_lookup)

    # Rimuovi duplicati globali (mantieni l’ultima interazione per timestamp)
    df = df.sort_values("timestamp")

    # Applica k-core iterativo per utenti e item


    # Sostituisci underscore in review_text
    df = replace_underscores_in_chunk(df, ["review_text"])

    # Salva CSV finale
    df[["user_id", "item_id", "review_text", "rating", "timestamp"]].to_csv(
        items_out, index=False
    )
    print("Done! Saved:", items_out)

def process_ratings_yelp(ratings_path, output_path,k):
    print(output_path)

    reader = pd.read_csv(
        ratings_path,
        sep="\t",
        chunksize=2_000_000,
        usecols=["user_id:token", "item_id:token", "rating:float",'review_id:token','timestamp:float']
    )
    items_out = os.path.join(output_path, "inter.csv")

    print("Counting user/item frequencies for k-core...")
    user_cnt = Counter()
    item_cnt = Counter()
    reader_count = pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token"],
        chunksize=2_000_000
    )
    for chunk in reader_count:
        user_cnt.update(chunk["user_id:token"])
        item_cnt.update(chunk["item_id:token"])

    # Step 2: set validi per k-core
    valid_users = {u for u, c in user_cnt.items() if c >= k}
    valid_items = {i for i, c in item_cnt.items() if c >= k}
    print(f"Valid users: {len(valid_users)}, valid items: {len(valid_items)}")


    first = True
    for chunk in reader:
        chunk = chunk.rename(columns={
            "user_id:token": "user_id",
            "item_id:token": "item_id",
            "review_id:token": "review_id",
            "rating:float": "rating",
            "timestamp:float": "timestamp"
        })
        jsonl_path = os.path.join(BASE_DIR, "dataset", "yelp", "yelp_academic_dataset_review.json")
        review_lookup = build_review_lookup(jsonl_path)
        # description = title + description + categories


        chunk = chunk[chunk["user_id"].isin(valid_users) & chunk["item_id"].isin(valid_items)]
        if chunk.empty:
            continue

        chunk["review_text"] = chunk["review_id"].map(review_lookup)

        chunk = replace_underscores_in_chunk(chunk, ["review_text"])

        # Salva items.csv
        chunk[["user_id", "item_id", "review_text", "rating","timestamp"]].to_csv(
            items_out,
            mode="w" if first else "a",
            index=False,
            header=first
        )
        first = False

def process_ratings_full(ratings_path, output_path, k):
    print(output_path)
    items_out = os.path.join(output_path, "inter_full.csv")

    # Leggi tutto il dataset in RAM
    df = pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token", "rating:float",
                 "title:token_seq", "text:token_seq", "timestamp:float"]
    )

    # Rinomina colonne
    df = df.rename(columns={
        "user_id:token": "user_id",
        "item_id:token": "item_id",
        "rating:float": "rating",
        "timestamp:float": "timestamp"
    })

    # Costruisci review_text
    df["review_text"] = (df["title:token_seq"].str.strip() + " " +
                         df["text:token_seq"].str.strip()).str.strip()

    # Elimina duplicati globali (mantieni l’ultima interazione per timestamp)
    df = df.sort_values("timestamp")
    print(df.shape)
    df = df.drop_duplicates(subset=["user_id", "item_id","rating","timestamp"], keep="last")
    print(df.shape)
    # Step 1: conta frequenze user/item per k-core
    while True:
        user_cnt = Counter(df["user_id"])
        item_cnt = Counter(df["item_id"])

        # Step 2: set validi per k-core
        valid_users = {u for u, c in user_cnt.items() if c >= k}
        valid_items = {i for i, c in item_cnt.items() if c >= k}

        prev_shape = df.shape
        df = df[df["user_id"].isin(valid_users) & df["item_id"].isin(valid_items)]
        print("After k-core filtering:", df.shape)
        if df.shape == prev_shape:
            break

    df = replace_underscores_in_chunk(df, ["review_text"])

    # Salva il CSV finale
    df[["user_id", "item_id", "review_text", "rating", "timestamp"]].to_csv(
        items_out, index=False
    )

    print("Done! Saved:", items_out)

def process_ratings(ratings_path, output_path,k):
    print(output_path)

    reader = pd.read_csv(
        ratings_path,
        sep="\t",
        chunksize=2_000_000,
        usecols=["user_id:token", "item_id:token", "rating:float","title:token_seq","text:token_seq",'timestamp:float']
    )
    items_out = os.path.join(output_path, "inter.csv")




    # Step 1: conta frequenze user/item
    print("Counting user/item frequencies for k-core...")
    user_cnt = Counter()
    item_cnt = Counter()
    reader_count = pd.read_csv(
        ratings_path,
        sep="\t",
        usecols=["user_id:token", "item_id:token"],
        chunksize=2_000_000
    )
    for chunk in reader_count:
        user_cnt.update(chunk["user_id:token"])
        item_cnt.update(chunk["item_id:token"])

    # Step 2: set validi per k-core
    valid_users = {u for u, c in user_cnt.items() if c >= k}
    valid_items = {i for i, c in item_cnt.items() if c >= k}
    print(f"Valid users: {len(valid_users)}, valid items: {len(valid_items)}")



    first = True
    for chunk in reader:
        chunk = chunk.rename(columns={
            "user_id:token": "user_id",
            "item_id:token": "item_id",
            "rating:float": "rating",
            "timestamp:float": "timestamp"
        })

        # description = title + description + categories
        chunk["review_text"] = (
            chunk["title:token_seq"].str.strip() + " " +
            chunk["text:token_seq"].str.strip()
        ).str.strip()

        chunk = chunk[chunk["user_id"].isin(valid_users) & chunk["item_id"].isin(valid_items)]
        if chunk.empty:
            continue

        chunk = replace_underscores_in_chunk(chunk, ["review_text"])

        # Salva items.csv
        chunk[["user_id", "item_id", "review_text", "rating","timestamp"]].to_csv(
            items_out,
            mode="w" if first else "a",
            index=False,
            header=first
        )
        first = False




def process_data_large(dataset_dir, output_path,dataset,k):
    print(dataset_dir, output_path, dataset)
    items_path = os.path.join(dataset_dir, f"{dataset}.item")
    ratings_path = os.path.join(dataset_dir, f"{dataset}.inter")
    if dataset != 'yelp':
        process_items_full(items_path, output_path)
        process_ratings_full(ratings_path, output_path,k)
    else:
        process_items_yelp_full(items_path, output_path)
        process_ratings_yelp_full(ratings_path, output_path,k)


def dataset_stats(inter_csv_path):
    # Leggi interazioni
    df = pd.read_csv(inter_csv_path)

    # Numero unico di utenti e item
    n_users = df['user_id'].nunique()
    n_items = df['item_id'].nunique()
    n_interactions = len(df)

    # Media items per utente
    avg_items_per_user = n_interactions / n_users

    # Media utenti per item
    avg_users_per_item = n_interactions / n_items

    # Sparsity = 1 - (interazioni osservate / interazioni possibili)
    sparsity = 1 - (n_interactions / (n_users * n_items))

    print(f"Numero utenti: {n_users}")
    print(f"Numero item: {n_items}")
    print(f"Numero interazioni: {n_interactions}")
    print(f"Avg items per user: {avg_items_per_user:.2f}")
    print(f"Avg users per item: {avg_users_per_item:.2f}")
    print(f"Sparsity: {sparsity:.6f}")

import argparse
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        nargs='?',  # <--- anche questo opzionale
        default='amazon_beauty',
        help='Dataset'
    )
    parser.add_argument(
        'k',
        nargs='?',  # <--- anche questo opzionale
        default='5',
        help='Dataset'
    )
    args = parser.parse_args()
    dataset = args.dataset
    k = int(args.k)
    DATASET_DIR = os.path.join(BASE_DIR, "dataset", dataset)
    OUTPUT_DIR = os.path.join(BASE_DIR, "cornac_data", dataset)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_data_large(
        dataset_dir=DATASET_DIR,
        output_path=OUTPUT_DIR,dataset=dataset,k=k
    )





if __name__ == "__main__":
    #main()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset',
        nargs='?',  # <--- anche questo opzionale
        default='amazon_beauty',
        help='Dataset'
    )
    parser.add_argument(
        'k',
        nargs='?',  # <--- anche questo opzionale
        default='5',
        help='Dataset'
    )
    args = parser.parse_args()
    dataset = args.dataset
    for dataset in ['amazon_beauty','amazon_baby','yelp']:
        print(dataset)
        dataset_stats(f'./cornac_data/{dataset}/inter_full.csv')