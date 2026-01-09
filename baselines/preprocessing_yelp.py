import json
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch

# -----------------------------
# Config
# -----------------------------
INPUT_JSONL = "./data/yelp/yelp_academic_dataset_review.json"
OUTPUT_INTER = "./dataset/yelp/syelp.inter"
OUTPUT_INTER2 = "./dataset/yelp/yelp.inter"
OUTPUT_ITEM2 = "./dataset/yelp/yelp.item"
OUTPUT_USER2 = "./dataset/yelp/yelp.user"
output_file = "./dataset/yelp/merged.inter"
TEXT_FIELD = "text"
BATCH_SIZE = 1024
EMB_DIM = 384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# Model
# -----------------------------
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEVICE)
if DEVICE == "cuda":
    model = model.half()

# -----------------------------
# JSONL streaming
# -----------------------------
def jsonl_stream(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

# -----------------------------
# Main conversion
# -----------------------------
def convert_amazon_to_recbole(input_data, output_inter_file):

    rows = []

    batch_texts = []
    batch_meta = []

    for r in tqdm(jsonl_stream(input_data)):

        text = r.get(TEXT_FIELD, "")

        # salva metadata (1-a-1 con il testo)
        batch_meta.append(r)
        batch_texts.append(text if text and text.strip() else "")

        if len(batch_texts) == BATCH_SIZE:
            embs = model.encode(
                batch_texts,
                convert_to_numpy=True,
                normalize_embeddings=True
            )

            for meta, emb in zip(batch_meta, embs):
                rows.append({
                    "user_id:token": meta.get("user_id"),
                    "item_id:token": meta.get("business_id"),
                    "review_id:token": meta.get("review_id"),
                    "rating:float": meta.get("stars"),
                    "timestamp:token": meta.get("date"),
                    "useful:float": meta.get("useful", 0),
                    "funny:float": meta.get("funny", 0),
                    "cool:float": meta.get("cool", 0),
                    "inter_emb:float_seq": ", ".join(map(str, emb))
                })

            batch_texts.clear()
            batch_meta.clear()

    # ultimo batch
    if batch_texts:
        embs = model.encode(
            batch_texts,
            convert_to_numpy=True,
        )
        for meta, emb in zip(batch_meta, embs):
            rows.append({
                "user_id:token": meta.get("user_id"),
                "item_id:token": meta.get("business_id"),
                "review_id:token": meta.get("review_id"),
                "rating:float": meta.get("rating"),
                "timestamp:token": meta.get("date"),
                "useful:float": meta.get("useful", 0),
                "funny:float": meta.get("funny", 0),
                "cool:float": meta.get("cool", 0),
                "inter_emb:float_seq": ", ".join(map(str, emb))
            })

    df = pd.DataFrame(rows)
    df.to_csv(output_inter_file, sep="\t", index=False)

# -----------------------------
# Entry point
# -----------------------------

def merge_csvs():
    chunk_size = 100000  # dimensione chunk

    # Carica file1 come dizionario {review_id: stars} in RAM
    df1 = pd.read_csv(OUTPUT_INTER2, sep="\t",usecols=["review_id:token", "stars:float"])
    stars_dict = pd.Series(df1['stars:float'].values, index=df1['review_id:token']).to_dict()

    first_chunk = True
    for chunk in pd.read_csv(OUTPUT_INTER,sep="\t", chunksize=chunk_size):
        # Mappa i review_id della chunk al valore stars
        chunk["rating:float"] = chunk["review_id:token"].map(stars_dict)

        # Scrivi su file (append dopo la prima chunk)
        if first_chunk:
            chunk.to_csv(output_file, sep="\t",index=False, mode="w")
            first_chunk = False
        else:
            chunk.to_csv(output_file, sep="\t",index=False, mode="a", header=False)
    # df1 = pd.read_csv(OUTPUT_INTER)  # quello dove vuoi aggiungere la colonna
    # df2 = pd.read_csv(OUTPUT_INTER_1)  # quello che contiene "stars"
    #
    # # Assumiamo che i due CSV abbiano **stessa lunghezza e ordine**
    # df1["rating:float"] = df2["stars:float"]
    #
    # # Salva il nuovo CSV
    # df1.to_csv("merged.inter", index=False)

import pandas as pd

def remove_commas_pandas(input_file: str, output_file: str, float_seq_column: str = 'inter_emb:float_seq', sep='\t'):
    """
    Converte le colonne float_seq separate da virgola in colonne separate da spazio usando pandas.

    Args:
        input_file (str): Path del file di input.
        output_file (str): Path del file di output.
        float_seq_column (str): Nome della colonna contenente i float_seq.
        sep (str): Separatore delle colonne (default: tab '\t').
    """
    # Leggi il file
    df = pd.read_csv(input_file, sep=sep)

    # Sostituisci le virgole con spazi nella colonna float_seq
    df["inter_emb:token_seq"] = df["inter_emb:token_seq"].str.replace(', ', ' ')
    df.drop(columns=["inter_emb:float_seq"], inplace=True, errors='ignore')
    # Salva il file corretto
    df.to_csv(output_file, sep=sep, index=False)
    print(f"File convertito salvato in: {output_file}")

import pandas as pd

def clean_token_seq(input_file: str, output_file: str, token_seq_cols: list, sep='\t'):
    """
    Pulisce le colonne token_seq:
    - Sostituisce gli spazi con underscore "_" in ogni elemento
    - Rimuove le virgole
    - Mantiene gli elementi separati da spazio

    Args:
        input_file (str): Path del file di input
        output_file (str): Path del file di output
        token_seq_cols (list): Lista delle colonne da trattare come token_seq
        sep (str): Separatore delle colonne (default: '\t')
    """
    # Leggi il file
    df = pd.read_csv(input_file, sep=sep, dtype=str)

    for col in token_seq_cols:
        if col in df.columns:
            # Applica la trasformazione
            df[col] = df[col].apply(lambda x: ' '.join([token.replace(' ', '_') for token in str(x).split(',')]))

    # Salva il file corretto
    df.to_csv(output_file, sep=sep, index=False)
    print(f"File pulito salvato in: {output_file}")

import pandas as pd

def clean_token_seq(input_file: str, output_file: str, token_seq_cols: list, sep='\t'):
    """
    Pulisce le colonne token_seq:
    - Sostituisce gli spazi con underscore "_" in ogni elemento
    - Rimuove le virgole
    - Mantiene gli elementi separati da spazio

    Args:
        input_file (str): Path del file di input
        output_file (str): Path del file di output
        token_seq_cols (list): Lista delle colonne da trattare come token_seq
        sep (str): Separatore delle colonne (default: '\t')
    """
    # Leggi il file
    df = pd.read_csv(input_file, sep=sep, dtype=str)

    for col in token_seq_cols:
        if col in df.columns:
            # Applica la trasformazione
            df[col] = df[col].apply(lambda x: ' '.join([token.replace(' ', '_') for token in str(x).split(',')]))

    # Salva il file corretto
    df.to_csv(output_file, sep=sep, index=False)
    print(f"File pulito salvato in: {output_file}")


def reduce_embeddings_pca(
    input_file: str,
    output_file: str,
    emb_column: str = 'inter_emb:token_seq',
    original_dim: int = 384,
    reduced_dim: int = 128,
    sep: str = '\t',
    batch_size: int = 100_000
):
    """
    Riduce la dimensione degli embedding usando PCA globale.

    Args:
        input_file (str): file input
        output_file (str): file output
        emb_column (str): colonna con embedding (stringa)
        original_dim (int): dim originale (es. 384)
        reduced_dim (int): dim target (es. 128)
        sep (str): separatore
        batch_size (int): batch per evitare OOM
    """

    print("📥 Caricamento dati...")
    df = pd.read_csv(input_file, sep=sep, low_memory=False)
    print("🔄 Parsing embedding...")

    df = df.rename(columns={"inter_emb:token_seq": "inter_emb:float_seq"})
    if "timestamp:token" in df.columns:
        print("⏱️ Converting timestamp:token to timestamp:float ...")
        # converte stringhe tipo '2014-02-05 20:30:30' in float epoch
        df["timestamp:float"] = pd.to_datetime(df["timestamp:token"], errors='coerce').apply(
            lambda x: x.timestamp() if pd.notnull(x) else 0.0)
        # opzionale: rimuoviamo la colonna token originale se non serve
        df.drop(columns=["timestamp:token"], inplace=True)
    df.to_csv(output_file, sep=sep, index=False)
    print(f"💾 Embedding ridotti salvati in: {output_file}")

if __name__ == "__main__":
    #remove_commas_pandas(OUTPUT_INTER2, "./dataset/yelp/yelp_2.inter")
    #clean_token_seq(OUTPUT_ITEM2, "./dataset/yelp/yelp_2.item",["item_name:token_seq",	"address:token_seq",	"city:token_seq",	"categories:token_seq"])

    reduce_embeddings_pca(
        input_file=OUTPUT_INTER2,
        output_file=output_file,

    )
    #merge_csvs()
    # print("Processing Yelp reviews with embeddings...")
    # convert_amazon_to_recbole(
    #     input_data=INPUT_JSONL,
    #     output_inter_file=OUTPUT_INTER
    # )
    # print("Done.")
