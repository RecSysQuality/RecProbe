#!/usr/bin/env python3
"""
Converte un file JSONL in CSV estraendo: user_id, item_id, rating, text, timestamp
Funziona in streaming, adatto per file di grandi dimensioni (es. 20GB).

Uso:
    python jsonl_to_csv.py input.jsonl output.csv

    # Con barra di avanzamento (richiede tqdm):
    pip install tqdm
    python jsonl_to_csv.py input.jsonl output.csv --progress
"""
import datetime

RECBOLE_FIELDS = ["user_id:token", "item_id:token", "rating:float", "text:token_seq", "timestamp:float"]
RECBOLE_COLS = ["user_id", "item_id", "rating", "text", "timestamp"]

import json
import csv
import re
import sys
from pathlib import Path
import pandas as pd

FIELDS = ["user_id","item_id","rating","text","timestamp"]

# Rimuove tutti i caratteri di controllo (0x00-0x1f e 0x7f-0x9f), sostituendoli con spazio
_CONTROL_RE = re.compile(r'[\x00-\x1f\x7f-\x9f]')


def sanitize(value) -> str:
    if not isinstance(value, str):
        return value
    return _CONTROL_RE.sub(' ', value).strip()


def convert_yelp_jsonl_to_csv(input_path: str, output_path: str, show_progress: bool = False):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Errore: file non trovato → {input_path}")
        sys.exit(1)

    progress = None
    if show_progress:
        try:
            from tqdm import tqdm
            file_size = input_file.stat().st_size
            progress = tqdm(total=file_size, unit="B", unit_scale=True, desc="Elaborazione")
        except ImportError:
            print("tqdm non installato, procedo senza barra di avanzamento.")

    rows_written = 0
    rows_skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8", newline="") as fout:

        writer = csv.DictWriter(fout, fieldnames=FIELDS, extrasaction="ignore",
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for line_num, line in enumerate(fin, start=1):
            if progress:
                progress.update(len(line.encode("utf-8")))

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Riga {line_num}: JSON non valido ({e}) — saltata", file=sys.stderr)
                rows_skipped += 1
                continue

            row = {
                "user_id": obj['user_id'],
                "item_id": obj['business_id'],
                "rating": obj['stars'],
                "text": obj['text'],
                "timestamp": obj.get("date", ""),
            }

            writer.writerow(row)
            rows_written += 1

            if not progress and rows_written % 500_000 == 0:
                print(f"  → {rows_written:,} righe scritte...")

    if progress:
        progress.close()

    print(f"\n✅ Completato!")
    print(f"   Righe scritte : {rows_written:,}")
    print(f"   Righe saltate : {rows_skipped:,}")
    print(f"   Output        : {output_path}")


def extract_kcore(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Estrae il k-core dal dataframe"""
    df.drop_duplicates(inplace=True)
    print(df.columns)
    print(len(df))
    if k > 1:
        while True:
            print('kcore...')
            initial_len = len(df)
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()

            df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
            df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]
            print(len(df))
            if len(df) == initial_len:
                break

    return df


def convert_jsonl_to_csv(input_path: str, output_path: str, show_progress: bool = False):
    input_file = Path(input_path)
    if not input_file.exists():
        print(f"Errore: file non trovato → {input_path}")
        sys.exit(1)

    progress = None
    if show_progress:
        try:
            from tqdm import tqdm
            file_size = input_file.stat().st_size
            progress = tqdm(total=file_size, unit="B", unit_scale=True, desc="Elaborazione")
        except ImportError:
            print("tqdm non installato, procedo senza barra di avanzamento.")

    fields = ["user_id", "item_id", "rating", "text", "timestamp"]

    rows_written = 0
    rows_skipped = 0

    with open(input_path, "r", encoding="utf-8") as fin, \
            open(output_path, "w", encoding="utf-8", newline="") as fout:

        writer = csv.DictWriter(fout, fieldnames=fields, extrasaction="ignore",
                                quoting=csv.QUOTE_ALL)
        writer.writeheader()

        for line_num, line in enumerate(fin, start=1):
            if progress:
                progress.update(len(line.encode("utf-8")))

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  Riga {line_num}: JSON non valido ({e}) — saltata", file=sys.stderr)
                rows_skipped += 1
                continue

            item_id = sanitize(obj.get("asin") or obj.get("parent_asin") or obj.get("item_id", ""))

            row = {
                "user_id": sanitize(obj.get("user_id", "")),
                "item_id": item_id,
                "rating": obj.get("rating", ""),
                "text": sanitize(obj.get("text", "")),
                "timestamp": obj.get("timestamp", ""),
            }

            writer.writerow(row)
            rows_written += 1

            if not progress and rows_written % 500_000 == 0:
                print(f"  → {rows_written:,} righe scritte...")

    if progress:
        progress.close()

    print(f"\n✅ Completato!")
    print(f"   Righe scritte : {rows_written:,}")
    print(f"   Righe saltate : {rows_skipped:,}")
    print(f"   Output        : {output_path}")


def convert():
    ROOT_DIR = Path(__file__).resolve().parent
    # for dataset in ["Books", "Toys_and_Games"]:
    #     print('dataset', dataset)
    #     BASE_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}.jsonl").resolve()
    #     OUT_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}_merged.csv").resolve()
    #
    #     convert_jsonl_to_csv(BASE_DATA_DIR, OUT_DATA_DIR, show_progress=True)

    dataset = 'yelp'
    print('dataset', dataset)
    BASE_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/yelp_academic_dataset_review.json").resolve()
    OUT_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}_merged.csv").resolve()

    convert_yelp_jsonl_to_csv(BASE_DATA_DIR, OUT_DATA_DIR, True)


def temporal_leave_one_out_split(df, users='all'):
    # Assicurati che il timestamp sia datetime
    # df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ordina per utente e tempo
    df = df.sort_values(by=['user_id', 'timestamp'])
    # indice progressivo per ogni utente
    df['pos'] = df.groupby('user_id').cumcount()
    # dimensione del gruppo
    df['size'] = df.groupby('user_id')['user_id'].transform('size')
    # df['item_id'] = df['parent_asin']
    # condizioni
    train_df = df[(df['size'] < 3) | (df['pos'] < df['size'] - 2)]
    val_df = df[(df['size'] >= 3) & (df['pos'] == df['size'] - 2)]
    test_df = df[(df['size'] >= 3) & (df['pos'] == df['size'] - 1)]

    # pulizia

    user_interactions = df.groupby("user_id").size()

    # percentili
    p25 = user_interactions.quantile(0.25)
    p75 = user_interactions.quantile(0.75)

    print("25 percentile:", p25)
    print("75 percentile:", p75)

    # utenti con poche interazioni
    low_users = user_interactions[user_interactions <= p25].index.tolist()

    # utenti con molte interazioni
    high_users = user_interactions[user_interactions >= p75].index.tolist()

    train_df = train_df.drop(columns=['pos', 'size']).reset_index(drop=True)
    val_df = val_df.drop(columns=['pos', 'size']).reset_index(drop=True)
    test_df = test_df.drop(columns=['pos', 'size']).reset_index(drop=True)

    if users == 'active':
        val_df = val_df[val_df['user_id'].isin(high_users)]
        test_df = test_df[test_df['user_id'].isin(high_users)]
    if users == 'lt':
        val_df = val_df[val_df['user_id'].isin(low_users)]
        test_df = test_df[test_df['user_id'].isin(low_users)]

    return train_df, val_df, test_df


# ─────────────────────────────────────────────
# SALVATAGGIO IN FORMATO RECBOLE .csv
# ─────────────────────────────────────────────
DATETIME_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M:%S.%f",  # microsecondi: 2012-10-24 13:33:39.123456
    "%Y-%m-%dT%H:%M:%S",
    "%Y-%m-%dT%H:%M:%S.%f",
    "%Y-%m-%d",
]


def parse_timestamp(ts_str: str) -> float:
    import re
    ts_str = ts_str.strip().strip('"').strip("'")
    # Già numerico?
    try:
        return float(ts_str)
    except ValueError:
        pass
    # Tronca nanosecondi (9 cifre) a microsecondi (6 cifre)
    # es. "2012-10-24 13:33:39.000000000" → "2012-10-24 13:33:39.000000"
    ts_str = re.sub(r'(\.\d{6})\d+', r'\1', ts_str)
    for fmt in DATETIME_FORMATS:
        try:
            dt = datetime.strptime(ts_str, fmt)
            return dt.replace(tzinfo=datetime.timezone.utc).timestamp()
        except ValueError:
            continue
    raise ValueError(f"Timestamp non riconosciuto: '{ts_str}'")


def save_inter(df: pd.DataFrame, path: Path):
    """Salva un DataFrame nel formato .csv di RecBole (TSV con header tipizzato)."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce").astype("int64") // 1_000_000

    df = df.rename(columns={"user_id": "user_id:token", "item_id": "item_id:token", "rating": "rating:float",
                            "text": "text:token_seq", "timestamp": "timestamp:float"})

    df.to_csv(path, index=False, sep='\t')
    print(f"  salvato: {path} ({len(df):,} righe)")


# ─────────────────────────────────────────────
# PIPELINE PRINCIPALE
# ─────────────────────────────────────────────

# def convert():
#     ROOT_DIR = Path(__file__).resolve().parent
#     for dataset in ["Books", "Toys_and_Games"]:
#         print('dataset', dataset)
#         BASE_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}.jsonl").resolve()
#         OUT_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}.csv").resolve()
#         convert_jsonl_to_inter(BASE_DATA_DIR, OUT_DATA_DIR, show_progress=True)
#
#     dataset = 'yelp'
#     print('dataset', dataset)
#     BASE_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/yelp_academic_dataset_review.json").resolve()
#     OUT_DATA_DIR = (ROOT_DIR / f"../data_review/{dataset}/{dataset}.csv").resolve()
#     convert_yelp_jsonl_to_inter(BASE_DATA_DIR, OUT_DATA_DIR, True)


def preprocess():
    ROOT_DIR = Path(__file__).resolve().parent
    DATA_DIR = (ROOT_DIR / "../data_review").resolve()

    datasets = ["Books", "Toys_and_Games", "yelp"]
    datasets = ['yelp']
    for dataset in datasets:
        print(f"\n{'=' * 50}")
        print(f"  Preprocessing: {dataset}")
        print(f"{'=' * 50}")

        merged_inter = DATA_DIR / dataset / f"{dataset}_merged.csv"

        if not merged_inter.exists():
            print(f"  ⚠️  {merged_inter} non trovato, eseguo convert() prima.")
            convert()

        # Leggi il .csv (salta header tipizzato, usa nomi puliti)
        df = pd.read_csv(merged_inter)


        print(df.columns)
        print(f"  Righe caricate: {len(df):,}")

        # ── Dedup + k-core (k=10) sul dataset completo prima dello split ──
        print(f"\n  → Rimozione duplicati e k-core (k=10)...")
        before = len(df)
        df = df.drop_duplicates(subset=['user_id', 'item_id', 'rating', 'timestamp'])
        print(len(df))
        df = extract_kcore(df, k=10)

        print(f"  Righe dopo dedup+kcore: {len(df):,} (rimosse {before - len(df):,})")

        out = DATA_DIR / dataset

        # ── Split all users ──
        print("\n  → Split leave-one-out (all users)...")
        train_df, val_df, test_df = temporal_leave_one_out_split(df, users='all')
        save_inter(train_df, out / f"train_{dataset}.csv")
        save_inter(val_df, out / f"vali_{dataset}.csv")
        save_inter(test_df, out / f"test_{dataset}.csv")

        # ── Split active users ──
        print("\n  → Split leave-one-out (active users)...")
        _, val_active, test_active = temporal_leave_one_out_split(df, users='active')
        save_inter(val_active, out / f"vali_{dataset}_active.csv")
        save_inter(test_active, out / f"test_{dataset}_active.csv")

        # ── Split long-tail users ──
        print("\n  → Split leave-one-out (long-tail users)...")
        _, val_lt, test_lt = temporal_leave_one_out_split(df, users='lt')
        save_inter(val_lt, out / f"vali_{dataset}_lt.csv")
        save_inter(test_lt, out / f"test_{dataset}_lt.csv")

        print(f"\n  ✅ {dataset} completato.")


if __name__ == "__main__":
    #convert()
    preprocess()