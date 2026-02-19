import pandas as pd
import numpy as np
import random
from datetime import datetime
import os
import glob
import os
import glob
import pandas as pd
from datetime import datetime

def parse_timestamp(ts):
    # se è già stringa data

    if ts == 0:
        return 0

    if isinstance(ts, str):
        return pd.to_datetime(ts, errors="raise")

    if isinstance(ts, pd.Timestamp):
        return ts

    # se è numerico
    ts = int(ts)

    if ts < 1e11:          # ~ seconds (1970–5138)
        return pd.to_datetime(ts, unit="s")
    elif ts < 1e14:        # milliseconds
        return pd.to_datetime(ts, unit="ms")
    elif ts < 1e17:        # microseconds
        return pd.to_datetime(ts, unit="us")
    else:                  # nanoseconds
        return pd.to_datetime(ts, unit="ns")


def per_node_budget(node_size, total_left,min_per_node,max_per_node):
    n = np.random.randint(
        min_per_node,
        max_per_node+1
    )
    return min(n, node_size-1, total_left)

def sample_ratings_optimized(mu, sigma, n, config):

    if config.rating_behavior.sampling_strategy == 'gaussian':

        if not np.isfinite(mu):
            mu = (config.rating_behavior.min_rating +
                  config.rating_behavior.max_rating) / 2

        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0

        r = np.random.normal(mu, sigma, size=n)

        r = np.nan_to_num(
            r,
            nan=mu,
            posinf=config.rating_behavior.max_rating,
            neginf=config.rating_behavior.min_rating
        )

        return np.clip(
            np.rint(r),
            config.rating_behavior.min_rating,
            config.rating_behavior.max_rating
        ).astype(int)

    return np.random.randint(
        config.rating_behavior.min_rating,
        config.rating_behavior.max_rating + 1,
        size=n
    )

def sample_ratings(base_ratings, n, config):

    if config.rating_behavior.sampling_strategy == 'gaussian':

        mu = np.nanmean(base_ratings)
        sigma = np.nanstd(base_ratings)

        # fallback sicuri
        if not np.isfinite(mu):
            mu = (config.rating_behavior.min_rating +
                  config.rating_behavior.max_rating) / 2

        if not np.isfinite(sigma) or sigma == 0:
            sigma = 1.0

        r = np.random.normal(mu, sigma, size=n)

        r = np.nan_to_num(
            r,
            nan=mu,
            posinf=config.rating_behavior.max_rating,
            neginf=config.rating_behavior.min_rating
        )

        return np.clip(
            np.rint(r),
            config.rating_behavior.min_rating,
            config.rating_behavior.max_rating
        ).astype(int)

    return np.random.randint(
        config.rating_behavior.min_rating,
        config.rating_behavior.max_rating + 1,
        size=n
    )

# def sample_ratings(base_ratings, n, config):
#     if config.rating_behavior.sampling_strategy == 'gaussian':
#         mu, sigma = base_ratings.mean(), base_ratings.std() or 1.0
#         r = np.random.normal(mu, sigma, size=n)
#         return np.clip(np.rint(r), config.rating_behavior.min_rating, config.rating_behavior.max_rating).astype(int)
#     return np.random.randint(config.rating_behavior.min_rating, config.rating_behavior.max_rating + 1, size=n)

def sample_reviews(df, config):
    if config.min_length_of_review > 0:
        token_counts = df['review_text'].fillna("").str.split().str.len()
        # Filtra
        df_filtered = df[token_counts >= config.min_length_of_review].copy()
        return df_filtered
    return df

def ordered_nodes(df, target, strategy):
    nodes = df[target].value_counts(ascending=strategy == 'least').index.tolist()
    if strategy == 'uniform':
        random.shuffle(nodes)
    return nodes





def create_unique_table(dataset):

    input_folder = f"./baselines/results/{dataset}/"
    output_file = f"./baselines/results/{dataset}/comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"

    # 1️⃣ Prendo tutti i TSV
    files = glob.glob(os.path.join(input_folder, "*.tsv"))

    if not files:
        raise ValueError("Nessun file TSV trovato nella cartella.")

    # 2️⃣ Separo clean e noisy
    clean_files = [f for f in files if "clean" in os.path.basename(f)]
    noisy_files = [f for f in files if "noisy" in os.path.basename(f)]

    if not clean_files or not noisy_files:
        raise ValueError("Servono almeno un file clean e uno noisy.")

    # 3️⃣ Leggo e concateno tutti i clean
    df_clean = pd.concat(
        [pd.read_csv(f, sep="\t") for f in clean_files],
        ignore_index=True
    )

    # 4️⃣ Leggo e concateno tutti i noisy
    df_noisy = pd.concat(
        [pd.read_csv(f, sep="\t") for f in noisy_files],
        ignore_index=True
    )

    if "model" not in df_clean.columns or "model" not in df_noisy.columns:
        raise ValueError("Entrambi i dataframe devono avere la colonna 'model'.")

    # 5️⃣ Aggiungo suffissi
    df_clean = df_clean.copy()
    df_noisy = df_noisy.copy()

    df_clean["model"] = df_clean["model"].astype(str) + "_clean"
    df_noisy["model"] = df_noisy["model"].astype(str) + "_noisy"

    # 6️⃣ Base model
    df_clean["base_model"] = df_clean["model"].str.replace("_clean", "", regex=False)
    df_noisy["base_model"] = df_noisy["model"].str.replace("_noisy", "", regex=False)

    # 7️⃣ Merge globale
    merged = pd.merge(
        df_clean,
        df_noisy,
        on="base_model",
        how="outer",
        suffixes=("_clean", "_noisy")
    )

    # 8️⃣ Alterno righe
    rows = []

    for _, row in merged.iterrows():

        # CLEAN
        clean_cols = [c for c in merged.columns if c.endswith("_clean")]
        if not pd.isna(row.get("model_clean")):
            clean_row = {
                col.replace("_clean", ""): row[col]
                for col in clean_cols
            }
            rows.append(clean_row)

        # NOISY
        noisy_cols = [c for c in merged.columns if c.endswith("_noisy")]
        if not pd.isna(row.get("model_noisy")):
            noisy_row = {
                col.replace("_noisy", ""): row[col]
                for col in noisy_cols
            }
            rows.append(noisy_row)

    final_df = pd.DataFrame(rows)

    # 9️⃣ Salvo
    final_df.to_csv(output_file, sep="\t", index=False)

    print(f"File salvato come: {output_file}")

    return final_df
