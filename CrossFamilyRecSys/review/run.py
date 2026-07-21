
#!/usr/bin/env python3
"""
run_review_models.py
====================
Allena DeepCoNN, NARRE e TransNet su tutte le varianti di rumore.

DIFFERENZE RISPETTO A run_cf.py (modelli CF RecBole)
-----------------------------------------------------
1. PIPELINE CUSTOM — niente RecBole
   DeepCoNN, NARRE e TransNet leggono recensioni testuali (word2vec + review
   tokenizzati). Non sono compatibili con l'interfaccia RecBole. Il training
   usa PyTorch puro: DataLoader custom, loop epochs, early stopping manuale.

2. PREPROCESSING AL VOLO
   Prima del training il runner costruisce da zero per ogni run:
     • vocab + word2vec (GloVe 300d o random se non disponibile)
     • review tokenizzate per utente e per item
     • split train/vali/test (80/10/10 random seed fisso)
   I file preprocessati vengono salvati in staging_review/<tag>/
   e riusati se già esistenti (mtime check).

3. FORMATO DATI SORGENTE
   I file .inter (o .csv) devono contenere almeno le colonne:
     user_id, item_id, rating, timestamp, review_text
   Il campo review_text viene tokenizzato e paddato a max_words per review.
   Ogni utente/item mantiene le ultime max_reviews recensioni (troncato/paddato).

4. METRICHE — solo rating prediction
   RMSE, MAE, AUC (binarizzando rating ≥ POS_THRESHOLD = 4).
   Calcolate su test set standard e sui subset active / long-tail.
   Niente beyond-accuracy (coverage/novelty/gini): non applicabile a
   modelli che producono un singolo score scalare per coppia (u, i).

5. MODELLI SUPPORTATI
   • deepconn   — TextCNN su recensioni concatenate, FM finale
   • deepconn++ — TextCNN + bias espliciti user/item
   • narre      — TextCNN per-review + attention pesata su item/user ID
   • transnet   — Source (u+i review) + Target (review comune), loss mse+kl
   • transnet++ — TransNet con embedding ID espliciti nel source

6. LOSS
   DeepCoNN / NARRE: MSE(pred, rating).
   TransNet / TransNet++: MSE(source_pred, rating) + MSE(target_pred, rating)
                          + lambda_mse * ||source_ir - target_ir||^2
   Stessa loss dei paper originali.

7. EARLY STOPPING
   Monitora RMSE su validation set.
   Patience configurabile (default 10 epoch).
   Salva il checkpoint migliore su disco (saved_review/<tag>/<model>.pth).

8. STRUTTURA OUTPUT (compatibile con run_cf.py)
   Ogni run produce un dict:
     {model, dataset, noise_type, level, status,
      elapsed_s, best_valid_score,
      checkpoint_dir, test_result, test_result_active, test_result_lt}
   Salvato in append su results/results_review_summary.jsonl.
"""

import argparse
import csv
import json
import logging
import math
import pickle
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from tqdm import tqdm
try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ── Compatibilità PyTorch ≥ 2.6 ──────────────────────────────────────────────
_torch_load_orig = torch.load
def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _torch_load_orig(*args, **kwargs)
torch.load = _torch_load_compat

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ==============================================================================
#  PATHS
# ==============================================================================
ROOT_DIR      = Path(__file__).resolve().parent
_ds_suffix = os.environ.get("DATASET_NAME", "all")
BASE_DATA_DIR = (ROOT_DIR / "../../data_review").resolve()
RESULTS_FILE  = ROOT_DIR / "results" / f"results_review_summary_{_ds_suffix}_sent.jsonl"
SAVED_DIR     = ROOT_DIR / "saved_review"
STAGING_DIR   = ROOT_DIR / "staging_review"
GLOVE_PATH   = ROOT_DIR / "glove.6B.300d.txt"


# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = {
    "yelp":          BASE_DATA_DIR / "yelp",
    "Books":         BASE_DATA_DIR / "Books",
    "Toys_and_Games": BASE_DATA_DIR / "Toys_and_Games",
}

ALL_MODELS = ["deepconn", "narre", "transnet"]

ALL_NOISE_TYPES = [
    "clean",
    "rating_flip","sentiment_flip","context_swap","sequence_disruption",
    "random_burst_fake",
    "targeted_injection_fake",

]
#ALL_NOISE_TYPES = ['sentiment_flip']

# ==============================================================================
#  IPERPARAMETRI PER DATASET × MODELLO
# ==============================================================================
#
# LOGICA GENERALE
# ───────────────
# word_dim          : dimensione embedding GloVe (300 standard)
# latent_size       : dimensione feature estratte da TextCNN (= num_filters)
# kernel_sizes      : kernel CNN multi-scala
# max_reviews       : recensioni per utente/item (troncato o paddato con review vuota)
# max_words         : parole per recensione (troncato o paddato con PAD_IDX=0)
# batch_size        : campioni per step
# learning_rate     : Adam lr
# dropout           : dropout su feature e layer lineari
# epochs / patience : early stopping su RMSE validazione
# lambda_mse        : peso loss ||source_ir - target_ir||^2 (solo TransNet*)
#
# DIMENSIONI DEI DATASET
#   Toys_and_Games → "piccolo"  (poche recensioni/utente, catalogo 14K)
#   yelp           → "medio"    (recensioni più lunghe, catalogo 56K)
#   Books          → "grande"   (molte recensioni/item ma brevi, catalogo 113K)
#
# PRINCIPI PER PARAMETRO
# ───────────────────────
# latent_size
#   64 su Toys: catalogo piccolo, meno feature necessarie, meno rischio overfitting.
#   64 su Yelp/Books: allineato a embedding_size dei modelli CF per confronto.
#
# max_reviews
#   Toys: 10 — utenti con poche recensioni, prendere di più aggiunge rumore.
#   Yelp: 15 — utenti più attivi, più recensioni utili.
#   Books: 12 — via di mezzo; Books ha molti item con pochissime recensioni.
#
# max_words
#   Toys: 150 — recensioni brevi.
#   Yelp: 200 — recensioni più lunghe (ristoranti, esperienze).
#   Books: 150 — recensioni mediamente brevi per libri Amazon.
#
# batch_size
#   512 su Toys/Yelp: dataset gestibili in memoria.
#   256 su Books: dataset grande, batch più piccoli per stabilità gradienti.
#
# patience
#   15 su tutti: sufficiente per distinguere plateau reale da fluttuazioni.


MODEL_HPARAMS: dict[str, dict[str, dict]] = {

    "Toys_and_Games": {
        "deepconn": {
            "word_dim":        300,
            "latent_size":     64,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      2048,
            "learning_rate":   0.001,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "deepconn++": {
            "word_dim":        300,
            "latent_size":     64,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      2048,
            "learning_rate":   0.001,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "narre": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      2048,
            "learning_rate":   0.001,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "transnet": {
            "word_dim":        300,
            "latent_size":     64,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      2048,
            "learning_rate":   0.001,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "transnet++": {
            "word_dim":        300,
            "latent_size":     64,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      2048,
            "learning_rate":   0.001,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
    },

    "yelp": {
        "deepconn": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      4096,
            "learning_rate":   0.0008,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "deepconn++": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      4096,
            "learning_rate":   0.0008,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "narre": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      4096,
            "learning_rate":   0.0008,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "transnet": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      4096,
            "learning_rate":   0.0008,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
        "transnet++": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      4096,
            "learning_rate":   0.0008,
            "weight_decay":    1e-5,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          100,
            "patience":        20,
            "word_embed_size": 300,
        },
    },

    "Books": {
        "deepconn": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      8192,
            "learning_rate":   0.0005,
            "weight_decay":    1e-4,
            "dropout":         0.5,
            "epochs":          120,
            "patience":        25,
            "word_embed_size": 300,
        },
        "deepconn++": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      8192,
            "learning_rate":   0.0005,
            "weight_decay":    1e-4,
            "dropout":         0.5,
            "epochs":          120,
            "patience":        25,
            "word_embed_size": 300,
        },
        "narre": {
            "word_dim":        300,
            "latent_size":     256,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      8192,
            "learning_rate":   0.0005,
            "weight_decay":    1e-4,
            "dropout":         0.5,
            "epochs":          120,
            "patience":        25,
            "word_embed_size": 300,
        },
        "transnet": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      8192,
            "learning_rate":   0.0005,
            "weight_decay":    1e-4,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          120,
            "patience":        25,
            "word_embed_size": 300,
        },
        "transnet++": {
            "word_dim":        300,
            "latent_size":     128,
            "kernel_sizes":    [3, 4, 5],
            "max_reviews":     10,
            "max_words":       100,
            "batch_size":      8192,
            "learning_rate":   0.0005,
            "weight_decay":    1e-4,
            "dropout":         0.5,
            "lambda_mse":      0.1,
            "epochs":          120,
            "patience":        25,
            "word_embed_size": 300,
        },
    },
}


MODEL_HPARAMSSS: dict[str, dict[str, dict]] = {

    # ── Toys & Games ──────────────────────────────────────────────────────────
    "Toys_and_Games": {
        "deepconn": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   10,
            "max_words":     150,
            "batch_size":    512,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,
            "word_embed_size": 300,

        },
        "deepconn++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   10,
            "max_words":     150,
            "batch_size":    512,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,
            "word_embed_size": 300,

        },
        "narre": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   10,
            "max_words":     150,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,
            "word_embed_size": 300,

        },
        "transnet": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   10,
            "max_words":     150,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
        "transnet++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   10,
            "max_words":     150,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        80,
            "patience":      15,
            "word_embed_size": 300,

        },
    },

    # ── Yelp ──────────────────────────────────────────────────────────────────
    "yelp": {
        "deepconn": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   15,
            "max_words":     200,
            "batch_size":    512,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
        "deepconn++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   15,
            "max_words":     200,
            "batch_size":    512,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
        "narre": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   15,
            "max_words":     200,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
        "transnet": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   15,
            "max_words":     200,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
        "transnet++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   15,
            "max_words":     200,
            "batch_size":    256,
            "learning_rate": 0.001,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        80,
            "patience":      15,            "word_embed_size": 300,

        },
    },

    # ── Amazon Books ──────────────────────────────────────────────────────────
    "Books": {
        "deepconn": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   12,
            "max_words":     150,
            "batch_size":    256,
            "learning_rate": 0.0005,
            "dropout":       0.5,
            "epochs":        100,
            "patience":      15,            "word_embed_size": 300,

        },
        "deepconn++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   12,
            "max_words":     150,
            "batch_size":    256,
            "learning_rate": 0.0005,
            "dropout":       0.5,
            "epochs":        100,
            "patience":      15,            "word_embed_size": 300,

        },
        "narre": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   12,
            "max_words":     150,
            "batch_size":    128,
            "learning_rate": 0.0005,
            "dropout":       0.5,
            "epochs":        100,
            "patience":      15,            "word_embed_size": 300,

        },
        "transnet": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   12,
            "max_words":     150,
            "batch_size":    128,
            "learning_rate": 0.0005,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        100,
            "patience":      15,            "word_embed_size": 300,

        },
        "transnet++": {
            "word_dim":      300,
            "latent_size":   64,
            "kernel_sizes":  [3, 4, 5],
            "max_reviews":   12,
            "max_words":     150,
            "batch_size":    128,
            "learning_rate": 0.0005,
            "dropout":       0.5,
            "lambda_mse":    0.1,
            "epochs":        100,
            "patience":      15,            "word_embed_size": 300,

        },
    },
}


def get_model_hparams(dataset_name: str, model: str) -> dict:
    """
    Restituisce gli iperparametri per la coppia (dataset, model).
    Loga sempre cosa viene applicato; usa un default minimo se mancante.
    """
    hparams = MODEL_HPARAMS.get(dataset_name, {}).get(model, {})
    if hparams:
        log.info(
            f"  [hparams] {dataset_name}/{model} → "
            + ", ".join(f"{k}={v}" for k, v in hparams.items())
        )
    else:
        log.warning(
            f"  [hparams] Nessun override per {dataset_name}/{model}: "
            "uso defaults minimi"
        )
        hparams = {
            "word_dim": 50, "latent_size": 64, "kernel_sizes": [3, 4, 5],
            "max_reviews": 10, "max_words": 150, "batch_size": 256,
            "learning_rate": 0.001, "dropout": 0.5,"word_embed_size":128,
            "lambda_mse": 0.1, "epochs": 50, "patience": 10,
        }
    return hparams


# ==============================================================================
#  NOISE REGISTRY — identico a run_cf.py
# ==============================================================================
def build_noise_registry(dataset_name: str) -> dict:
    return {
        "clean": {
            "subdir": "clean",
            "levels": {"0.0": f"train_{dataset_name}.csv"},
        },
        "sequence_disruption": {
            "subdir": "sequence_disruption",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },
        "random_burst_fake": {
            "subdir": "random_burst_fake",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },
        "targeted_injection_fake": {
            "subdir": "targeted_injection_fake",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },
        "rating_flip": {
            "subdir": "rating_flip",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },
        "context_swap": {
            "subdir": "context_swap",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },

        "sentiment_flip": {
            "subdir": "sentiment_flip",
            "levels": {
                "0.10": "corrupted_0.10_1.inter",
                "0.20": "corrupted_0.20_1.inter",
                "0.30": "corrupted_0.30_1.inter",
            },
        },
    }

# ==============================================================================
#  PREPROCESSING — parsing, tokenizzazione, word2vec, split
# ==============================================================================

PAD_IDX   = 0    # indice parola di padding
UNK_IDX   = 1    # indice parola sconosciuta
SEED      = 2024  # seed per riproducibilità split


def _tokenize(text: str) -> list[str]:
    """Tokenizzazione minimale: lowercase + split su non-alfanumerico."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _parse_inter_with_review(path: Path) -> list[dict]:
    """
    Parsa un file .inter o .csv con colonne:
      user_id, item_id, rating, timestamp, review_text
    Restituisce lista di dict con queste chiavi.
    Tollera separatori tab, virgola, spazi multipli.
    La colonna review_text è opzionale: se manca, review = "".
    """
    with path.open(encoding="utf-8") as f:
        raw_header = f.readline().rstrip("\n")
        raw_lines  = [l.rstrip("\n") for l in f if l.strip()]

    if "\t" in raw_header:
        sep = "\t"
    elif re.search(r"  +", raw_header):
        sep = None
    else:
        sep = ","

    def split_line(line):
        return line.split(sep) if sep else line.split()

    cols = [c.strip().split(":")[0].lower() for c in split_line(raw_header)]

    def idx(name, alts=()):
        for n in (name,) + alts:
            if n in cols:
                return cols.index(n)
        return None

    uid_i  = idx("user_id")
    iid_i  = idx("item_id")
    ts_i   = idx("timestamp")
    rat_i  = idx("rating", ("rate", "ratings"))
    rev_i  = idx("review_text", ("review", "text", "reviewtext"))

    if uid_i is None or iid_i is None:
        raise ValueError(f"Colonne user_id/item_id mancanti in {path.name}. Colonne: {cols}")
    if rat_i is None:
        raise ValueError(f"Colonna rating mancante in {path.name}. Colonne: {cols}")

    rows = []
    for line in raw_lines:
        parts = split_line(line)
        needed = max(uid_i, iid_i, rat_i)
        if ts_i  is not None: needed = max(needed, ts_i)
        if rev_i is not None: needed = max(needed, rev_i)
        if len(parts) <= needed:
            continue
        try:
            rows.append({
                "user_id":     parts[uid_i].strip(),
                "item_id":     parts[iid_i].strip(),
                "rating":      float(parts[rat_i].strip()) ,
                "timestamp":   float(parts[ts_i].strip())   if ts_i  is not None else 0.0,
                "review_text": parts[rev_i].strip()         if rev_i is not None else "",
            })
        except Exception as e:
            print(parts)
    return rows


def _build_vocab_and_embeddings(
    rows: list[dict],
    word_dim: int,
    glove_path: Path | None = None,
) -> tuple[dict[str, int], np.ndarray]:
    """
    Costruisce vocab e matrice di embedding.
    Se glove_path è fornito e il file esiste, carica GloVe 300d.
    Altrimenti inizializza con vettori casuali (normale 0,0.1).

    Restituisce:
      word2idx : dict str → int  (PAD=0, UNK=1, parole da 2 in poi)
      embeddings: np.ndarray shape [vocab_size, word_dim]
    """
    # Costruisci vocab dal corpus
    counter: Counter = Counter()
    for r in rows:
        counter.update(_tokenize(r["review_text"]))

    word2idx: dict[str, int] = {"<PAD>": PAD_IDX, "<UNK>": UNK_IDX}
    for word, _ in counter.most_common():
        word2idx[word] = len(word2idx)

    vocab_size = len(word2idx)
    embeddings = np.random.normal(0, 0.1, (vocab_size, word_dim)).astype(np.float32)
    embeddings[PAD_IDX] = 0.0  # padding = zero vector

    # Carica GloVe se disponibile
    if glove_path is not None and glove_path.exists():
        log.info(f"  Caricamento GloVe da {glove_path} …")
        loaded = 0
        with glove_path.open(encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip().split(" ")
                word  = parts[0]
                if word in word2idx and len(parts) == word_dim + 1:
                    embeddings[word2idx[word]] = np.array(parts[1:], dtype=np.float32)
                    loaded += 1
        log.info(f"  GloVe: {loaded}/{vocab_size} parole caricate")
    else:
        log.info("  GloVe non disponibile — embedding casuali (normal 0, 0.1)")

    return word2idx, embeddings


def _encode_reviews(
    rows: list[dict],
    word2idx: dict[str, int],
    max_reviews: int,
    max_words: int,
) -> tuple[dict[str, list[list[int]]], dict[str, list[list[int]]]]:
    """
    Per ogni utente e item costruisce la lista di recensioni tokenizzate e paddate.

    Restituisce:
      user_reviews : dict uid_str → lista di max_reviews review, ognuna lunga max_words
      item_reviews : dict iid_str → stessa struttura
    """
    # Raggruppa review per utente e item
    user_rev_raw: dict[str, list[list[int]]] = defaultdict(list)
    item_rev_raw: dict[str, list[list[int]]] = defaultdict(list)

    for r in rows:
        tokens = _tokenize(r["review_text"])
        enc    = [word2idx.get(t, UNK_IDX) for t in tokens]
        # Tronca / padda a max_words
        enc = (enc[:max_words] + [PAD_IDX] * max_words)[:max_words]
        user_rev_raw[r["user_id"]].append(enc)
        item_rev_raw[r["item_id"]].append(enc)

    empty_review = [PAD_IDX] * max_words

    def _pad_reviews(rev_list: list[list[int]]) -> list[list[int]]:
        # Prendi le ultime max_reviews recensioni (le più recenti)
        rev_list = rev_list[-max_reviews:]
        # Padda con review vuote se ne mancano
        while len(rev_list) < max_reviews:
            rev_list.append(empty_review)
        return rev_list

    user_reviews = {uid: _pad_reviews(revs) for uid, revs in user_rev_raw.items()}
    item_reviews = {iid: _pad_reviews(revs) for iid, revs in item_rev_raw.items()}
    return user_reviews, item_reviews


def _split_rows(
    rows: list[dict],
    train_ratio: float = 0.8,
    vali_ratio:  float = 0.1,
    seed: int = SEED,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Split casuale 80/10/10 con seed fisso."""
    rng = random.Random(seed)
    shuffled = list(rows)
    rng.shuffle(shuffled)
    n     = len(shuffled)
    n_tr  = int(n * train_ratio)
    n_val = int(n * vali_ratio)
    return shuffled[:n_tr], shuffled[n_tr:n_tr + n_val], shuffled[n_tr + n_val:]


# ==============================================================================
#  STAGING — preprocessa e salva su disco
# ==============================================================================
def stage_review_dataset(
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
    hparams: dict,
    glove_path: Path | None = None,
) -> dict:
    """
    Legge i file sorgente, preprocessa e salva in staging_review/<tag>/:
      - rows_train.pkl, rows_vali.pkl, rows_test.pkl
      - word2vec.pkl   (np.ndarray [vocab_size x word_dim])
      - word2idx.pkl   (dict str→int)
      - user_reviews.pkl, item_reviews.pkl  (dict uid/iid → encoded reviews)
      - meta.json      (total_users, total_items, split sizes)

    Restituisce il dict di meta informazioni + path della staging dir.
    """
    tag        = f"{dataset_name}__{noise_type}__{level}"
    stage_dir  = staging_root / tag
    meta_path  = stage_dir / "meta.json"

    train_src = dataset_dir / train_subdir / train_filename
    vali_src  = dataset_dir / f"vali_{dataset_name}.csv"
    test_src  = dataset_dir / f"test_{dataset_name}.csv"

    for src in [train_src, vali_src, test_src]:
        if not src.exists():
            raise FileNotFoundError(f"File mancante: {src}")

    # Controlla se lo staging è già aggiornato
    if meta_path.exists():
        mtime_stage = meta_path.stat().st_mtime
        mtime_src   = max(s.stat().st_mtime for s in [train_src, vali_src, test_src])
        if mtime_src <= mtime_stage:
            log.info(f"  Staging già aggiornato: {stage_dir}")
            with meta_path.open() as f:
                meta = json.load(f)
            meta["stage_dir"] = stage_dir
            return meta

    stage_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"  Preprocessing → {stage_dir}")

    # Parsing
    train_rows = _parse_inter_with_review(train_src)
    vali_rows  = _parse_inter_with_review(vali_src)
    test_rows  = _parse_inter_with_review(test_src)

    # Se vali/test mancano di review, usa solo train per costruire il corpus
    all_rows = train_rows + vali_rows + test_rows

    # Vocab + embeddings (costruiti sul corpus completo per coprire più parole)
    word2idx, embeddings = _build_vocab_and_embeddings(all_rows, hparams["word_dim"], glove_path)

    # Encode reviews (su tutti i dati, per avere le review di ogni utente/item)
    user_reviews, item_reviews = _encode_reviews(
        all_rows, word2idx,
        hparams["max_reviews"], hparams["max_words"],
    )

    # Mappa uid/iid stringa → intero (0 = padding/unknown)
    all_users = sorted({r["user_id"] for r in all_rows})
    all_items = sorted({r["item_id"] for r in all_rows})
    uid2int   = {u: i + 1 for i, u in enumerate(all_users)}   # 1-based, 0 = PAD
    iid2int   = {it: i + 1 for i, it in enumerate(all_items)}

    # Salvataggio
    def _dump(obj, name):
        with (stage_dir / name).open("wb") as f:
            pickle.dump(obj, f, protocol=4)

    _dump(train_rows,    "rows_train.pkl")
    _dump(vali_rows,     "rows_vali.pkl")
    _dump(test_rows,     "rows_test.pkl")
    _dump(embeddings,    "word2vec.pkl")
    _dump(word2idx,      "word2idx.pkl")
    _dump(user_reviews,  "user_reviews.pkl")
    _dump(item_reviews,  "item_reviews.pkl")
    _dump(uid2int,       "uid2int.pkl")
    _dump(iid2int,       "iid2int.pkl")

    meta = {
        "total_users": len(all_users),
        "total_items": len(all_items),
        "vocab_size":  len(word2idx),
        "n_train":     len(train_rows),
        "n_vali":      len(vali_rows),
        "n_test":      len(test_rows),
        "tag":         tag,
    }
    with meta_path.open("w") as f:
        json.dump(meta, f, indent=2)

    log.info(
        f"  Staging OK — utenti={meta['total_users']}, item={meta['total_items']}, "
        f"vocab={meta['vocab_size']}, train={meta['n_train']}, "
        f"vali={meta['n_vali']}, test={meta['n_test']}"
    )
    meta["stage_dir"] = stage_dir
    return meta


def _load_stage(stage_dir: Path) -> dict:
    """Carica tutti i pickle di una staging dir in un dict."""
    def _load(name):
        with (stage_dir / name).open("rb") as f:
            return pickle.load(f)

    return {
        "rows_train":    _load("rows_train.pkl"),
        "rows_vali":     _load("rows_vali.pkl"),
        "rows_test":     _load("rows_test.pkl"),
        "word2vec":      _load("word2vec.pkl"),
        "word2idx":      _load("word2idx.pkl"),
        "user_reviews":  _load("user_reviews.pkl"),
        "item_reviews":  _load("item_reviews.pkl"),
        "uid2int":       _load("uid2int.pkl"),
        "iid2int":       _load("iid2int.pkl"),
    }


# ==============================================================================
#  DATASET PYTORCH
# ==============================================================================
# class ReviewDataset(Dataset):
#     """
#     Dataset PyTorch per modelli review-based.
#
#     Ogni campione è una tupla:
#       (this_review, users_who_reviewed_item, reviewed_items_by_user,
#        user_reviews_enc, item_reviews_enc, user_id_int, item_id_int, rating)
#
#     • this_review : concatenazione user+item review (per TransNet target)
#     • users_who_reviewed_item : lista uid interi degli ultimi max_reviews utenti
#                                  che hanno recensito questo item (per NARRE)
#     • reviewed_items_by_user  : lista iid interi degli ultimi max_reviews item
#                                  recensiti dall'utente (per NARRE)
#     • user_reviews_enc        : [max_reviews x max_words] int
#     • item_reviews_enc        : [max_reviews x max_words] int
#     • user_id_int, item_id_int: interi per bias/embedding
#     • rating                  : float
#     """
#
#     def __init__(
#         self,
#         rows:          list[dict],
#         user_reviews:  dict[str, list[list[int]]],
#         item_reviews:  dict[str, list[list[int]]],
#         uid2int:       dict[str, int],
#         iid2int:       dict[str, int],
#         max_reviews:   int,
#         max_words:     int,
#     ):
#         self.rows         = rows
#         self.user_reviews = user_reviews
#         self.item_reviews = item_reviews
#         self.uid2int      = uid2int
#         self.iid2int      = iid2int
#         self.max_reviews  = max_reviews
#         self.max_words    = max_words
#
#         # Precalcola per NARRE: item→lista uid che l'hanno recensito
#         #                       user→lista iid che ha recensito
#         item_to_users: dict[str, list[str]] = defaultdict(list)
#         user_to_items: dict[str, list[str]] = defaultdict(list)
#         for r in rows:
#             item_to_users[r["item_id"]].append(r["user_id"])
#             user_to_items[r["user_id"]].append(r["item_id"])
#         self.item_to_users = item_to_users
#         self.user_to_items = user_to_items
#
#     def __len__(self):
#         return len(self.rows)
#
#     def _pad_id_list(self, lst: list[int]) -> list[int]:
#         """Tronca/padda lista di id interi a max_reviews."""
#         lst = lst[-self.max_reviews:]
#         lst = lst + [0] * self.max_reviews
#         return lst[:self.max_reviews]
#
#     def __getitem__(self, idx):
#         r   = self.rows[idx]
#         uid = r["user_id"]
#         iid = r["item_id"]
#
#         empty_review = [PAD_IDX] * self.max_words
#
#         def _get_reviews(d, key):
#             revs = d.get(key, [])
#             revs = revs[-self.max_reviews:]
#             while len(revs) < self.max_reviews:
#                 revs.append(empty_review)
#             return revs[:self.max_reviews]
#
#         user_revs = _get_reviews(self.user_reviews, uid)   # [max_reviews x max_words]
#         item_revs = _get_reviews(self.item_reviews, iid)
#
#         # this_review: concatenazione appiattita (per TransNet target)
#         this_flat = [tok for rev in user_revs + item_revs for tok in rev]
#
#         # NARRE: id utenti che hanno recensito l'item
#         users_of_item = [
#             self.uid2int.get(u, 0)
#             for u in self.item_to_users.get(iid, [])
#         ]
#         users_of_item = self._pad_id_list(users_of_item)
#
#         # NARRE: id item recensiti dall'utente
#         items_of_user = [
#             self.iid2int.get(it, 0)
#             for it in self.user_to_items.get(uid, [])
#         ]
#         items_of_user = self._pad_id_list(items_of_user)
#
#         return (
#             torch.tensor(this_flat,      dtype=torch.long),
#             torch.tensor(users_of_item,  dtype=torch.long),
#             torch.tensor(items_of_user,  dtype=torch.long),
#             torch.tensor(user_revs,      dtype=torch.long),
#             torch.tensor(item_revs,      dtype=torch.long),
#             torch.tensor(self.uid2int.get(uid, 0), dtype=torch.long),
#             torch.tensor(self.iid2int.get(iid, 0), dtype=torch.long),
#             torch.tensor(r["rating"],    dtype=torch.float),
#         )

class ReviewDataset(Dataset):
    def __init__(self, rows, user_reviews, item_reviews, uid2int, iid2int, max_reviews, max_words):
        n = len(rows)

        # Pre-alloca array numpy — fatto una volta sola
        self.user_revs_arr = np.zeros((n, max_reviews, max_words), dtype=np.int32)
        self.item_revs_arr = np.zeros((n, max_reviews, max_words), dtype=np.int32)
        self.this_flat_arr = np.zeros((n, 2 * max_reviews * max_words), dtype=np.int32)
        self.users_of_item_arr = np.zeros((n, max_reviews), dtype=np.int64)
        self.items_of_user_arr = np.zeros((n, max_reviews), dtype=np.int64)
        self.uid_arr = np.zeros(n, dtype=np.int64)
        self.iid_arr = np.zeros(n, dtype=np.int64)
        self.rating_arr = np.zeros(n, dtype=np.float32)

        # Precalcola item→utenti e user→item (uguale a prima)
        item_to_users = defaultdict(list)
        user_to_items = defaultdict(list)
        for r in rows:
            item_to_users[r["item_id"]].append(r["user_id"])
            user_to_items[r["user_id"]].append(r["item_id"])

        empty = [PAD_IDX] * max_words

        for i, r in enumerate(rows):
            uid = r["user_id"]
            iid = r["item_id"]

            self.uid_arr[i] = uid2int.get(uid, 0)
            self.iid_arr[i] = iid2int.get(iid, 0)
            self.rating_arr[i] = r["rating"]

            # User reviews
            u_revs = (user_reviews.get(uid, []) + [empty] * max_reviews)[:max_reviews]
            for j, rev in enumerate(u_revs[:max_reviews]):
                self.user_revs_arr[i, j] = rev[:max_words]

            # Item reviews
            i_revs = (item_reviews.get(iid, []) + [empty] * max_reviews)[:max_reviews]
            for j, rev in enumerate(i_revs[:max_reviews]):
                self.item_revs_arr[i, j] = rev[:max_words]

            # this_flat (TransNet)
            self.this_flat_arr[i] = np.concatenate([
                self.user_revs_arr[i].ravel(),
                self.item_revs_arr[i].ravel(),
            ])

            # NARRE: utenti che hanno recensito l'item
            uoi = [uid2int.get(u, 0) for u in item_to_users.get(iid, [])]
            uoi = (uoi + [0] * max_reviews)[:max_reviews]
            self.users_of_item_arr[i] = uoi

            # NARRE: item recensiti dall'utente
            iou = [iid2int.get(it, 0) for it in user_to_items.get(uid, [])]
            iou = (iou + [0] * max_reviews)[:max_reviews]
            self.items_of_user_arr[i] = iou

    def __len__(self):
        return len(self.rating_arr)

    def __getitem__(self, idx):
        # Solo slice numpy — nessuna lista, nessun dict lookup, nessun torch.tensor da lista
        return (
            torch.from_numpy(self.this_flat_arr[idx]),
            torch.from_numpy(self.users_of_item_arr[idx]),
            torch.from_numpy(self.items_of_user_arr[idx]),
            torch.from_numpy(self.user_revs_arr[idx]),
            torch.from_numpy(self.item_revs_arr[idx]),
            torch.tensor(self.uid_arr[idx]),
            torch.tensor(self.iid_arr[idx]),
            torch.tensor(self.rating_arr[idx]),
        )


# ==============================================================================
#  MODELLI PYTORCH (wrappers inline di DeepCoNN, NARRE, TransNet)
#  Questi importano dai file originali — non li riscriviamo.
# ==============================================================================
# Il runner si aspetta che DeepCoNN, NARRE, TransNet siano importabili
# dalla directory corrente (stessa struttura del progetto originale).
# Se l'import fallisce, solleva un errore chiaro.

def _import_models():
    try:
        from models.DeepCoNN import DeepCoNN
        from models.NARRE    import NARRE
        from models.TransNet import TransNet
    except ImportError as e:
        raise ImportError(
            f"Impossibile importare i modelli: {e}\n"
            "Assicurati che DeepCoNN.py, NARRE.py, TransNet.py siano "
            "nella stessa directory di run_review_models.py e che "
            "pytorch_models/common_pytorch_models.py e models.utils.py siano presenti."
        ) from e
    return DeepCoNN, NARRE, TransNet


def _build_model(model_name: str, hyper_params: dict, stage_dir: Path) -> nn.Module:
    """
    Istanzia il modello corretto in base al nome.
    hyper_params deve contenere tutti i campi richiesti dai costruttori.
    word2vec viene caricato da stage_dir/word2vec.pkl tramite load_obj.

    Sovrascriviamo temporaneamente utils.load_obj per puntare allo staging.
    """
    import models.utils as _utils
    _orig_load_obj = _utils.load_obj

    # load_obj(path) viene chiamato con hyper_params['data_dir'] + '/word2vec'
    # Sostituiamo con una versione che legge da stage_dir/word2vec.pkl
    def _patched_load_obj(path):
        pkl_path = Path(str(path) + ".pkl")
        if not pkl_path.exists():
            pkl_path = Path(path)
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    _utils.load_obj = _patched_load_obj

    try:
        DeepCoNN, NARRE, TransNet = _import_models()

        if model_name in ("deepconn", "deepconn++"):
            model = DeepCoNN(hyper_params)
        elif model_name == "narre":
            model = NARRE(hyper_params)
        elif model_name in ("transnet", "transnet++"):
            model = TransNet(hyper_params)
        else:
            raise ValueError(f"Modello sconosciuto: {model_name}")
    finally:
        _utils.load_obj = _orig_load_obj

    return model


# ==============================================================================
#  METRICHE
# ==============================================================================
POS_THRESHOLD = 4.0


def compute_auc_from_ratings(
    true_ratings: np.ndarray,
    pred_ratings: np.ndarray,
) -> float | None:
    if not _SKLEARN_OK:
        log.warning("  sklearn non disponibile — AUC non calcolato.")
        return None
    y_true = (true_ratings >= POS_THRESHOLD).astype(int)
    if len(np.unique(y_true)) < 2:
        log.warning(
            f"  AUC non calcolabile: una sola classe "
            f"(threshold={POS_THRESHOLD}, pos={y_true.sum()}/{len(y_true)})"
        )
        return None
    try:
        return float(roc_auc_score(y_true, pred_ratings))
    except Exception as exc:
        log.warning(f"  AUC fallito: {exc}")
        return None


def compute_metrics(
    preds:  np.ndarray,
    truths: np.ndarray,
    label:  str = "",
) -> dict:
    rmse = float(np.sqrt(np.mean((preds - truths) ** 2)))
    mae  = float(np.mean(np.abs(preds - truths)))
    auc  = compute_auc_from_ratings(truths, preds)
    prefix = f"  [{label}] " if label else "  "
    log.info(
        prefix + f"RMSE={rmse:.4f}  MAE={mae:.4f}"
        + (f"  AUC={auc:.4f}" if auc is not None else "  AUC=N/A")
    )
    res = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
    if auc is not None:
        res["auc"] = round(auc, 4)
    return res


# ==============================================================================
#  FORWARD UNIFICATO
#  I tre modelli hanno firme di forward diverse. Questa funzione unifica
#  la chiamata e gestisce la loss speciale di TransNet.
# ==============================================================================

def _forward_and_loss(
    model:      nn.Module,
    batch:      tuple,
    model_name: str,
    lambda_mse: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Esegue forward e calcola la loss.

    Restituisce (loss, preds) dove preds è il tensor di predizioni scalari
    (già detached dal grafo per le metriche).

    Formato batch: (this_review, users_of_item, items_of_user,
                    user_revs, item_revs, uid_int, iid_int, rating)
    """
    (this_rev, users_of_item, items_of_user,
     user_revs, item_revs, uid_int, iid_int, rating) = batch

    data = (this_rev, users_of_item, items_of_user,
            user_revs, item_revs, uid_int, iid_int)

    if model_name in ("transnet", "transnet++"):
        # TransNet restituisce [source_pred, target_pred, mse_ir]
        output = model(data)
        source_pred, target_pred, mse_ir = output
        loss = (
            F.mse_loss(source_pred, rating)
            + F.mse_loss(target_pred, rating)
            + lambda_mse * mse_ir
        )
        preds = source_pred.detach()
    else:
        # DeepCoNN / NARRE restituiscono un singolo vettore di score
        preds_t = model(data)
        loss    = F.mse_loss(preds_t, rating)
        preds   = preds_t.detach()

    return loss, preds


# ==============================================================================
#  INFERENCE LOOP
# ==============================================================================
def _predict(
    model:      nn.Module,
    loader:     DataLoader,
    model_name: str,
    device:     torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Esegue inference su un DataLoader.
    Restituisce (preds, truths) come array numpy.
    """
    model.eval()
    all_preds, all_truths = [], []
    with torch.no_grad():
        for batch in loader:
            batch  = tuple(t.to(device) for t in batch)
            rating = batch[-1]
            _, preds = _forward_and_loss(model, batch, model_name)
            all_preds.extend(preds.cpu().tolist())
            all_truths.extend(rating.cpu().tolist())
    return np.array(all_preds, dtype=float), np.array(all_truths, dtype=float)


# ==============================================================================
#  SUBSET LOADER — filtra per set di uid stringa
# ==============================================================================
def _filter_rows_by_users(rows: list[dict], user_set: set[str]) -> list[dict]:
    return [r for r in rows if r["user_id"] in user_set]


def load_user_set_from_csv(path: Path) -> set[str]:
    """
    Legge un file CSV/inter con colonna user_id e restituisce l'insieme
    degli uid stringa. Usato per subset active/lt.
    """
    if not path.exists():
        return set()
    try:
        rows = _parse_inter_with_review(path)
        return {r["user_id"] for r in rows}
    except Exception:
        # Fallback: CSV semplice con header, prima colonna = user_id
        uids = set()
        with path.open(encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                uid = row.get("user_id") or row.get("USER_ID") or ""
                if uid:
                    uids.add(uid.strip())
        return uids


# ==============================================================================
#  SINGLE RUN
# ==============================================================================
import traceback
def run_single(
    model_name:    str,
    dataset_name:  str,
    dataset_dir:   Path,
    noise_type:    str,
    level:         str,
    train_filename: str,
    train_subdir:  str,
    staging_root:  Path,
    glove_path:    Path | None = None,
) -> dict:
    log.info("=" * 72)
    log.info(
        f"  Model={model_name}  Dataset={dataset_name}  "
        f"Noise={noise_type}  Level={level}"
    )
    log.info("=" * 72)

    t0 = time.time()

    # ── 1. Iperparametri ──────────────────────────────────────────────────────
    hparams = get_model_hparams(dataset_name, model_name)

    # ── 2. Staging / preprocessing ────────────────────────────────────────────
    try:
        meta = stage_review_dataset(
            dataset_name   = dataset_name,
            dataset_dir    = dataset_dir,
            noise_type     = noise_type,
            level          = level,
            train_filename = train_filename,
            train_subdir   = train_subdir,
            staging_root   = staging_root,
            hparams        = hparams,
            glove_path     = glove_path,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error(f"  ✗ Staging fallito: {e}")
        traceback.print_exc()
        return _err(model_name, dataset_name, noise_type, level, str(e), 0)

    stage_dir = meta["stage_dir"]

    # ── 3. Carica dati preprocessati ─────────────────────────────────────────
    stage = _load_stage(stage_dir)

    # hyper_params nel formato atteso dai modelli originali
    hyper_params = {
        "data_dir":     str(stage_dir),
        "model_type":   model_name,
        "latent_size":  hparams["latent_size"],
        "word_dim":     hparams["word_dim"],
        "kernel_sizes": hparams["kernel_sizes"],
        "dropout":      hparams["dropout"],
        "word_embed_size": hparams["word_embed_size"],
        "total_users":  meta["total_users"],
        "total_items":  meta["total_items"],
    }

    max_reviews = hparams["max_reviews"]
    max_words   = hparams["max_words"]

    def _make_loader(rows, shuffle=False,workers=2):
        ds = ReviewDataset(
            rows         = rows,
            user_reviews = stage["user_reviews"],
            item_reviews = stage["item_reviews"],
            uid2int      = stage["uid2int"],
            iid2int      = stage["iid2int"],
            max_reviews  = max_reviews,
            max_words    = max_words,
        )
        return DataLoader(
            ds,
            batch_size  = hparams["batch_size"],
            shuffle     = shuffle,
            num_workers = 2,
            pin_memory  = True,
        )
    if dataset_name == 'Toys_and_Games':
        workers = 4
    else:
        workers = 2
    train_loader = _make_loader(stage["rows_train"], shuffle=True,workers=workers)
    vali_loader  = _make_loader(stage["rows_vali"],  shuffle=False,workers=workers)
    test_loader  = _make_loader(stage["rows_test"],  shuffle=False,workers=workers)

    # ── 4. Costruzione modello ────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"  Device: {device}")

    try:
        model = _build_model(model_name, hyper_params, stage_dir)
        model = model.to(device)
    except Exception as e:
        log.error(f"  ✗ Build modello fallito: {e}")
        return _err(model_name, dataset_name, noise_type, level, str(e), time.time() - t0)

    optimizer   = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    lambda_mse  = hparams.get("lambda_mse", 0.1)
    patience    = hparams.get("patience", 10)
    epochs      = hparams.get("epochs", 50)

    run_saved_dir = SAVED_DIR / f"{dataset_name}__{noise_type}__{level}" / model_name
    run_saved_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt_path = run_saved_dir / "best_model.pth"

    # ── 5. Training loop con early stopping ───────────────────────────────────
    best_vali_rmse = float("inf")
    no_improve     = 0

    try:
        log.info(
            f"START  "
        )
        for epoch in tqdm(range(1, epochs + 1), desc="Epochs", unit="epoch"):

            model.train()
            total_loss = 0.0
            n_batches  = 0

            for batch in tqdm(train_loader, desc=f"  Epoch {epoch}", leave=False):
                batch = tuple(t.to(device) for t in batch)
                optimizer.zero_grad()
                loss, _ = _forward_and_loss(model, batch, model_name, lambda_mse)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                optimizer.step()
                total_loss += loss.item()
                n_batches  += 1

            avg_loss = total_loss / max(n_batches, 1)

            # Validazione
            vali_preds, vali_truths = _predict(model, vali_loader, model_name, device)
            vali_rmse = float(np.sqrt(np.mean((vali_preds - vali_truths) ** 2)))

            log.info(
                f"  Epoch {epoch:3d}/{epochs}  "
                f"loss={avg_loss:.4f}  vali_RMSE={vali_rmse:.4f}"
                + ("  ★" if vali_rmse < best_vali_rmse else "")
            )

            if vali_rmse < best_vali_rmse:
                best_vali_rmse = vali_rmse
                no_improve     = 0
                torch.save(
                    {"state_dict": model.state_dict(), "hyper_params": hyper_params},
                    best_ckpt_path,
                )
            else:
                no_improve += 1
                if no_improve >= patience:
                    log.info(f"  Early stopping a epoch {epoch} (patience={patience})")
                    break

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"  ✗ Training FALLITO dopo {elapsed:.1f}s — {exc}")
        return _err(model_name, dataset_name, noise_type, level, str(exc), elapsed)

    elapsed = time.time() - t0
    log.info(f"  ✓ Training completato in {elapsed:.1f}s  best_vali_RMSE={best_vali_rmse:.4f}")

    # ── 6. Carica checkpoint migliore e valuta sul test set ───────────────────
    try:
        ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["state_dict"])
    except Exception as e:
        log.warning(f"  Checkpoint non caricato ({e}), uso il modello corrente")

    test_preds, test_truths = _predict(model, test_loader, model_name, device)
    test_result = compute_metrics(test_preds, test_truths, label="test")

    # ── 7. Eval su subset active / long-tail ─────────────────────────────────
    active_users = load_user_set_from_csv(dataset_dir / f"test_{dataset_name}_active.csv")
    lt_users     = load_user_set_from_csv(dataset_dir / f"test_{dataset_name}_lt.csv")

    def _eval_subset(user_set: set[str], subset_label: str) -> dict:
        if not user_set:
            return {"error": f"user_set vuoto per {subset_label}"}
        rows_sub = _filter_rows_by_users(stage["rows_test"], user_set)
        if not rows_sub:
            return {"error": f"Nessuna riga test per {subset_label}"}
        loader_sub = _make_loader(rows_sub, shuffle=False)
        preds, truths = _predict(model, loader_sub, model_name, device)
        log.info(f"  → Eval {subset_label} ({len(rows_sub)} interazioni, {len(user_set)} utenti)")
        return compute_metrics(preds, truths, label=subset_label)

    result_active = _eval_subset(active_users, "active")
    result_lt     = _eval_subset(lt_users,     "lt")

    return {
        "model":              model_name,
        "dataset":            dataset_name,
        "noise_type":         noise_type,
        "level":              level,
        "status":             "ok",
        "elapsed_s":          round(elapsed, 1),
        "best_valid_score":   round(best_vali_rmse, 4),
        "checkpoint_dir":     str(run_saved_dir),
        "test_result":        test_result,
        "test_result_active": result_active,
        "test_result_lt":     result_lt,
    }


# ==============================================================================
#  HELPERS
# ==============================================================================
def _err(model, dataset, noise, level, msg, elapsed):
    return {
        "model": model, "dataset": dataset,
        "noise_type": noise, "level": level,
        "status": "error", "error": msg,
        "elapsed_s": round(elapsed, 1),
    }


def save_result(m: dict):
    RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_FILE.open("a") as f:
        f.write(json.dumps(m) + "\n")


def checkpoint_exists(dataset_name, noise_type, level, model_name):
    ckpt = SAVED_DIR / f"{dataset_name}__{noise_type}__{level}" / model_name / "best_model.pth"
    return ckpt.exists()


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="Review-based models (DeepCoNN, NARRE, TransNet) — custom training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--models",        nargs="+", default=ALL_MODELS)
    p.add_argument("--noise",         nargs="+", default=ALL_NOISE_TYPES)
    p.add_argument("--levels",        nargs="+", default=["0.10"])
    p.add_argument("--staging-dir",   default=str(STAGING_DIR))
    p.add_argument("--glove",         default=GLOVE_PATH,

                        )
    p.add_argument("--skip-existing", action="store_true",
                   help="Salta run per cui esiste già un checkpoint.")
    p.add_argument("--dry-run",       action="store_true",
                   help="Stampa solo i run pianificati senza eseguire.")
    return p.parse_args()


def main():
    args         = parse_args()
    staging_root = Path(args.staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)
    glove_path   = Path(args.glove) if args.glove else None

    runs = []
    for dataset_name in args.datasets:
        if dataset_name not in DATASETS:
            log.warning(f"Dataset sconosciuto '{dataset_name}', saltato.")
            continue
        dataset_dir = DATASETS[dataset_name]
        noise_reg   = build_noise_registry(dataset_name)

        for noise_type in args.noise:
            if noise_type not in noise_reg:
                continue
            noise_cfg  = noise_reg[noise_type]
            level_map  = noise_cfg["levels"]

            levels_run = args.levels if args.levels else list(level_map.keys())
            if noise_type == 'clean':
                levels_run = ["0.0"]
            print(f"levels run: {levels_run}")
            for level in levels_run:

                if level not in level_map:
                    continue

                if noise_type in ["clean",'sequence_disruption','context_swap','sentiment_flip']:
                    merge_csvs(dataset_name, noise_type, str(level))


                for model in args.models:
                    print(f"model run: {model}")

                    runs.append({
                        "dataset_name": dataset_name,
                        "dataset_dir":  dataset_dir,
                        "model":        model,
                        "noise_type":   noise_type,
                        "level":        level,
                        "train_file":   level_map[level],
                        "train_subdir": noise_cfg["subdir"],
                    })
                    print(runs)

    total = len(runs)
    log.info(f"Esperimenti pianificati: {total}")

    if args.dry_run:
        print(f"\n{'Dataset':<20} {'Model':<12} {'Noise':<26} {'Level'}")
        print("-" * 66)
        for r in runs:
            print(
                f"{r['dataset_name']:<20} {r['model']:<12} "
                f"{r['noise_type']:<26} {r['level']}"
            )
        print(f"\nTotale: {total}\n")
        return

    done, errors = 0, 0
    for r in runs:
        ds, model, noise, level = (
            r["dataset_name"], r["model"], r["noise_type"], r["level"]
        )

        if args.skip_existing and checkpoint_exists(ds, noise, level, model):
            log.info(f"  → Skip: {ds}__{noise}__{level} / {model}")
            done += 1
            continue
       # merge_csvs(ds, noise, str(level))

        metrics = run_single(
            model_name     = model,
            dataset_name   = ds,
            dataset_dir    = r["dataset_dir"],
            noise_type     = noise,
            level          = level,
            train_filename = r["train_file"],
            train_subdir   = r["train_subdir"],
            staging_root   = staging_root,
            glove_path     = glove_path,
        )
        save_result(metrics)
        done += 1
        if metrics["status"] != "ok":
            errors += 1
        log.info(f"  Progresso: {done}/{total}  (errori: {errors})")

    log.info("=" * 72)
    log.info(f"Completato.  {done - errors}/{total} riusciti.")
    log.info(f"Risultati  → {RESULTS_FILE}")
    log.info(f"Checkpoint → {SAVED_DIR}/")
    log.info("=" * 72)



def merge_csvs(dataset,noise,portion):
    # Leggi i due file
    if noise == 'clean':
        os.makedirs(f'{BASE_DATA_DIR}/{dataset}/clean',exist_ok=True)
        df2 = pd.read_csv(f'{BASE_DATA_DIR}/{dataset}/train_{dataset}.csv',sep='\t')
        df2 = df2.dropna()
        df2.to_csv(f'{BASE_DATA_DIR}/{dataset}/clean/train_{dataset}.csv',sep='\t',index=False)
        return

    if portion.endswith("0"):
        portion = portion[:-1]
    df1 = pd.read_csv(f'{BASE_DATA_DIR}/{dataset}/{noise}/prompts_{portion}.output.csv',sep=',')
    df1.dropna(inplace=True)
    df1.to_csv(f'{BASE_DATA_DIR}/{dataset}/{noise}/injected_{portion}0_1.inter',sep='\t',index=False)
    df1 = pd.read_csv(f'{BASE_DATA_DIR}/{dataset}/{noise}/injected_{portion}0_1.inter',sep='\t')
    df1.rename(columns={'text:token': 'text:token_seq'}, inplace=True)
    df1 = df1[['user_id:token','item_id:token','rating:float','timestamp:float','text:token_seq']]
    df2 = pd.read_csv(f'{BASE_DATA_DIR}/{dataset}/train_{dataset}.csv',sep='\t')
    df2 = df2[['user_id:token','item_id:token','rating:float','timestamp:float','text:token_seq']]
    df2.dropna(inplace=True)

    # Concatena le righe
    df_concat = pd.concat([df2, df1], ignore_index=True)
    df = df_concat.drop_duplicates(subset=["user_id:token","item_id:token"], keep='last')
    prima = len(df)
    df_clean = df.dropna()
    dopo = len(df_clean)
    df = df.dropna(subset=["rating:float"])
    print(f"Righe totali:    {prima}")
    print(f"Righe eliminate: {prima - dopo}")
    print(f"Righe rimaste:   {dopo}")
    print(f"\nNull per colonna:\n{df.isnull().sum()}")
    # Salva come TSV (tab-separated)
    df.to_csv(
        f'{BASE_DATA_DIR}/{dataset}/{noise}/corrupted_{portion}0_1.inter',
        sep='\t',
        index=False
    )

    print(df_concat.shape)
    return


def drop_null():
    for d in ['yelp','Books',"Toys_and_Games"]:
        for p in ["0.10","0.20","0.30"]:
            df = pd.read_csv(f'{BASE_DATA_DIR}/{d}/random_burst_fake/corrupted_{p}.inter', sep='\t')
            print(df.isnull().sum())
            print(df[df['rating:float'].isnull()].head(10))
            df = df.dropna()
            df.to_csv(f'{BASE_DATA_DIR}/{d}/random_burst_fake/corrupted_{p}_1.inter',index=False,sep='\t')

            df = pd.read_csv(f'{BASE_DATA_DIR}/{d}/targeted_injection_fake/corrupted_{p}.inter', sep='\t')
            print(df.isnull().sum())
            print(df[df['rating:float'].isnull()].head(10))
            df = df.dropna()
            df.to_csv(f'{BASE_DATA_DIR}/{d}/targeted_injection_fake/corrupted_{p}_1.inter',index=False,sep='\t')

            df = pd.read_csv(f'{BASE_DATA_DIR}/{d}/rating_flip/corrupted_{p}.inter', sep='\t')
            print(df.isnull().sum())
            print(df[df['rating:float'].isnull()].head(10))
            df = df.dropna()
            df.to_csv(f'{BASE_DATA_DIR}/{d}/rating_flip/corrupted_{p}_1.inter',index=False,sep='\t')


import pandas as pd
if __name__ == "__main__":


    main()
    # df1 = pd.read_csv('../../data_review/Toys_and_Games/sentiment_flip/corrupted_0.10.csv')
    # print()
    # df2 = pd.read_csv('../../data_review/Toys_and_Games/sentiment_flip/prompts_0.1.csv')
    # print()
