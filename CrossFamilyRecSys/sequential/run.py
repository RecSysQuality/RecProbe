#!/usr/bin/env python3
"""
run_experiments.py
==================
Allena GRU4Rec, SASRec e BERT4Rec su tutte le varianti di rumore.

SPLIT STRATEGY
--------------
Un unico file .inter per dataset. RecBole usa leave-one-out:
  - test  = ultimo item per utente
  - valid = penultimo item per utente
  - train = tutto il resto

Il file contiene TRAIN + VALID + TEST concatenati (tutti i dati).
RecBole ordina per timestamp e fa lo split automaticamente.

GRUPPI UTENTI (active / lt)
----------------------------
I file test_<DS>_active.csv e test_<DS>_lt.csv contengono subset
di utenti. Dopo il training, le metriche per active/lt vengono
calcolate filtrando i risultati per quegli utenti.

BEYOND-ACCURACY
---------------
Coverage, Novelty, Gini, Diversity — calcolate sulle top-k liste.
"""

import argparse
import json
import logging
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


import os

try:
    from recbole.quick_start import run_recbole
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed
    import torch
    import torch.nn.functional as F
except ImportError:
    sys.exit("RecBole non installato. Esegui:  pip install recbole")

# PyTorch 2.6 ha cambiato il default di weights_only a True,
# incompatibile con i checkpoint salvati da RecBole 1.2.1.
# Monkey-patch per ripristinare il comportamento precedente.
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
#  PATHS — relativi allo script, funziona da qualsiasi working directory
# ==============================================================================
import os
try:
    method = os.environ.get("METHOD")
except Exception as e:
    method = "BERT4Rec"


BASE_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = BASE_DIR.parent.parent / "data"
CONFIGS_DIR   = BASE_DIR
#RESULTS_FILE  = BASE_DIR / "results" / f"results_summary_all.jsonl"
_ds_suffix = os.environ.get("DATASET_NAME", "all")
RESULTS_FILE  = BASE_DIR / "results" / f"results_summary_{_ds_suffix}.jsonl"
SAVED_DIR     = BASE_DIR / "saved"
STAGING_DIR   = BASE_DIR / "staging"

# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = {
   # "Toys_and_Games":     BASE_DATA_DIR / "Toys_and_Games",
    "yelp":           BASE_DATA_DIR / "yelp",
    "Books":          BASE_DATA_DIR / "Books",
    "Toys_and_Games": BASE_DATA_DIR / "Toys_and_Games",
}

ALL_MODELS = ["GRU4Rec", "SASRec", "BERT4Rec"]
ALL_MODELS = [ "BERT4Rec"]
#ALL_MODELS = [method]

# ==============================================================================
#  IPERPARAMETRI PER DATASET × MODELLO
# ==============================================================================
#
# LOGICA GENERALE
# ───────────────
# I parametri qui sotto vengono iniettati in parameter_dict dentro run_single()
# e sovrascrivono i valori nei file .yaml (RecBole applica config_dict per ultimo).
#
# DIMENSIONI DEI DATASET
#   All_Beauty / Toys_and_Games  → "piccoli"  (~43K utenti, ~750K rating, seq ~17)
#   yelp                         → "medio"    (~99K utenti, 2.76M rating, seq ~28)
#   Books                        → "grande"   (~139K utenti, 3.34M rating, seq ~24)
#
# PRINCIPI PER PARAMETRO
# ───────────────────────
# hidden_size / embedding_size
#   Più il catalogo è grande, più capacità serve.
#   Books (113K item) → 128; Toys/Beauty (14K item) → 64.
#   Yelp (56K item) → 128 per catturare la varietà (venue eterogenee).
#
# n_layers / num_layers
#   Dataset grandi e sequenze lunghe traggono vantaggio da più layer
#   perché ci sono più pattern a lungo raggio da imparare.
#   Dataset piccoli: 2 layer bastano e riducono l'overfitting.
#
# dropout (hidden_dropout_prob / attn_dropout_prob / dropout_prob)
#   Dataset piccoli → dropout più alto (0.5) per regularizzare di più.
#   Dataset grandi  → dropout leggermente ridotto (0.3–0.4) perché i dati
#   stessi regolarizzano; troppo dropout ostacola la convergenza.
#
# learning_rate
#   Standard 0.001 su tutti. Con hidden_size grande si può scendere a
#   0.0005 se la valid loss oscilla dopo le prime 20–30 epoche.
#
# mask_ratio (solo BERT4Rec)
#   Seq corte (Toys/Beauty, max_len=20): 0.15 → maschera ~3 item su 20.
#   Seq lunghe (Yelp/Books, max_len=50): 0.20 → maschera ~10 item su 50.
#   Mascherare troppo su sequenze corte impoverisce il segnale supervisorio.
#
# STRUTTURA
# ─────────
# MODEL_HPARAMS[dataset][model] = dict con i parametri da iniettare.
# Chiavi identiche ai nomi usati da RecBole (e dai yaml).
# Aggiungi un dataset o un modello semplicemente aggiungendo una chiave.

MODEL_HPARAMS: dict[str, dict[str, dict]] = {

    # ── Toys & Games ──────────────────────────────────────────────────────────
    # Dimensioni: utenti 43K, item 14K, rating 753K, seq media ~17
    # Catalogo piccolo → hidden_size ridotto basta; dropout alto per regularizzare
    "Toys_and_Games": {
        "GRU4Rec": {
            "epochs":           100,
            "eval_step":        5,
            "stopping_step":    10,
            "embedding_size":   64,
            "hidden_size":      128,
            "num_layers":       1,
            "dropout_prob":     0.5,
            "learning_rate":    0.001,
        },
        "SASRec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "hidden_size":          64,
            "inner_size":           256,
            "n_layers":             2,
            "n_heads":              2,
            "hidden_dropout_prob":  0.5,
            "attn_dropout_prob":    0.5,
            "learning_rate":        0.001,
        },
        "BERT4Rec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "hidden_size":          64,
            "inner_size":           256,
            "n_layers":             2,
            "n_heads":              2,
            "hidden_dropout_prob":  0.5,
            "attn_dropout_prob":    0.5,
            "mask_ratio":           0.2,   # seq corte → meno mascheramento
            "learning_rate":        0.001,
        },
    },


    # ── Yelp ──────────────────────────────────────────────────────────────────
    # Dimensioni: utenti 99K, item 56K, rating 2.76M, seq media ~28
    # Catalogo medio-grande e venue eterogenee → hidden_size 128, più capacità
    # Dati abbondanti → dropout più basso (0.3–0.4); 2 layer rimangono stabili
    "yelp": {
        "GRU4Rec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "embedding_size":   128,
            "hidden_size":      256,
            "num_layers":       2,
            "dropout_prob":     0.3,
            "learning_rate":    0.001,
        },

        "SASRec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "hidden_size":          128,
            "inner_size":           512,
            "n_layers":             2,
            "n_heads":              2,
            "hidden_dropout_prob":  0.4,
            "attn_dropout_prob":    0.4,
            "learning_rate":        0.001,
        },
        "BERT4Rec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "hidden_size":          128,
            "inner_size":           512,
            "n_layers":             2,
            "n_heads":              2,
            "hidden_dropout_prob":  0.4,
            "attn_dropout_prob":    0.4,
            "mask_ratio":           0.2,    # seq lunghe → mascheramento standard
            "learning_rate":        0.001,
        },
    },

    # ── Amazon Books ──────────────────────────────────────────────────────────
    # Dimensioni: utenti 139K, item 113K, rating 3.34M, seq media ~24
    # Catalogo molto grande (113K) → hidden_size 128 necessario per coprire
    # lo spazio degli item. Sparsità altissima (99.97%) → più layer
    # aiutano a catturare co-occorrenze indirette. Dropout ridotto.
    "Books": {
        "GRU4Rec": {
            "epochs": 100,
            "eval_step": 20,
            "stopping_step": 20,
            "embedding_size":   128,
            "hidden_size":      256,
            "num_layers":       2,
            "dropout_prob":     0.3,
            "learning_rate":    0.0005,
        },
        "SASRec": {
            "epochs": 100,
            "eval_step": 5,
            "stopping_step": 10,
            "hidden_size":          128,
            "inner_size":           512,
            "n_layers":             3,      # catalogo enorme → layer extra utile
            "n_heads":              2,
            "hidden_dropout_prob":  0.3,
            "attn_dropout_prob":    0.3,
            "learning_rate":        0.0005, # lr più basso con hidden grande
        },
        "BERT4Rec": {
            "epochs": 100,
            "eval_step": 20,
            "stopping_step": 20,
            "hidden_size":          128,
            "inner_size":           512,
            "n_layers":             3,      # catalogo enorme → layer extra utile
            "n_heads":              2,
            "hidden_dropout_prob":  0.3,
            "attn_dropout_prob":    0.3,
            "mask_ratio":           0.2,
            "learning_rate":        0.0005,
        },
    },
}


def get_model_hparams(dataset_name: str, model: str) -> dict:
    """
    Restituisce gli iperparametri per la coppia (dataset, model).
    Se il dataset non è nel registry usa un dizionario vuoto
    (i valori restano quelli del file .yaml).
    Logga sempre quali parametri vengono applicati.
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
            "uso i valori dal file .yaml"
        )
    return hparams


# ==============================================================================
#  NOISE REGISTRY
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
                "0.1": "corrupted_0.10.inter",
                "0.2": "corrupted_0.20.inter",
                "0.3": "corrupted_0.30.inter",
            },
        },
        "random_burst_fake": {
            "subdir": "random_burst_fake",
            "levels": {
                "0.1": "corrupted_0.10.inter",
                "0.2": "corrupted_0.20.inter",
                "0.3": "corrupted_0.30.inter",
            },
        },
        "targeted_injection_fake": {
            "subdir": "targeted_injection_fake",
            "levels": {
                "0.1": "corrupted_0.10.inter",
                "0.2": "corrupted_0.20.inter",
                "0.3": "corrupted_0.30.inter",
            },
        },
    }

ALL_NOISE_TYPES = ["clean", "sequence_disruption", "random_burst_fake",
                   "targeted_injection_fake"]
ALL_NOISE_TYPES = ["clean"]

# ==============================================================================
#  STAGING — unico file .inter, leave-one-out split
# ==============================================================================
import re

def _parse_inter(path: Path) -> list[tuple[str, str, str]]:
    with path.open(encoding="utf-8") as f:
        raw_header = f.readline().rstrip("\n")
        lines = [l.rstrip("\n") for l in f if l.strip()]

    # Rileva separatore: tab, spazi multipli, o virgola
    if "\t" in raw_header:
        sep = "\t"
    elif re.search(r"  +", raw_header):
        sep = None          # None → split() splitta su qualsiasi whitespace
    else:
        sep = ","

    def split_line(line):
        return line.split(sep) if sep else line.split()

    cols = [c.strip().split(":")[0].lower() for c in split_line(raw_header)]

    try:
        uid_i = cols.index("user_id")
        iid_i = cols.index("item_id")
        ts_i  = cols.index("timestamp")
    except ValueError as e:
        raise ValueError(f"Colonna mancante in {path.name}: {e}")

    rows = []
    for line in lines:
        parts = split_line(line)
        if len(parts) <= max(uid_i, iid_i, ts_i):
            continue
        rows.append((
            parts[uid_i].strip(),
            parts[iid_i].strip(),
            parts[ts_i].strip(),
        ))
    return rows

def stage_dataset(
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
) -> tuple[Path, str]:
    """
    Crea staging/<tag>/<tag>.inter con TUTTI i dati (train+vali+test).
    RecBole usa leave-one-out per splittare automaticamente.

    Il file vali e test nella cartella principale servono solo per
    identificare gli utenti active/lt — NON vengono usati per lo split.
    """
    tag   = f"{dataset_name}__{noise_type}__{level}"
    stage = staging_root / tag
    stage.mkdir(parents=True, exist_ok=True)

    train_src = dataset_dir / train_subdir / train_filename
    vali_src  = dataset_dir / f"vali_{dataset_name}.csv"
    test_src  = dataset_dir / f"test_{dataset_name}.csv"

    for src in [train_src, vali_src, test_src]:
        if not src.exists():
            raise FileNotFoundError(f"File mancante: {src}")

    dest = stage / f"{tag}.inter"
    # Rigenera se il training file è più recente
    if dest.exists() and train_src.stat().st_mtime <= dest.stat().st_mtime:
        return stage, tag

    # Concatena train + vali + test in un unico file senza dedup.
    # RecBole ordina per timestamp per utente e fa leave-one-out:
    #   test  = ultima interazione per utente
    #   valid = penultima
    #   train = tutto il resto
    all_rows: list[tuple[str, str, str]] = []
    for src in [train_src, vali_src, test_src]:
        all_rows.extend(_parse_inter(src))

    with dest.open("w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\ttimestamp:float\n")
        for uid, iid, ts in all_rows:
            f.write(f"{uid}\t{iid}\t{ts}\n")

    log.info(f"  Staged {len(all_rows)} interazioni → {dest.name}")
    return stage, tag


def load_user_set(path: Path) -> set[str]:
    """Legge un CSV e restituisce l'insieme degli user_id."""
    if not path.exists():
        return set()
    rows = _parse_inter(path)
    return {uid for uid, _, _ in rows}


# ==============================================================================
#  BEYOND-ACCURACY METRICS
# ==============================================================================
def compute_beyond_accuracy(
    topk_lists:           list[list[int]],
    item_embeddings:      np.ndarray | None,
    item_popularity:      dict[int, int],
    n_train_interactions: int,
    catalog_size:         int,
) -> dict:
    all_rec   = [iid for lst in topk_lists for iid in lst]
    unique_rec = set(all_rec)

    # Coverage
    coverage = len(unique_rec) / catalog_size if catalog_size > 0 else 0.0

    # Novelty
    nov_per_user = []
    for lst in topk_lists:
        if not lst:
            continue
        si = []
        for iid in lst:
            pop = item_popularity.get(iid, 1)
            p_i = pop / max(n_train_interactions, 1)
            si.append(-math.log2(p_i))
        nov_per_user.append(float(np.mean(si)))
    novelty = float(np.mean(nov_per_user)) if nov_per_user else 0.0

    # Gini
    counts = np.array(sorted(Counter(all_rec).values()), dtype=float)
    n = len(counts)
    if n > 1:
        counts /= counts.sum()
        counts.sort()
        idx  = np.arange(1, n + 1)
        gini = float((2 * np.sum(idx * counts) / n) - (n + 1) / n)
        gini = max(0.0, min(1.0, gini))
    else:
        gini = 0.0

    # Diversity
    diversity = None
    if item_embeddings is not None:
        emb = F.normalize(
            torch.tensor(item_embeddings, dtype=torch.float32), dim=-1
        )
        div_per_user = []
        for lst in topk_lists:
            if len(lst) < 2:
                continue
            idx  = [min(i, emb.shape[0] - 1) for i in lst]
            vecs = emb[idx]
            sim  = (vecs @ vecs.T).numpy()
            k    = len(idx)
            mask = np.triu(np.ones((k, k), dtype=bool), k=1)
            avg_sim = sim[mask].mean() if mask.any() else 1.0
            div_per_user.append(1.0 - float(avg_sim))
        diversity = float(np.mean(div_per_user)) if div_per_user else 0.0

    result = {
        "coverage": round(coverage, 4),
        "novelty":  round(novelty,  4),
        "gini":     round(gini,     4),
    }
    if diversity is not None:
        result["diversity"] = round(diversity, 4)
    return result


# ==============================================================================
#  ACCURACY METRICS MANUALE (per subset di utenti)
# ==============================================================================
def _compute_accuracy_metrics(
    scores: "torch.Tensor",   # (N, n_items) — già con padding=−inf
    pos_items: "torch.Tensor", # (N,) — internal item id del ground-truth
    ks: list[int] = [5, 10, 20],
) -> dict:
    """
    Calcola Recall@K, NDCG@K, MRR@K, Hit@K dato lo score di ogni item
    e il ground-truth item per ogni utente.
    """
    results = {}
    N = scores.shape[0]

    # Rank del pos_item per ogni utente (0-based: 0 = primo posto)
    # sorted in descending order
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)  # (N, n_items)
    pos_items_exp  = pos_items.unsqueeze(1)                           # (N, 1)
    rank_matrix    = (sorted_indices == pos_items_exp)                # (N, n_items) bool
    # rank 0-based position
    rank = rank_matrix.float().argmax(dim=1).float()                  # (N,)

    for k in ks:
        hit   = (rank < k).float()
        ndcg  = (hit / torch.log2(rank + 2)).mean().item()
        recall = hit.mean().item()
        mrr    = (1.0 / (rank + 1) * hit).mean().item()
        hit_k  = recall  # Hit@K = Recall@K per un solo item positivo

        results[f"recall@{k}"] = round(recall, 4)
        results[f"ndcg@{k}"]   = round(ndcg,   4)
        results[f"mrr@{k}"]    = round(mrr,     4)
        results[f"hit@{k}"]    = round(hit_k,   4)

    return results


# ==============================================================================
#  EVAL SU UN SOTTOINSIEME DI UTENTI
#  Dopo il training RecBole ha già calcolato i risultati su tutti gli utenti.
#  Qui ricarichiamo il modello e calcoliamo le metriche filtrando per
#  gli utenti active o long-tail.
# ==============================================================================
def eval_user_subset(
    checkpoint_path:  Path,
    model_name:       str,
    tag:              str,
    staging_root:     Path,
    cfg_files:        list,
    user_set:         set[str],
    subset_label:     str,
    topk:             int = 10,
) -> dict:
    """
    Ricarica il modello e valuta solo sugli utenti in user_set.
    """
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    parameter_dict = {
        "data_path": str(staging_root),
        "dataset": tag,
        "save_dataset": False,
        "show_progress": True,
        "stopping_step": 20,
        "eval_step": 20,
        "train_neg_sample_args": None,
        # ↑
        "eval_args": {
            "group_by": "user",
            "order": "TO",
            "mode": "full",
            "split": {"LS": "valid_and_test"},
        },
    }

    try:
        cfg = Config(
            model=model_name,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])

        dataset = create_dataset(cfg)
        print(dataset.inter_feat.columns)
        _, _, test_data = data_preparation(cfg, dataset)

        # Mappa token → internal id per filtrare
        uid_field   = dataset.uid_field
        token2id    = {
            dataset.id2token(uid_field, i): i
            for i in range(dataset.user_num)
        }
        valid_uids  = {
            token2id[u] for u in user_set if u in token2id
        }

        if not valid_uids:
            return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        checkpoint  = torch.load(checkpoint_path, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        model       = model_class(cfg_ckpt, dataset).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        trainer = get_trainer(cfg_ckpt["MODEL_TYPE"], cfg_ckpt["model"])(cfg_ckpt, model)

        # Filtra test_data per gli utenti del sottoinsieme.
        # NOTA: iterare un DataLoader RecBole produce tuple (Interaction, ...);
        # il primo elemento è sempre l'Interaction con i dati.
        from recbole.data.interaction import Interaction
        import torch as th

        # Raccoglie: top-k per beyond-accuracy, pos_item per accuracy manuale
        topk_lists      = []
        all_pos_items   = []   # item ground-truth per ogni utente del subset
        all_scores_list = []   # (scores_filtrati, pos_item_filtrati) per accuracy

        with torch.no_grad():
            for batch_tuple in test_data:
                # RecBole DataLoader restituisce (Interaction,) o (Interaction, None)
                batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                        else batch_tuple
                batch = batch.to(cfg["device"])

                uids = batch[uid_field].cpu().tolist()
                mask = torch.tensor(
                    [u in valid_uids for u in uids], dtype=torch.bool
                )
                if not mask.any():
                    continue

                scores = model.full_sort_predict(batch)
                scores[:, 0] = -float("inf")

                # Top-k filtrati per beyond-accuracy
                topk_ids          = torch.topk(scores, k=topk, dim=-1).indices
                topk_ids_filtered = topk_ids[mask]
                topk_lists.extend(topk_ids_filtered.cpu().tolist())

                # Scores e pos_item filtrati per accuracy manuale
                scores_filtered = scores[mask]
                # Il campo item di test è il positive item target
                pos_items = batch[dataset.iid_field][mask].cpu()
                all_scores_list.append(scores_filtered.cpu())
                all_pos_items.append(pos_items)

        if not topk_lists:
            return {"error": f"Nessun batch per {subset_label}"}

        # ── Calcolo manuale delle metriche di accuracy sul subset ──────────────
        # Raggruppa tutti gli score e i pos_item
        all_scores_t   = torch.cat(all_scores_list, dim=0)   # (N_subset, n_items)
        all_pos_items_t = torch.cat(all_pos_items, dim=0)    # (N_subset,)

        # Calcola rank del pos_item per ogni utente
        # (numero di item con score >= pos_item score, escludendo padding 0)
        acc_result = _compute_accuracy_metrics(
            all_scores_t, all_pos_items_t, ks=[10, 20, 50]
        )

        # Beyond-accuracy sul subset
        # Nota: inter_feat[field] è un Tensor PyTorch → usare .tolist() direttamente,
        # non .values.tolist() (.values su Tensor è un metodo built-in, non una property).
        iid_tensor    = dataset.inter_feat[dataset.iid_field]
        pop_counter   = Counter(iid_tensor.tolist())
        n_total_inter = len(dataset.inter_feat)
        catalog_size  = dataset.item_num

        item_emb = None
        try:
            item_emb = model.item_embedding.weight.detach().cpu().numpy()
        except AttributeError:
            pass

        ba = compute_beyond_accuracy(
            topk_lists            = topk_lists,
            item_embeddings       = item_emb,
            item_popularity       = pop_counter,
            n_train_interactions  = n_total_inter,
            catalog_size          = catalog_size,
        )
        log.info(
            f"    [{subset_label}] coverage={ba['coverage']:.4f}  "
            f"novelty={ba['novelty']:.4f}  gini={ba['gini']:.4f}"
        )
        combined = dict(acc_result)
        combined.update(ba)
        return combined

    except Exception as exc:
        log.error(f"  ✗ Eval [{subset_label}] fallita: {exc}")
        return {"error": str(exc)}


# ==============================================================================
#  SINGLE RUN
# ==============================================================================
def run_single(
    model: str,
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
    topk: int = 10,
) -> dict:

    log.info("=" * 72)
    log.info(f"  Model={model}  Dataset={dataset_name}  "
             f"Noise={noise_type}  Level={level}")
    log.info("=" * 72)

    # ── 1. Staging ────────────────────────────────────────────────────────────
    try:
        _, tag = stage_dataset(
            dataset_name, dataset_dir,
            noise_type, level,
            train_filename, train_subdir,
            staging_root,
        )
        staged_inter = staging_root / tag / f"{tag}.inter"
        print("=== FILE STAGED ===")
        with staged_inter.open() as f:
            for i, line in enumerate(f):
                print(repr(line))
                if i >= 3:
                    break
    except (FileNotFoundError, ValueError) as e:
        log.error(f"  ✗ {e}")
        return _err(model, dataset_name, noise_type, level, str(e), 0)

    run_saved_dir = SAVED_DIR / tag / model
    run_saved_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = str(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg   = str(CONFIGS_DIR / "models"   / f"{model}.yaml")
    cfg_files   = [f for f in [dataset_cfg, model_cfg] if Path(f).exists()]

    # Iperparametri base (infrastruttura RecBole, uguali per tutti)
    parameter_dict = {
        "data_path":             str(staging_root),
        "dataset":               tag,
        "checkpoint_dir":        str(run_saved_dir),
        "save_dataset":          False,
        "show_progress":         True,
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "stopping_step":         20,
        "train_neg_sample_args": None,
        "eval_step": 20,
        "eval_args": {
            "group_by": "user",
            "order":    "TO",
            "mode":     "full",
            "split":    {"LS": "valid_and_test"},
        },
    }

    # Iperparametri specifici per (dataset × modello): sovrascrivono il .yaml
    # perché config_dict è applicato per ultimo da RecBole.
    parameter_dict.update(get_model_hparams(dataset_name, model))

    # ── 2. Training — uso Config+trainer diretto invece di run_recbole ────────
    # run_recbole in 1.2.1 processa parameter_dict in ordine diverso
    # e droppa timestamp. Con Config diretto funziona correttamente.
    t0 = time.time()
    try:
        from recbole.trainer import Trainer

        cfg = Config(
            model=model,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])
        print("load_col:", cfg["load_col"])
        print("TIME_FIELD:", cfg["TIME_FIELD"])
        print("unused_col:", cfg["unused_col"])
        dataset                    = create_dataset(cfg)
        print(dataset.inter_feat.columns)
        train_data, valid_data, test_data = data_preparation(cfg, dataset)

        model_class   = get_model(cfg["model"])
        trained_model = model_class(cfg, dataset).to(cfg["device"])
        trainer       = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, trained_model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data,
            saved=True, show_progress=cfg["show_progress"],
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=True, show_progress=False
        )

        elapsed = time.time() - t0
        log.info(f"  ✓ Training completato in {elapsed:.1f}s  "
                 f"best_valid={best_valid_score:.4f}")

        result = {
            "best_valid_score": best_valid_score,
            "test_result":      test_result,
        }

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"  ✗ Training FALLITO dopo {elapsed:.1f}s — {exc}")
        return _err(model, dataset_name, noise_type, level, str(exc), elapsed)

    checkpoints = sorted(run_saved_dir.glob("*.pth"))
    best_ckpt   = checkpoints[-1] if checkpoints else None
    if best_ckpt is None:
        return _err(model, dataset_name, noise_type, level, "no checkpoint", elapsed)

    # ── 3. Eval su active e long-tail ─────────────────────────────────────────
    active_users = load_user_set(dataset_dir / f"test_{dataset_name}_active.csv")
    lt_users     = load_user_set(dataset_dir / f"test_{dataset_name}_lt.csv")

    log.info(f"  → Eval active users ({len(active_users)} utenti)")
    result_active = eval_user_subset(
        checkpoint_path = best_ckpt,
        model_name      = model,
        tag             = tag,
        staging_root    = staging_root,
        cfg_files       = cfg_files,
        user_set        = active_users,
        subset_label    = "active",
        topk            = topk,
    )

    log.info(f"  → Eval long-tail users ({len(lt_users)} utenti)")
    result_lt = eval_user_subset(
        checkpoint_path = best_ckpt,
        model_name      = model,
        tag             = tag,
        staging_root    = staging_root,
        cfg_files       = cfg_files,
        user_set        = lt_users,
        subset_label    = "lt",
        topk            = topk,
    )

    # ── 4. Beyond-accuracy sul test set standard ──────────────────────────────
    log.info("  → Beyond-accuracy sul test set standard")
    try:
        cfg = Config(
            model=model,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        print("load_col:", cfg["load_col"])
        print("TIME_FIELD:", cfg["TIME_FIELD"])
        print("unused_col:", cfg["unused_col"])

        init_seed(cfg["seed"], cfg["reproducibility"])
        dataset = create_dataset(cfg)
        _, _, test_data = data_preparation(cfg, dataset)

        checkpoint  = torch.load(best_ckpt, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        trained_model = model_class(cfg_ckpt, dataset).to(cfg["device"])
        trained_model.load_state_dict(checkpoint["state_dict"])
        trained_model.eval()

        topk_lists = []
        with torch.no_grad():
            for batch_tuple in test_data:
                batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                        else batch_tuple
                batch = batch.to(cfg["device"])
                scores = trained_model.full_sort_predict(batch)
                scores[:, 0] = -float("inf")
                topk_ids = torch.topk(scores, k=topk, dim=-1).indices
                topk_lists.extend(topk_ids.cpu().tolist())

        iid_tensor    = dataset.inter_feat[dataset.iid_field]
        pop_counter   = Counter(iid_tensor.tolist())
        n_total_inter = len(dataset.inter_feat)
        catalog_size  = dataset.item_num

        item_emb = None
        try:
            item_emb = trained_model.item_embedding.weight.detach().cpu().numpy()
        except AttributeError:
            pass

        ba_standard = compute_beyond_accuracy(
            topk_lists            = topk_lists,
            item_embeddings       = item_emb,
            item_popularity       = pop_counter,
            n_train_interactions  = n_total_inter,
            catalog_size          = catalog_size,
        )

        # Mergia beyond-accuracy nel test_result standard
        test_result_standard = dict(result.get("test_result", {}))
        test_result_standard.update(ba_standard)

    except Exception as exc:
        log.error(f"  ✗ Beyond-accuracy standard fallita: {exc}")
        test_result_standard = result.get("test_result", {})

    return {
        "model":              model,
        "dataset":            dataset_name,
        "noise_type":         noise_type,
        "level":              level,
        "status":             "ok",
        "elapsed_s":          round(elapsed, 1),
        "best_valid_score":   result.get("best_valid_score"),
        "checkpoint_dir":     str(run_saved_dir),
        "test_result":        test_result_standard,
        "test_result_active": result_active,
        "test_result_lt":     result_lt,
    }


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

def checkpoint_exists(dataset_name, noise_type, level, model):
    tag  = f"{dataset_name}__{noise_type}__{level}"
    ckpt = SAVED_DIR / tag / model
    return ckpt.exists() and any(ckpt.glob("*.pth"))


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="RecBole sequential — single-file LOO split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--models",        nargs="+", default=ALL_MODELS)
    p.add_argument("--noise",         nargs="+", default=ALL_NOISE_TYPES)
    p.add_argument("--levels",        nargs="+", default=None)
    p.add_argument("--topk",          type=int,  default=10)
    p.add_argument("--staging-dir",   default=str(STAGING_DIR))
    p.add_argument("--skip-existing", action="store_true")
    p.add_argument("--dry-run",       action="store_true")
    return p.parse_args()


def main():
    args         = parse_args()
    staging_root = Path(args.staging_dir)
    staging_root.mkdir(parents=True, exist_ok=True)

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

            for level in levels_run:
                if level not in level_map:
                    continue
                for model in args.models:
                    runs.append({
                        "dataset_name": dataset_name,
                        "dataset_dir":  dataset_dir,
                        "model":        model,
                        "noise_type":   noise_type,
                        "level":        level,
                        "train_file":   level_map[level],
                        "train_subdir": noise_cfg["subdir"],
                    })

    total = len(runs)
    log.info(f"Esperimenti pianificati: {total}")

    if args.dry_run:
        print(f"\n{'Dataset':<20} {'Model':<10} {'Noise':<22} {'Level'}")
        print("-" * 60)
        for r in runs:
            print(f"{r['dataset_name']:<20} {r['model']:<10} "
                  f"{r['noise_type']:<22} {r['level']}")
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

        metrics = run_single(
            model          = model,
            dataset_name   = ds,
            dataset_dir    = r["dataset_dir"],
            noise_type     = noise,
            level          = level,
            train_filename = r["train_file"],
            train_subdir   = r["train_subdir"],
            staging_root   = staging_root,
            topk           = args.topk,
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


if __name__ == "__main__":
    main()