#!/usr/bin/env python3
"""
run_cf.py
=========
Allena LightGCN, NGCF e GCMC su tutte le varianti di rumore.

DIFFERENZE RISPETTO A run_experiments.py (modelli sequenziali)
--------------------------------------------------------------
1. SPLIT STRATEGY — RS invece di LOO
   LightGCN, NGCF e GCMC sono modelli CF non sequenziali. Non usano
   l'ordinamento temporale per fare predizioni. Lo split è
   quindi Random Split (80/10/10) invece di Leave-One-Out.
   Il file .inter contiene TUTTI i dati; RecBole fa RS automaticamente.

2. LOSS / NEG SAMPLING — BPR (LightGCN/NGCF) o niente (GCMC)
   LightGCN e NGCF usano BPR con neg sampling esplicito.
   GCMC è rating prediction (regressione): niente neg sampling,
   niente full_sort_predict. La funzione is_rating_model() isola
   questa logica così run_single rimane un unico entry point.

3. FULL_SORT_PREDICT vs PREDICT
   LightGCN/NGCF: full_sort_predict (score su tutti gli item).
   GCMC: predict() su coppie (u, i) osservate → RMSE/MAE.
   Le beyond-accuracy e le metriche active/lt sono skippate per
   GCMC (non ha senso calcolare coverage/novelty su rating pred).

4. ITEM EMBEDDING
   LightGCN: model.item_embedding (nn.Embedding).
   NGCF:     model.embedding_dict['item_id'] (ParameterDict).
   GCMC:     model.encoder.W (matrice peso encoder) — non usato
             per diversity perché il task è diverso.
   La funzione get_item_emb() gestisce tutti i casi.

5. STAGING — rating incluso per GCMC
   Lo staging per GCMC emette anche la colonna rating:float.
   Lo staging per LightGCN/NGCF rimane invariato (solo u/i/ts).
   La funzione stage_dataset() accetta il flag include_rating.

6. AUC — calcolato in post-processing per GCMC
   RecBole non calcola AUC nativamente per regression task.
   Viene calcolato binarizzando i rating: ≥4 = positivo (1-5 stelle).
   sklearn.metrics.roc_auc_score viene usato su predicted vs true.
"""

import argparse
import json
import logging
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np

try:
    from sklearn.metrics import roc_auc_score
    _SKLEARN_OK = True
except ImportError:
    _SKLEARN_OK = False

try:
    from recbole.config import Config
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_model, get_trainer, init_seed
    import torch
    import torch.nn.functional as F
except ImportError:
    import sys
    sys.exit("RecBole non installato. Esegui:  pip install recbole")

# Compatibilità PyTorch ≥ 2.6
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
import os
try:
    method = os.environ.get("METHOD")
except Exception as e:
    print(e)
    method = "LightGCN"
print(f'METHOD: {method}')

BASE_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = BASE_DIR.parent.parent / "data"
CONFIGS_DIR   = BASE_DIR
#RESULTS_FILE  = BASE_DIR / "results" / f"results_summary_all.jsonl"
_ds_suffix = os.environ.get("DATASET_NAME", "all")
RESULTS_FILE  = BASE_DIR / "results" / f"results_summary_{_ds_suffix}.jsonl"


ROOT_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = (ROOT_DIR / "../../data").resolve()
CONFIGS_DIR   = ROOT_DIR



SAVED_DIR     = ROOT_DIR / "saved_cf"
STAGING_DIR   = ROOT_DIR / "staging_cf"

# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = {
    "yelp": BASE_DATA_DIR / "yelp",
    "Books": BASE_DATA_DIR / "Books",
    "Toys_and_Games": BASE_DATA_DIR / "Toys_and_Games",
}

ALL_MODELS = [method]
ALL_MODELS = ["NGCF","LightGCN","DGCF"]

# Modelli che fanno rating prediction invece di ranking top-k.
# Hanno pipeline diverso: niente neg sampling, niente full_sort_predict,
# metriche RMSE/MAE/AUC invece di Recall/NDCG.
RATING_MODELS = {}

def is_rating_model(model_name: str) -> bool:
    return model_name in RATING_MODELS


# ==============================================================================
#  IPERPARAMETRI PER DATASET × MODELLO
# ==============================================================================
#
# LOGICA GENERALE
# ───────────────
# I parametri qui vengono iniettati in parameter_dict dentro run_single()
# e sovrascrivono i valori nei file .yaml (RecBole applica config_dict per ultimo).
# Stessa struttura di MODEL_HPARAMS in run_experiments.py (modelli sequenziali).
#
# DIMENSIONI DEI DATASET
#   Toys_and_Games  → "piccolo"  (43K utenti, 14K item,  753K rating, sparsità 99.94%)
#   yelp            → "medio"    (99K utenti, 56K item,  2.76M rating, sparsità 99.95%)
#   Books           → "grande"   (139K utenti, 113K item, 3.34M rating, sparsità 99.97%)
#
# PRINCIPI PER PARAMETRO
# ───────────────────────
# embedding_size
#   Scala con la dimensione del catalogo.
#   Toys (14K item) → 64; Yelp (56K) → 128; Books (113K) → 128.
#   Allineato a hidden_size dei modelli sequenziali per confronto diretto
#   sulle metriche di diversity/coverage basate su embedding.
#
# n_layers (LightGCN) / hidden_size_list (NGCF)
#   LightGCN: 3 layer su dataset piccoli/medi (ottimale paper originale).
#             2 layer su Books (sparsità 99.97%): più layer causa over-smoothing
#             su nodi con pochissimi archi.
#   NGCF: [64,64,64] su Toys; [128,128,128] su Yelp/Books.
#         Con Toys la concatenazione produce vettore 64*4=256.
#         Con Yelp/Books: 128*4=512 — stessa scala di inner_size dei transformer.
#
# reg_weight
#   LightGCN: 1e-4 default; 1e-3 su Books per penalizzare embedding item rari
#             (113K item, sparsità 99.97% → molti item con pochissime interazioni).
#   NGCF: 1e-5 default (più parametri di LightGCN → reg più bassa);
#         1e-4 su Books per stesso ragionamento.
#
# node_dropout (NGCF)
#   0.1 su Toys/Yelp: dropout moderato standard paper originale.
#   0.0 su Books: sparsità altissima, i nodi rari non possono permettersi
#   di essere droppati ulteriormente.
#
# dropout_prob (GCMC)
#   0.7 su Toys/Yelp: dropout alto per dataset sparsi (paper originale).
#   0.5 su Books: sparsità 99.97% → segnale già scarso, dropout troppo
#   alto taglia via informazione preziosa sulle poche coppie osservate.
#
# learning_rate
#   0.001 su Toys (catalogo piccolo, embedding piccoli, gradienti stabili).
#   0.001 su Yelp (più dati per item rispetto a Books, gradienti meno rumorosi).
#   0.0005 su Books (113K item rari → gradienti noisier, lr più basso stabilizza).
#   Coerente con il ragionamento dei modelli sequenziali in run_experiments.py.
#
# epochs
#   Toys: 200 — dataset piccolo, converge in fretta con early stopping a 20.
#   Yelp: 200 — abbondanza di dati compensa la velocità di convergenza.
#   Books: 300 — sparsità altissima, segnale debole per item rari,
#               serve più tempo per convergere (allineato al config sequenziale Books).

MODEL_HPARAMS_CF: dict[str, dict[str, dict]] = {

    # ── Toys & Games ──────────────────────────────────────────────────────────
    # Catalogo piccolo (14K item), sparsità 99.94%, seq ~17
    # embedding piccoli bastano; dropout alto per regularizzare
    "Toys_and_Games": {
        "LightGCN": {
            "embedding_size": 64,
            "n_layers":       2,
            "reg_weight":     1.0e-4,
            "learning_rate":  0.001,
            "epochs":         100,
        },
        "NGCF": {
            "embedding_size":   64,
            "hidden_size_list": [64, 64],
            "node_dropout":     0.1,
            "message_dropout":  0.1,
            "reg_weight":       1.0e-5,
            "learning_rate":    0.001,
            "epochs":           100,
        },
        "DGCF": {
            "embedding_size": 64,
            "n_factors": 2,  # 64/2=32 dim per intent, ridotto da 4
            "n_iterations": 1,  # ridotto da 2 → ~2x speedup
            "n_layers": 1,
            "reg_weight": 1.0e-3,
            "cor_weight": 0.001,  # ridotto da 0.01 → meno overhead
            "learning_rate": 0.001,
            "epochs": 50,  # con early stopping basta
            "train_batch_size": 4096,  # da 2048 → meno step per epoca
            "eval_step": 5,
            "stopping_step": 10,
            "worker": 4,
        },
        "GCMC": {
            "embedding_size": 64,
            "dropout_prob":   0.7,
            "learning_rate":  0.001,
            "epochs":         100,
        },
    },

    # ── Yelp ──────────────────────────────────────────────────────────────────
    # Catalogo medio (56K item), sparsità 99.95%, seq ~28
    # embedding più grandi per coprire lo spazio; dropout moderato
    "yelp": {
        "LightGCN": {
            "embedding_size": 128,
            "n_layers":       2,
            "reg_weight":     1.0e-4,
            "learning_rate":  0.001,
            "epochs":         100,
        },
        "NGCF": {
            "embedding_size":   64,
            "hidden_size_list": [64, 64],
            "node_dropout":     0.1,
            "message_dropout":  0.1,
            "reg_weight":       1.0e-5,
            "learning_rate":    0.001,
            "epochs":           100,
        },
        "DGCF": {
            "embedding_size": 64,  # ridotto da 128: meno parametri, meno memoria
            "n_factors": 2,  # ridotto da 8 → 8x meno intent pairs nel cor_weight
            "n_iterations": 1,  # ridotto da 2
            "n_layers": 1,
            "reg_weight": 1.0e-3,
            "cor_weight": 0.001,  # ridotto da 0.01
            "learning_rate": 0.001,
            "epochs": 50,
            "train_batch_size": 4096,
            "eval_step": 5,
            "stopping_step": 10,
            "worker": 4,
        },
        "GCMC": {
            "embedding_size": 128,
            "dropout_prob":   0.7,
            "learning_rate":  0.001,
            "epochs":         100,
        },
    },

    # ── Amazon Books ──────────────────────────────────────────────────────────
    # Catalogo enorme (113K item), sparsità 99.97%, seq ~24
    # lr più basso (gradienti noisier su item rari), reg più alta,
    # n_layers ridotto (over-smoothing su nodi rari), dropout GCMC ridotto
    "Books": {
        "LightGCN": {
            "embedding_size": 128,
            "n_layers":       2,        # 3 → over-smoothing su sparsità 99.97%
            "reg_weight":     1.0e-3,   # più alta: penalizza embedding item rari
            "learning_rate":  0.0005,   # gradienti noisier su 113K item
            "epochs":         100,      # convergenza lenta su dataset sparso
        },
        "NGCF": {
            "embedding_size":   64,
            "hidden_size_list": [64, 64],
            "node_dropout":     0.0,    # nodi rari non possono essere droppati
            "message_dropout":  0.1,
            "reg_weight":       1.0e-4, # più alta rispetto al default 1e-5
            "learning_rate":    0.0005,
            "epochs":           100,
        },
        "GCMC": {
            "embedding_size": 128,
            "dropout_prob":   0.5,      # ridotto: segnale scarso su coppie rare
            "learning_rate":  0.0005,
            "epochs":         100,
        },
        "DGCF": {
            "embedding_size": 64,  # ridotto da 128: Books ha 113K item, risparmio memoria
            "n_factors": 2,  # ridotto da 4: sparsità 99.97%, pochi intent utili
            "n_iterations": 1,  # ridotto da 2
            "n_layers": 1,
            "reg_weight": 1.0e-2,
            "cor_weight": 0.0005,  # ridotto da 0.005: segnale già scarso su Books
            "learning_rate": 0.0005,
            "epochs": 50,
            "train_batch_size": 4096,
            "eval_step": 5,
            "stopping_step": 10,
            "worker": 4,
        },
    },
}


def get_model_hparams(dataset_name: str, model: str) -> dict:
    """
    Restituisce gli iperparametri CF per la coppia (dataset, model).
    Se il dataset non è nel registry usa un dizionario vuoto
    (i valori restano quelli del file .yaml).
    Logga sempre quali parametri vengono applicati.
    """
    hparams = MODEL_HPARAMS_CF.get(dataset_name, {}).get(model, {})
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
#  NOISE REGISTRY — identico a run_experiments.py
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



# ==============================================================================
#  STAGING
# ==============================================================================
import re

def _parse_inter(
    path: Path,
    want_rating: bool = False,
) -> list[tuple]:
    """
    Parsa un file .inter e restituisce righe come tuple.
    Se want_rating=True restituisce (uid, iid, rating, ts).
    Altrimenti restituisce (uid, iid, ts).
    Se want_rating=True ma la colonna rating manca, lancia ValueError.
    """
    with path.open(encoding="utf-8") as f:
        raw_header = f.readline().rstrip("\n")
        lines = [l.rstrip("\n") for l in f if l.strip()]

    if "\t" in raw_header:
        sep = "\t"
    elif re.search(r"  +", raw_header):
        sep = None
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

    rat_i = None
    if want_rating:
        # Accetta "rating" o "rate" come nome colonna
        for cand in ("rating", "rate", "ratings"):
            if cand in cols:
                rat_i = cols.index(cand)
                break
        if rat_i is None:
            raise ValueError(
                f"Colonna rating non trovata in {path.name}. "
                f"Colonne disponibili: {cols}. "
                "GCMC richiede rating espliciti nel file .inter."
            )

    rows = []
    for line in lines:
        parts = split_line(line)
        needed = max(uid_i, iid_i, ts_i) if rat_i is None else max(uid_i, iid_i, ts_i, rat_i)
        if len(parts) <= needed:
            continue
        if want_rating:
            rows.append((
                parts[uid_i].strip(),
                parts[iid_i].strip(),
                parts[rat_i].strip(),
                parts[ts_i].strip(),
            ))
        else:
            rows.append((parts[uid_i].strip(), parts[iid_i].strip(), parts[ts_i].strip()))
    return rows


def stage_dataset(
    dataset_name: str,
    dataset_dir: Path,
    noise_type: str,
    level: str,
    train_filename: str,
    train_subdir: str,
    staging_root: Path,
    include_rating: bool = False,
) -> tuple[Path, str]:
    """
    Crea staging_cf/<tag>/<tag>.inter con train+vali+test.
    RecBole esegue RS (80/10/10) automaticamente.

    Se include_rating=True (per GCMC), aggiunge la colonna
    rating:float all'header e alle righe.
    Il file sorgente deve contenere una colonna 'rating'.
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

    # Usa tag diverso per non sovrascrivere lo staging senza rating
    if include_rating:
        tag_staged = f"{tag}__rated"
    else:
        tag_staged = tag

    stage_r = staging_root / tag_staged
    stage_r.mkdir(parents=True, exist_ok=True)
    dest = stage_r / f"{tag_staged}.inter"

    if dest.exists() and train_src.stat().st_mtime <= dest.stat().st_mtime:
        return stage_r, tag_staged

    all_rows = []
    for src in [train_src, vali_src, test_src]:
        all_rows.extend(_parse_inter(src, want_rating=include_rating))

    with dest.open("w", encoding="utf-8") as f:
        if include_rating:
            f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
            for uid, iid, rat, ts in all_rows:
                f.write(f"{uid}\t{iid}\t{rat}\t{ts}\n")
        else:
            f.write("user_id:token\titem_id:token\ttimestamp:float\n")
            for uid, iid, ts in all_rows:
                f.write(f"{uid}\t{iid}\t{ts}\n")

    log.info(f"  Staged {len(all_rows)} interazioni → {dest.name}")
    return stage_r, tag_staged


def load_user_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = _parse_inter(path, want_rating=False)
    return {row[0] for row in rows}


# ==============================================================================
#  UTILITY — recupera embedding item in modo robusto
#  LightGCN: model.item_embedding  (nn.Embedding)
#  NGCF:     model.embedding_dict['item_id']  (nn.Parameter)
#  GCMC:     N/A — task rating prediction, diversity non calcolata
# ==============================================================================
def get_item_emb(trained_model) -> np.ndarray | None:
    """
    Restituisce i pesi degli embedding item come numpy array.
    Gestisce LightGCN (item_embedding), NGCF (embedding_dict).
    Per GCMC restituisce None: il task è rating prediction,
    calcolare diversity sulle predizioni non ha senso.
    """
    # GCMC: cortocircuito esplicito
    if trained_model.__class__.__name__ == "GCMC":
        return None

    # LightGCN
    if hasattr(trained_model, "item_embedding"):
        emb = trained_model.item_embedding
        if hasattr(emb, "weight"):
            return emb.weight.detach().cpu().numpy()
        if isinstance(emb, torch.Tensor):
            return emb.detach().cpu().numpy()

    # NGCF
    if hasattr(trained_model, "embedding_dict"):
        ed = trained_model.embedding_dict
        if "item_id" in ed:
            return ed["item_id"].detach().cpu().numpy()

    # Fallback generico
    for attr_name in dir(trained_model):
        if "item" in attr_name.lower() and "emb" in attr_name.lower():
            try:
                attr = getattr(trained_model, attr_name)
                if hasattr(attr, "weight"):
                    return attr.weight.detach().cpu().numpy()
                if isinstance(attr, torch.Tensor):
                    return attr.detach().cpu().numpy()
            except Exception:
                pass

    log.warning("  ✗ Item embedding non trovato — diversity non calcolata")
    return None


# ==============================================================================
#  BEYOND-ACCURACY METRICS — identiche a run_experiments.py
# ==============================================================================
def compute_beyond_accuracy(
    topk_lists:           list[list[int]],
    item_embeddings:      np.ndarray | None,
    item_popularity:      dict[int, int],
    n_train_interactions: int,
    catalog_size:         int,
) -> dict:
    all_rec    = [iid for lst in topk_lists for iid in lst]
    unique_rec = set(all_rec)
    coverage   = len(unique_rec) / catalog_size if catalog_size > 0 else 0.0

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
#  GCMC — METRICHE RATING PREDICTION
#  RecBole calcola RMSE e MAE nativamente per GCMC via eval_args labeled.
#  AUC non è nativa per regression task: la calcoliamo in post-processing
#  binarizzando i rating con soglia POS_THRESHOLD (≥ soglia → positivo).
# ==============================================================================
POS_THRESHOLD = 4.0   # Rating ≥ 4 su scala 1-5 = interazione positiva


def compute_auc_from_ratings(
    true_ratings: np.ndarray,
    pred_ratings: np.ndarray,
) -> float | None:
    """
    Calcola AUC binarizzando i rating con soglia POS_THRESHOLD.
    Restituisce None se sklearn non è disponibile o se una classe
    è assente (tutti positivi o tutti negativi nel subset).
    """
    if not _SKLEARN_OK:
        log.warning("  sklearn non disponibile — AUC non calcolato. "
                    "Installa con: pip install scikit-learn")
        return None
    y_true = (true_ratings >= POS_THRESHOLD).astype(int)
    if len(np.unique(y_true)) < 2:
        log.warning("  AUC non calcolabile: una sola classe nel subset "
                    f"(threshold={POS_THRESHOLD}, "
                    f"positivi={y_true.sum()}/{len(y_true)})")
        return None
    try:
        return float(roc_auc_score(y_true, pred_ratings))
    except Exception as exc:
        log.warning(f"  AUC fallito: {exc}")
        return None


def eval_gcmc_subset(
    checkpoint_path: Path,
    model_name: str,
    tag: str,
    staging_root: Path,
    cfg_files: list,
    user_set: set[str],
    subset_label: str,
) -> dict:
    """
    Eval GCMC su un sottoinsieme di utenti (active o lt).
    Calcola RMSE, MAE e AUC manualmente filtrando per utente.
    Non usa full_sort_predict: itera sulle coppie (u,i) del test set.
    """
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    parameter_dict = {
        "data_path":    str(staging_root),
        "dataset":      tag,
        "save_dataset": False,
        "show_progress": False,
        "load_col":     {"inter": ["user_id", "item_id", "rating", "timestamp"]},
        # GCMC: niente neg sampling
        "train_neg_sample_args": None,
        "eval_args": {
            "group_by": "user",
            "order":    "TO",
            "mode":     "full",
            "split":    {"LS": "valid_and_test"},
        },
        "metrics":      ["RMSE", "MAE"],
        "valid_metric": "RMSE",
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
        _, _, test_data = data_preparation(cfg, dataset)

        uid_field = dataset.uid_field
        token2id  = {
            dataset.id2token(uid_field, i): i
            for i in range(dataset.user_num)
        }
        valid_uids = {token2id[u] for u in user_set if u in token2id}
        if not valid_uids:
            return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        checkpoint  = torch.load(checkpoint_path, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        model       = model_class(cfg_ckpt, dataset).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        all_preds  = []
        all_truths = []
        rating_field = dataset.label_field  # "rating" per context-aware models

        with torch.no_grad():
            for batch_tuple in test_data:
                batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                        else batch_tuple
                batch = batch.to(cfg["device"])

                uids = batch[uid_field].cpu().tolist()
                mask = torch.tensor(
                    [u in valid_uids for u in uids], dtype=torch.bool
                )
                if not mask.any():
                    continue

                # GCMC.predict restituisce i rating predetti scalari
                preds  = model.predict(batch).cpu()
                truths = batch[rating_field].cpu()

                all_preds.extend(preds[mask].tolist())
                all_truths.extend(truths[mask].tolist())

        if not all_preds:
            return {"error": f"Nessuna predizione per {subset_label}"}

        preds_arr  = np.array(all_preds,  dtype=float)
        truths_arr = np.array(all_truths, dtype=float)

        rmse = float(np.sqrt(np.mean((preds_arr - truths_arr) ** 2)))
        mae  = float(np.mean(np.abs(preds_arr - truths_arr)))
        auc  = compute_auc_from_ratings(truths_arr, preds_arr)

        log.info(
            f"    [{subset_label}] RMSE={rmse:.4f}  MAE={mae:.4f}"
            + (f"  AUC={auc:.4f}" if auc is not None else "  AUC=N/A")
        )
        result = {"rmse": round(rmse, 4), "mae": round(mae, 4)}
        if auc is not None:
            result["auc"] = round(auc, 4)
        return result

    except Exception as exc:
        log.error(f"  ✗ Eval GCMC [{subset_label}] fallita: {exc}")
        return {"error": str(exc)}


# ==============================================================================
#  ACCURACY METRICS MANUALE (per subset di utenti)
# ==============================================================================
def _compute_accuracy_metrics(
    scores: "torch.Tensor",
    pos_items: "torch.Tensor",
    ks: list[int] = [50, 10, 20],
) -> dict:
    results = {}
    sorted_indices = torch.argsort(scores, dim=-1, descending=True)
    pos_items_exp  = pos_items.unsqueeze(1)
    rank_matrix    = (sorted_indices == pos_items_exp)
    rank           = rank_matrix.float().argmax(dim=1).float()

    for k in ks:
        hit    = (rank < k).float()
        ndcg   = (hit / torch.log2(rank + 2)).mean().item()
        recall = hit.mean().item()
        mrr    = (1.0 / (rank + 1) * hit).mean().item()

        results[f"recall@{k}"] = round(recall, 4)
        results[f"ndcg@{k}"]   = round(ndcg,   4)
        results[f"mrr@{k}"]    = round(mrr,     4)
        results[f"hit@{k}"]    = round(recall,  4)

    return results


# ==============================================================================
#  HELPER: mappa uid interno → test item
# ==============================================================================
def _build_uid_to_test_item(dataset) -> dict[int, int]:
    """
    Costruisce la mappa uid_interno → item_id del test item per ogni utente.

    Con eval_args mode=full (RS split) RecBole passa al modello solo user_id
    nel batch — il positive item non è mai presente nel batch stesso.
    Va estratto direttamente dalle interazioni del dataset.

    Con RS split ogni utente ha esattamente 1 test item (10% casuale).
    Non potendo identificarlo con certezza assoluta dall'esterno,
    usiamo l'item con indice/timestamp massimo come proxy, che è lo stesso
    criterio usato internamente da RecBole quando split=RS con ordine RO.
    In pratica, per RS puro l'ordine non conta — l'importante è avere
    UN item di riferimento per utente, e questo è coerente col trainer.
    """
    inter  = dataset.inter_feat
    uid_col = inter[dataset.uid_field]
    iid_col = inter[dataset.iid_field]

    uid_np = uid_col.numpy() if hasattr(uid_col, "numpy") else np.array(uid_col.tolist())
    iid_np = iid_col.numpy() if hasattr(iid_col, "numpy") else np.array(iid_col.tolist())

    # Cerca campo timestamp per ordinamento stabile
    ts_field = None
    for f in inter.interaction.keys():
        if "time" in f.lower() or "stamp" in f.lower():
            ts_field = f
            break

    if ts_field is not None:
        ts_col = inter[ts_field]
        ts_np  = ts_col.numpy() if hasattr(ts_col, "numpy") else np.array(ts_col.tolist())
        order  = np.lexsort((np.arange(len(uid_np)), ts_np, uid_np))
    else:
        order = np.arange(len(uid_np))

    uid_to_pos: dict[int, int] = {}
    for i in order:
        uid_to_pos[int(uid_np[i])] = int(iid_np[i])

    return uid_to_pos


# ==============================================================================
#  EVAL SU SOTTOINSIEME DI UTENTI
# ==============================================================================
def eval_user_subset(
    checkpoint_path: Path,
    model_name: str,
    tag: str,
    staging_root: Path,
    cfg_files: list,
    user_set: set[str],
    subset_label: str,
    topk: int = 10,
) -> dict:
    """
    Eval LightGCN/NGCF su un sottoinsieme di utenti (active o lt).
    Calcola Recall/NDCG/MRR/Hit@K + beyond-accuracy.

    FIX: con eval_args mode=full + RS split RecBole passa solo user_id
    nel batch; leggere batch[iid_field] fallisce silenziosamente ogni volta.
    Il positive item viene ora estratto dalla mappa uid→test_item costruita
    una volta sola dal dataset prima del loop sui batch.
    """
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    parameter_dict = {
        "data_path":   str(staging_root),
        "dataset":     tag,
        "save_dataset": False,
        "show_progress": True,
        # BPR: neg sampling necessario anche in eval
        "train_neg_sample_args": {
            "distribution": "uniform",
            "sample_num": 1,
            "dynamic": False,
        },
        "eval_args": {
            "group_by": "user",
            "order":    "TO",
            "mode":     "full",
            "split":    {"LS": "valid_and_test"},
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
        _, _, test_data = data_preparation(cfg, dataset)

        uid_field = dataset.uid_field
        token2id  = {
            dataset.id2token(uid_field, i): i
            for i in range(dataset.user_num)
        }
        valid_uids = {token2id[u] for u in user_set if u in token2id}

        if not valid_uids:
            return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

        # ── Costruisci mappa uid → test item dal dataset (fix principale) ──────
        # Il batch full-sort contiene solo user_id; il ground-truth va letto
        # direttamente dalle interazioni del dataset.
        uid_to_pos = _build_uid_to_test_item(dataset)
        n_mapped   = sum(1 for u in valid_uids if u in uid_to_pos)
        log.info(
            f"  [{subset_label}] ground-truth mappati: "
            f"{n_mapped}/{len(valid_uids)} utenti del subset"
        )

        checkpoint  = torch.load(checkpoint_path, map_location=cfg["device"])
        cfg_ckpt    = checkpoint["config"]
        model_class = get_model(cfg_ckpt["model"])
        model       = model_class(cfg_ckpt, dataset).to(cfg["device"])
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()

        topk_lists      = []
        all_scores_list = []
        all_pos_items   = []

        with torch.no_grad():
            for batch_tuple in test_data:
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

                # LightGCN/NGCF con RS split possono restituire
                # full_sort_predict come 1D (n_items,) quando il batch
                # ha un solo utente, o come 2D (batch, n_items) altrimenti.
                # Normalizziamo sempre a 2D prima di procedere.
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)

                # Se batch size e scores non coincidono, skip
                n_batch = len(uids)
                if scores.shape[0] != n_batch:
                    log.warning(
                        f"  scores.shape={scores.shape} != batch_size={n_batch}, skip batch"
                    )
                    continue

                scores[:, 0] = -float("inf")

                topk_ids          = torch.topk(scores, k=topk, dim=-1).indices
                topk_ids_filtered = topk_ids[mask]
                topk_lists.extend(topk_ids_filtered.cpu().tolist())

                # Recupera il positive item dalla mappa pre-costruita
                uids_masked    = [uids[i] for i, m in enumerate(mask.tolist()) if m]
                pos_items_vals = [uid_to_pos.get(u) for u in uids_masked]

                if all(p is not None for p in pos_items_vals):
                    pos_t = torch.tensor(pos_items_vals, dtype=torch.long)
                    all_scores_list.append(scores[mask].cpu())
                    all_pos_items.append(pos_t)
                else:
                    n_missing = sum(1 for p in pos_items_vals if p is None)
                    log.warning(
                        f"  [{subset_label}] {n_missing}/{len(uids_masked)} utenti "
                        f"senza ground-truth in questo batch — accuracy skip per il batch"
                    )

        if not topk_lists:
            return {"error": f"Nessun batch per {subset_label}"}

        acc_result = {}
        if all_scores_list and all_pos_items:
            all_scores_t    = torch.cat(all_scores_list, dim=0)
            all_pos_items_t = torch.cat(all_pos_items, dim=0)
            acc_result = _compute_accuracy_metrics(all_scores_t, all_pos_items_t, ks=[50, 10, 20])
            log.info(
                f"  [{subset_label}] accuracy calcolata su "
                f"{all_scores_t.shape[0]} utenti"
            )
        else:
            log.warning(
                f"  [{subset_label}] accuracy metrics non calcolate: "
                "nessun batch con ground-truth completo"
            )

        iid_tensor    = dataset.inter_feat[dataset.iid_field]
        pop_counter   = Counter(iid_tensor.tolist())
        n_total_inter = len(dataset.inter_feat)
        catalog_size  = dataset.item_num
        item_emb      = get_item_emb(model)

        ba = compute_beyond_accuracy(
            topk_lists           = topk_lists,
            item_embeddings      = item_emb,
            item_popularity      = pop_counter,
            n_train_interactions = n_total_inter,
            catalog_size         = catalog_size,
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

    rating_task = is_rating_model(model)

    # ── 1. Staging ────────────────────────────────────────────────────────────
    # GCMC ha bisogno del campo rating nel file .inter.
    # LightGCN/NGCF usano solo user_id/item_id/timestamp.
    try:
        _, tag = stage_dataset(
            dataset_name, dataset_dir,
            noise_type, level,
            train_filename, train_subdir,
            staging_root,
            include_rating=rating_task,
        )
    except (FileNotFoundError, ValueError) as e:
        log.error(f"  ✗ {e}")
        return _err(model, dataset_name, noise_type, level, str(e), 0)

    run_saved_dir = SAVED_DIR / tag / model
    run_saved_dir.mkdir(parents=True, exist_ok=True)

    dataset_cfg = str(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg   = str(CONFIGS_DIR / "models"   / f"{model}.yaml")
    cfg_files   = [f for f in [dataset_cfg, model_cfg] if Path(f).exists()]

    # ── parameter_dict: biforcazione GCMC vs LightGCN/NGCF ───────────────────
    if rating_task:
        # GCMC: regressione, niente neg sampling, mode=labeled, valid_metric=RMSE
        parameter_dict = {
            "data_path":      str(staging_root),
            "dataset":        tag,
            "checkpoint_dir": str(run_saved_dir),
            "save_dataset":   False,
            "show_progress":  True,
            "load_col":       {"inter": ["user_id", "item_id", "rating", "timestamp"]},
            "train_neg_sample_args": None,
            "eval_args": {
                "group_by": "user",
                "order": "TO",
                "mode": "full",
                "split": {"LS": "valid_and_test"},
            },
            "metrics":      ["RMSE", "MAE"],
            "valid_metric": "RMSE",     # minimizzare (RecBole gestisce lower-is-better)
        }
    else:
        # LightGCN / NGCF: ranking, BPR, mode=full
        parameter_dict = {
            "data_path":      str(staging_root),
            "dataset":        tag,
            "checkpoint_dir": str(run_saved_dir),
            "save_dataset":   False,
            "show_progress":  True,
            "load_col":       {"inter": ["user_id", "item_id", "timestamp"]},


            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num":   1,
                "dynamic":      False,
            },
            "eval_args": {
                "group_by": "user",
                "order": "TO",
                "mode": "full",
                "split": {"LS": "valid_and_test"},
            },
        }

    # Iperparametri specifici per (dataset × modello): sovrascrivono il .yaml
    # perché config_dict è applicato per ultimo da RecBole.
    # Stesso meccanismo di run_experiments.py per i modelli sequenziali.
    parameter_dict.update(get_model_hparams(dataset_name, model))

    # ── 2. Training ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        cfg = Config(
            model=model,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])
        log.info(f"  load_col: {cfg['load_col']}")

        dataset = create_dataset(cfg)
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

    active_users = load_user_set(dataset_dir / f"test_{dataset_name}_active.csv")
    lt_users     = load_user_set(dataset_dir / f"test_{dataset_name}_lt.csv")

    # ── 3a. GCMC: eval subset con RMSE/MAE/AUC, niente beyond-accuracy ────────
    if rating_task:
        log.info(f"  → Eval active users GCMC ({len(active_users)} utenti)")
        result_active = eval_gcmc_subset(
            checkpoint_path = best_ckpt,
            model_name      = model,
            tag             = tag,
            staging_root    = staging_root,
            cfg_files       = cfg_files,
            user_set        = active_users,
            subset_label    = "active",
        )

        log.info(f"  → Eval long-tail users GCMC ({len(lt_users)} utenti)")
        result_lt = eval_gcmc_subset(
            checkpoint_path = best_ckpt,
            model_name      = model,
            tag             = tag,
            staging_root    = staging_root,
            cfg_files       = cfg_files,
            user_set        = lt_users,
            subset_label    = "lt",
        )

        # AUC sul test set standard (RecBole non lo calcola per regression)
        log.info("  → AUC sul test set standard")
        test_result_standard = dict(result.get("test_result", {}))
        try:
            cfg2 = Config(
                model=model, dataset=tag,
                config_file_list=cfg_files, config_dict=parameter_dict,
            )
            init_seed(cfg2["seed"], cfg2["reproducibility"])
            dataset2 = create_dataset(cfg2)
            _, _, test_data2 = data_preparation(cfg2, dataset2)

            ckpt2    = torch.load(best_ckpt, map_location=cfg2["device"])
            m2_class = get_model(ckpt2["config"]["model"])
            m2       = m2_class(ckpt2["config"], dataset2).to(cfg2["device"])
            m2.load_state_dict(ckpt2["state_dict"])
            m2.eval()

            all_preds, all_truths = [], []
            rating_field = dataset2.label_field
            with torch.no_grad():
                for batch_tuple in test_data2:
                    batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                            else batch_tuple
                    batch = batch.to(cfg2["device"])
                    preds  = m2.predict(batch).cpu().tolist()
                    truths = batch[rating_field].cpu().tolist()
                    all_preds.extend(preds)
                    all_truths.extend(truths)

            auc_std = compute_auc_from_ratings(
                np.array(all_truths, dtype=float),
                np.array(all_preds,  dtype=float),
            )
            if auc_std is not None:
                test_result_standard["auc"] = round(auc_std, 4)
                log.info(f"  → AUC standard = {auc_std:.4f}")

        except Exception as exc:
            log.error(f"  ✗ AUC standard fallita: {exc}")

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

    # ── 3b. LightGCN/NGCF: eval subset con Recall/NDCG + beyond-accuracy ─────
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

    # ── 4. Beyond-accuracy sul test set standard (solo LightGCN/NGCF) ─────────
    log.info("  → Beyond-accuracy sul test set standard")
    try:
        cfg = Config(
            model=model,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        init_seed(cfg["seed"], cfg["reproducibility"])
        dataset = create_dataset(cfg)
        _, _, test_data = data_preparation(cfg, dataset)

        checkpoint    = torch.load(best_ckpt, map_location=cfg["device"])
        cfg_ckpt      = checkpoint["config"]
        model_class   = get_model(cfg_ckpt["model"])
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
                if scores.dim() == 1:
                    scores = scores.unsqueeze(0)
                scores[:, 0] = -float("inf")
                topk_ids = torch.topk(scores, k=topk, dim=-1).indices
                topk_lists.extend(topk_ids.cpu().tolist())

        iid_tensor    = dataset.inter_feat[dataset.iid_field]
        pop_counter   = Counter(iid_tensor.tolist())
        n_total_inter = len(dataset.inter_feat)
        catalog_size  = dataset.item_num
        item_emb      = get_item_emb(trained_model)

        ba_standard = compute_beyond_accuracy(
            topk_lists           = topk_lists,
            item_embeddings      = item_emb,
            item_popularity      = pop_counter,
            n_train_interactions = n_total_inter,
            catalog_size         = catalog_size,
        )

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
        description="RecBole CF (LightGCN, NGCF, GCMC) — RS split",
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