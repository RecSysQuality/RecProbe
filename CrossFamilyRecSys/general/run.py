#!/usr/bin/env python3
"""
run_baselines.py
================
Allena Pop, ItemKNN e AsymKNN su tutte le varianti di rumore.

DIFFERENZE RISPETTO A run_cf.py (LightGCN, NGCF, GCMC)
--------------------------------------------------------
1. MODELLI NON PARAMETRICI
   Pop, ItemKNN e AsymKNN non hanno parametri da apprendere nel
   senso classico: niente learning_rate, epochs, embedding.
   Il "training" è solo la costruzione della matrice di similarità
   (KNN) o il conteggio delle frequenze (Pop).
   Di conseguenza:
   - Niente checkpoint .pth salvato su disco.
   - L'eval sui subset active/lt usa il modello tenuto IN MEMORIA
     dopo il fit, senza ricaricare da checkpoint.
   - MODEL_HPARAMS_BL contiene solo k, shrink, alpha — niente lr/epochs.

2. EVAL SU SUBSET ACTIVE/LT
   In run_cf.py eval_user_subset ricarica il modello da checkpoint.
   Qui il modello viene passato direttamente come argomento a
   eval_baseline_subset() che itera sul test_data e filtra per utente.
   Stesso calcolo di beyond-accuracy (coverage, novelty, gini).

3. NIENTE BEYOND-ACCURACY CON DIVERSITY
   Pop e ItemKNN non hanno item embedding: diversity non è calcolabile.
   compute_beyond_accuracy restituisce solo coverage, novelty, gini.

4. STAGING — identico a run_cf.py
   Stessa funzione stage_dataset, solo user_id/item_id/timestamp.
   Niente rating (questi modelli non fanno rating prediction).

5. NEG SAMPLING
   Pop e ItemKNN usano full_sort_predict con mode=full.
   Non hanno bisogno di neg sampling durante il training.
   train_neg_sample_args viene impostato a None.

6. SPLIT — RS (Random Split) su solo training set  ← FIX LEAKAGE
   ItemKNN/AsymKNN costruiscono la matrice di co-occorrenza su TUTTA
   inter_feat durante il fit. Se il file .inter contiene anche vali e test
   (come fa run_cf.py per LightGCN) la matrice "vede" i test item → leakage.
   Fix: staging con solo training; RecBole divide con RS [0.8, 0.1, 0.1].
   La subset eval (active/lt) usa i file .vali/.test salvati separatamente.
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
ROOT_DIR      = Path(__file__).resolve().parent
BASE_DATA_DIR = (ROOT_DIR / "../../data").resolve()
CONFIGS_DIR   = ROOT_DIR
RESULTS_FILE  = ROOT_DIR / "results/results_baselines_summary_all.jsonl"
STAGING_DIR   = ROOT_DIR / "staging_cf"   # stesso staging di run_cf.py

# ==============================================================================
#  DATASETS
# ==============================================================================
DATASETS = {
    "Toys_and_Games": BASE_DATA_DIR / "Toys_and_Games",
    "yelp": BASE_DATA_DIR / "yelp",
    "Books": BASE_DATA_DIR / "Books",

}

ALL_MODELS = ["EASE","Pop","ItemKNN"]

# ==============================================================================
#  IPERPARAMETRI PER DATASET × MODELLO
# ==============================================================================
#
# Pop non ha iperparametri architetturali: entry vuota → usa yaml.
#
# ItemKNN e AsymKNN: k è il parametro principale.
#
# k (numero di vicini)
#   Toys (14K item, sparsità 99.94%):
#     k=50 — catalogo piccolo, molti item con poche co-occorrenze.
#     k grande non aiuta: la matrice è già quasi tutta zero.
#   Yelp (56K item, sparsità 99.95%):
#     k=100 — default paper originale. Catalogo medio, abbastanza
#     co-occorrenze per trovare 100 vicini significativi.
#   Books (113K item, sparsità 99.97%):
#     k=50 — sparsità altissima. Con k=100 la maggior parte dei vicini
#     ha similarità quasi nulla e aggiunge solo rumore.
#     Meno vicini ma più "veri" migliorano la qualità.
#
# shrink (termine di Herlocker per penalizzare co-occorrenze rare)
#   0 su Toys/Yelp: default, per confronto pulito con letteratura.
#   10 su Books: penalizza le poche co-occorrenze su dataset sparso,
#   riducendo il rumore da item con 1-2 interazioni condivise.
#
# asymmetric_alpha (solo AsymKNN)
#   0.5 su tutti: equivale a cosine similarity, confronto diretto
#   con ItemKNN. Può essere abbassato (0.3) per ridurre popularity
#   bias, ma 0.5 è il valore standard per il confronto.

MODEL_HPARAMS_BL: dict[str, dict[str, dict]] = {
    "Toys_and_Games": {
        "Pop": {},  # nessun iperparametro architetturale
        "ItemKNN": {
            "k": 50,
            "shrink": 0,
            "normalize": True,
        },
        "AsymKNN": {
            "k": 50,
            "alpha": 0.3,  # peso cosine asinmetrica (default 0.5 = cosine sym.)
            "beta": 1.0,  # bilanciamento normalizzazione score (default)
            "q": 1,  # locality of scoring function (default)
            "knn_method": "user",

        },
    },

    # ── Yelp ──────────────────────────────────────────────────────────────────
    "yelp": {
        "Pop": {},
        "ItemKNN": {
            "k": 100,
            "shrink": 0,
            "normalize": True,
        },
        "AsymKNN": {
            "k": 100,
            "alpha": 0.3,
            "beta": 1.0,
            "q": 1,
            "knn_method": "user",
        },
    },

    # ── Amazon Books ──────────────────────────────────────────────────────────
    "Books": {
        "Pop": {},
        "ItemKNN": {
            "k": 50,  # sparsità 99.97%: k basso per vicini più "veri"
            "shrink": 10,  # penalizza co-occorrenze rare
            "normalize": True,
        },
        "AsymKNN": {
            "k": 50,
            "alpha": 0.3,
            "beta": 1.0,
            "q": 1,
            "knn_method": "user",

        },
    },
}


def get_model_hparams(dataset_name: str, model: str) -> dict:
    hparams = MODEL_HPARAMS_BL.get(dataset_name, {}).get(model, {})
    if hparams:
        log.info(
            f"  [hparams] {dataset_name}/{model} → "
            + ", ".join(f"{k}={v}" for k, v in hparams.items())
        )
    else:
        log.info(f"  [hparams] {dataset_name}/{model} → usa valori yaml (nessun override)")
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
#  STAGING — identico a run_cf.py (riusa lo stesso staging se già fatto)
# ==============================================================================
import re

def _parse_inter(path: Path) -> list[tuple]:
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

    rows = []
    for line in lines:
        parts = split_line(line)
        if len(parts) <= max(uid_i, iid_i, ts_i):
            continue
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
) -> tuple[Path, str]:
    """
    Crea tre file separati:
      <tag>.train.inter  — solo training
      <tag>.valid.inter  — solo validation
      <tag>.test.inter   — solo test

    e usa benchmark_filename in RecBole per caricarli separatamente.

    PERCHÉ TRE FILE SEPARATI (fix leakage ItemKNN/AsymKNN):
    ──────────────────────────────────────────────────────────────────────
    Con un unico file .inter + LS:valid_and_test RecBole assegna
    logicamente l'ultimo item a test, ma inter_feat contiene TUTTE le
    righe caricate dal file. ItemKNN costruisce la matrice di similarità
    chiamando dataset.inter_matrix() su inter_feat → vede train+vali+test
    → leakage diretto (recall gonfiato fino a 0.4+).

    Con benchmark_filename RecBole carica i tre file separatamente:
    inter_feat (usata dal fit di ItemKNN) conterrà SOLO il .train.inter.
    Le righe di .valid.inter e .test.inter vengono usate solo per
    costruire valid_data e test_data, non entrano mai nella matrice KNN.

    Lo split rimane temporale: i file sono già divisi a monte per
    timestamp, quindi benchmark_filename preserva esattamente quella
    divisione senza toccarla.
    ──────────────────────────────────────────────────────────────────────
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

    # benchmark_filename richiede esattamente questi tre suffissi
    train_dest = stage / f"{tag}.train.inter"
    vali_dest  = stage / f"{tag}.valid.inter"
    test_dest  = stage / f"{tag}.test.inter"

    # Riusa se già presenti e aggiornati
    if (train_dest.exists() and vali_dest.exists() and test_dest.exists()
            and train_src.stat().st_mtime <= train_dest.stat().st_mtime):
        log.info(f"  Staging già presente → {tag}.[train/valid/test].inter")
        return stage, tag

    HEADER = "user_id:token\titem_id:token\ttimestamp:float\n"

    for src, dest in [(train_src, train_dest),
                      (vali_src,  vali_dest),
                      (test_src,  test_dest)]:
        rows = _parse_inter(src)
        with dest.open("w", encoding="utf-8") as f:
            f.write(HEADER)
            for uid, iid, ts in rows:
                f.write(f"{uid}\t{iid}\t{ts}\n")
        log.info(f"  Staged {len(rows):>7} righe → {dest.name}")

    return stage, tag


def load_user_set(path: Path) -> set[str]:
    if not path.exists():
        return set()
    rows = _parse_inter(path)
    return {row[0] for row in rows}


# ==============================================================================
#  BEYOND-ACCURACY METRICS
#  Identiche a run_cf.py. I modelli baseline non hanno item embedding
#  → diversity non è calcolabile → viene sempre None.
# ==============================================================================
def compute_beyond_accuracy(
    topk_lists:      list[list[int]],
    item_popularity: dict[int, int],
    n_train_inter:   int,
    catalog_size:    int,
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
            p_i = pop / max(n_train_inter, 1)
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

    return {
        "coverage": round(coverage, 4),
        "novelty":  round(novelty,  4),
        "gini":     round(gini,     4),
    }


# ==============================================================================
#  ACCURACY METRICS MANUALE (per subset di utenti)
# ==============================================================================
def _compute_accuracy_metrics(
    scores:    "torch.Tensor",
    pos_items: "torch.Tensor",
    ks: list[int] = [50, 10, 20],
) -> dict:
    results        = {}
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
#  EVAL SU SUBSET ACTIVE/LT — BASELINE VERSION
#
#  DIFFERENZA CHIAVE rispetto a eval_user_subset in run_cf.py:
#  I modelli baseline NON salvano checkpoint .pth → il modello viene
#  passato direttamente come argomento (già in memoria dopo il fit).
#  Niente torch.load, niente ricreazione del modello da zero.
# ==============================================================================
def _build_uid_to_test_item(dataset, full_inter_feat=None) -> dict[int, int]:
    """
    Costruisce la mappa uid_interno → item_id del test item per ogni utente.

    Con eval_args = {order: "TO", split: {LS: valid_and_test}} RecBole usa
    Leave-one-out temporale: il PENULTIMO item va in valid, l'ULTIMO in test.
    Il dataloader full-sort passa al modello solo user_id (nessun item_id nel
    batch) — quindi il positive item va estratto direttamente dal dataset.

    Strategia: ordina tutte le interazioni per (uid, timestamp) e prendi
    l'item con timestamp massimo per ogni utente → è il test item.
    In caso di parità di timestamp usa l'ultimo per indice (ordine RecBole).
    """
    inter = full_inter_feat if full_inter_feat is not None else dataset.inter_feat

    uid_col   = inter[dataset.uid_field]
    iid_col   = inter[dataset.iid_field]

    uid_np = uid_col.numpy() if hasattr(uid_col, "numpy") else np.array(uid_col.tolist())
    iid_np = iid_col.numpy() if hasattr(iid_col, "numpy") else np.array(iid_col.tolist())

    # Cerca il campo timestamp (può chiamarsi "timestamp" o simile)
    ts_field = None
    for f in inter.interaction.keys():
        if "time" in f.lower() or "stamp" in f.lower():
            ts_field = f
            break

    if ts_field is not None:
        ts_col = inter[ts_field]
        ts_np  = ts_col.numpy() if hasattr(ts_col, "numpy") else np.array(ts_col.tolist())
        # Ordina per (uid, timestamp, indice) — ultimo per uid = test item
        order = np.lexsort((np.arange(len(uid_np)), ts_np, uid_np))
    else:
        # Nessun timestamp: l'ordine nel file è già temporale per RecBole
        order = np.arange(len(uid_np))

    uid_to_pos: dict[int, int] = {}
    for i in order:
        uid_to_pos[int(uid_np[i])] = int(iid_np[i])
    # Dopo il loop, uid_to_pos[u] = item con indice/timestamp massimo = test item

    return uid_to_pos


def eval_baseline_subset(
    trained_model: "torch.nn.Module",
    dataset:       object,
    test_data:     object,
    user_set:      set[str],
    subset_label:  str,
    item_popularity: dict[int, int],
    n_train_inter:   int,
    topk: int = 10,
    device: str = "cpu",
    full_inter_feat  = None,
) -> dict:
    """
    Eval di un modello baseline su un sottoinsieme di utenti.
    Usa il modello già trainato in memoria — niente checkpoint.
    Calcola Recall/NDCG/MRR/Hit@K + coverage, novelty, gini.

    FIX (rispetto alla versione precedente):
    Con eval_args mode=full + order=TO RecBole passa solo user_id nel batch;
    il positive item non è mai presente nel batch stesso.
    La mappa uid→test_item viene costruita una volta sola dal dataset
    prima del loop, poi consultata per ogni utente nel subset.
    """
    if not user_set:
        return {"error": f"user_set vuoto per {subset_label}"}

    uid_field = dataset.uid_field
    token2id  = {
        dataset.id2token(uid_field, i): i
        for i in range(dataset.user_num)
    }
    valid_uids = {token2id[u] for u in user_set if u in token2id}

    if not valid_uids:
        return {"error": f"Nessun utente di {subset_label} trovato nel dataset"}

    # ── Costruisci mappa uid → test item dal dataset (fix principale) ──────────
    # Il batch full-sort contiene solo user_id; il ground-truth va letto
    # direttamente dalle interazioni del dataset ordinate per timestamp.
    uid_to_pos = _build_uid_to_test_item(dataset)
    n_mapped   = sum(1 for u in valid_uids if u in uid_to_pos)
    log.info(
        f"  [{subset_label}] ground-truth mappati: "
        f"{n_mapped}/{len(valid_uids)} utenti del subset"
    )

    topk_lists      = []
    all_scores_list = []
    all_pos_items   = []

    trained_model.eval()
    with torch.no_grad():
        for batch_tuple in test_data:
            batch = batch_tuple[0] if isinstance(batch_tuple, (tuple, list)) \
                    else batch_tuple
            batch = batch.to(device)

            uids = batch[uid_field].cpu().tolist()
            mask = torch.tensor(
                [u in valid_uids for u in uids], dtype=torch.bool
            )
            if not mask.any():
                continue

            scores = trained_model.full_sort_predict(batch)

            # Normalizza a 2D
            if scores.dim() == 1:
                scores = scores.unsqueeze(0)

            n_batch = len(uids)
            if scores.shape[0] != n_batch:
                log.warning(f"  scores.shape={scores.shape} != batch={n_batch}, skip")
                continue

            scores[:, 0] = -float("inf")

            topk_ids = torch.topk(scores, k=topk, dim=-1).indices
            topk_lists.extend(topk_ids[mask].cpu().tolist())

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

    ba = compute_beyond_accuracy(
        topk_lists      = topk_lists,
        item_popularity = item_popularity,
        n_train_inter   = n_train_inter,
        catalog_size    = dataset.item_num,
    )
    log.info(
        f"    [{subset_label}] coverage={ba['coverage']:.4f}  "
        f"novelty={ba['novelty']:.4f}  gini={ba['gini']:.4f}"
    )
    combined = dict(acc_result)
    combined.update(ba)
    return combined


# ==============================================================================
#  SINGLE RUN
# ==============================================================================
def run_single(
    model:          str,
    dataset_name:   str,
    dataset_dir:    Path,
    noise_type:     str,
    level:          str,
    train_filename: str,
    train_subdir:   str,
    staging_root:   Path,
    topk:           int = 10,
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
    except (FileNotFoundError, ValueError) as e:
        log.error(f"  ✗ {e}")
        return _err(model, dataset_name, noise_type, level, str(e), 0)

    # AsymKNN è implementato in RecBole come ItemKNN con asymmetric_alpha.
    # Il model name per RecBole è sempre "ItemKNN".
    recbole_model_name = "ItemKNN" if model == "AsymKNN" else model

    dataset_cfg = str(CONFIGS_DIR / "datasets" / f"{dataset_name}.yaml")
    model_cfg   = str(CONFIGS_DIR / "models"   / f"{model}.yaml")
    cfg_files   = [f for f in [dataset_cfg, model_cfg] if Path(f).exists()]

    parameter_dict = {
        "data_path":             str(staging_root),
        "dataset":               tag,
        "save_dataset":          False,
        "show_progress":         True,
        "load_col":              {"inter": ["user_id", "item_id", "timestamp"]},
        "stopping_step":         20,
        "eval_step":             20,
        # Baseline non parametrici: nessun neg sampling durante il training.
        "train_neg_sample_args": None,
        # FIX LEAKAGE: tre file separati train/valid/test.
        # benchmark_filename dice a RecBole di caricare i file separatamente:
        # inter_feat (usata da ItemKNN.fit) conterrà SOLO il .train.inter.
        # Senza questo, inter_feat include train+vali+test e la matrice
        # di similarità viene costruita su dati futuri → recall gonfiato.
        # Lo split rimane temporale (i file sono già divisi per timestamp).
        "benchmark_filename": ["train", "valid", "test"],
        "eval_args": {
            "group_by": "user",
            "order":    "TO",
            "mode":     "full",
            "split":    {"LS": "valid_and_test"},
        },
        "metrics":      ["Recall", "NDCG", "MRR", "Hit"],
        "topk":         [10, 20, 50],
        "valid_metric": "NDCG@10",
    }

    # Iperparametri specifici per (dataset × modello)
    parameter_dict.update(get_model_hparams(dataset_name, model))

    # ── 2. Training ───────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        cfg = Config(
            model=recbole_model_name,
            dataset=tag,
            config_file_list=cfg_files,
            config_dict=parameter_dict,
        )
        log.info(f"  benchmark_filename: {cfg['benchmark_filename']}")
        log.info(f"  split: {cfg['eval_args']}")
        init_seed(cfg["seed"], cfg["reproducibility"])
        log.info(f"  load_col: {cfg['load_col']}")

        dataset = create_dataset(cfg)

        log.info(f"  inter_feat size: {len(dataset.inter_feat)}")

        train_data, valid_data, test_data = data_preparation(cfg, dataset)
        full_inter_feat = dataset.inter_feat  # ← aggiungi questa

        dataset.inter_feat = train_data.dataset.inter_feat
        log.info(f"  inter_feat size dopo fix: {len(dataset.inter_feat)}")
        model_class   = get_model(cfg["model"])
        trained_model = model_class(cfg, dataset).to(cfg["device"])
        trainer       = get_trainer(cfg["MODEL_TYPE"], cfg["model"])(cfg, trained_model)

        best_valid_score, best_valid_result = trainer.fit(
            train_data, valid_data,
            saved=False,       # baseline non parametrici: nessun checkpoint
            show_progress=cfg["show_progress"],
        )
        test_result = trainer.evaluate(
            test_data, load_best_model=False, show_progress=False
        )

        elapsed = time.time() - t0
        log.info(f"  ✓ Training completato in {elapsed:.1f}s  "
                 f"best_valid={best_valid_score:.4f}")

    except Exception as exc:
        elapsed = time.time() - t0
        log.error(f"  ✗ Training FALLITO dopo {elapsed:.1f}s — {exc}")
        return _err(model, dataset_name, noise_type, level, str(exc), elapsed)

    # ── 3. Popolarità per novelty e gini ──────────────────────────────────────
    iid_tensor    = dataset.inter_feat[dataset.iid_field]
    pop_counter   = Counter(iid_tensor.tolist())
    n_train_inter = len(dataset.inter_feat)

    # ── 4. Eval su active e long-tail ─────────────────────────────────────────
    # Il modello è già in memoria: lo passiamo direttamente.
    # Niente torch.load — i baseline non salvano checkpoint.
    active_users = load_user_set(dataset_dir / f"test_{dataset_name}_active.csv")
    lt_users     = load_user_set(dataset_dir / f"test_{dataset_name}_lt.csv")

    log.info(f"  → Eval active users ({len(active_users)} utenti)")
    result_active = eval_baseline_subset(
        trained_model   = trained_model,
        dataset         = dataset,
        test_data       = test_data,
        user_set        = active_users,
        subset_label    = "active",
        item_popularity = pop_counter,
        n_train_inter   = n_train_inter,
        topk            = topk,
        device          = str(cfg["device"]),
        full_inter_feat=full_inter_feat,
    )

    log.info(f"  → Eval long-tail users ({len(lt_users)} utenti)")
    result_lt = eval_baseline_subset(
        trained_model   = trained_model,
        dataset         = dataset,
        test_data       = test_data,
        user_set        = lt_users,
        subset_label    = "lt",
        item_popularity = pop_counter,
        n_train_inter   = n_train_inter,
        topk            = topk,
        device          = str(cfg["device"]),
        full_inter_feat=full_inter_feat,
    )

    # ── 5. Beyond-accuracy sul test set standard ──────────────────────────────
    log.info("  → Beyond-accuracy sul test set standard")
    try:
        topk_lists = []
        trained_model.eval()
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

        ba_standard = compute_beyond_accuracy(
            topk_lists      = topk_lists,
            item_popularity = pop_counter,
            n_train_inter   = n_train_inter,
            catalog_size    = dataset.item_num,
        )
        test_result_standard = dict(test_result)
        test_result_standard.update(ba_standard)
        log.info(
            f"  coverage={ba_standard['coverage']:.4f}  "
            f"novelty={ba_standard['novelty']:.4f}  "
            f"gini={ba_standard['gini']:.4f}"
        )

    except Exception as exc:
        log.error(f"  ✗ Beyond-accuracy standard fallita: {exc}")
        test_result_standard = dict(test_result)

    return {
        "model":              model,
        "dataset":            dataset_name,
        "noise_type":         noise_type,
        "level":              level,
        "status":             "ok",
        "elapsed_s":          round(elapsed, 1),
        "best_valid_score":   best_valid_score,
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


# ==============================================================================
#  CLI
# ==============================================================================
def parse_args():
    p = argparse.ArgumentParser(
        description="RecBole baselines (Pop, ItemKNN, AsymKNN) — RS split",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--datasets",      nargs="+", default=list(DATASETS.keys()))
    p.add_argument("--models",        nargs="+", default=ALL_MODELS)
    p.add_argument("--noise",         nargs="+", default=ALL_NOISE_TYPES)
    p.add_argument("--levels",        nargs="+", default=None)
    p.add_argument("--topk",          type=int,  default=10)
    p.add_argument("--staging-dir",   default=str(STAGING_DIR))
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
        print(f"\n{'Dataset':<20} {'Model':<10} {'Noise':<25} {'Level'}")
        print("-" * 65)
        for r in runs:
            print(f"{r['dataset_name']:<20} {r['model']:<10} "
                  f"{r['noise_type']:<25} {r['level']}")
        print(f"\nTotale: {total}\n")
        return

    done, errors = 0, 0
    for r in runs:
        metrics = run_single(
            model          = r["model"],
            dataset_name   = r["dataset_name"],
            dataset_dir    = r["dataset_dir"],
            noise_type     = r["noise_type"],
            level          = r["level"],
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
    log.info("=" * 72)


if __name__ == "__main__":
    main()