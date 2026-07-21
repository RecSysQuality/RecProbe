import os
import random

import pandas as pd
import numpy as np
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

_llm_model = None
_llm_tokenizer = None
# from vllm import LLM, SamplingParams
#
# _vllm_model = None


"""
PATCH — sostituisce le parti lente di utils.py per random_burst e targeted_injection.

Strategia:
  - Prima di costruire gli archi, genera UN UNICO BATCH di review generiche brevi
    (pool_size review, es. 512 o 1024).
  - Durante la costruzione degli archi, pesca dal pool con rng.choice → zero chiamate LLM nel loop.
  - Il pool viene rigenerato solo se cambia il tipo di attacco (nuke vs neutro).

Come integrare:
  1. Copia le due funzioni qui sotto nel tuo utils.py (sovrascrivi le versioni originali).
  2. Aggiungi la funzione _generate_nuke_pool() accanto alle altre helper.
  3. In _build_fake_interactions_item_centric() aggiungi il parametro `review_pool`
     e usa pool sampling invece di chiamare generate_reviews_batch per ogni arco.
"""

import numpy as np
import pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# NUOVA HELPER — genera il pool di review generiche brevi in un unico batch
# ---------------------------------------------------------------------------

def _generate_nuke_pool(rng, pool_size: int = 512, max_new_tokens: int = 30,dataset = None) -> list[str]:
    """
    Genera un pool di `pool_size` review negative brevi e generiche in un unico
    batch LLM. Ogni review è ≤ ~20 parole (max_new_tokens=30 è più che sufficiente).

    Usa prompt minimali e variati per ottenere diversità nel pool senza overhead.
    """
    # Varianti di prompt per aumentare la diversità del pool
    templates = [
        f"Write a very short, generic negative review (1 star) in the context of {dataset} dataset. Under 15 words. Only the review text.",
        f"Write a very short, generic negative review (2 stars) {dataset} dataset. Under 15 words. Only the review text.",
        f"Write a very short negative review (1 star) {dataset} dataset. Be brief and generic. Only the review text.",
        f"Write a very short negative review (2 stars) {dataset} dataset. Be brief and generic. Only the review text.",
    ]

    # Distribuisci i prompt tra i template in modo casuale
    indices = rng.integers(0, len(templates), size=pool_size)
    prompts = [templates[i] for i in indices]

    print(f"  [LLM] Generating review pool ({pool_size} reviews, single batch)...")
    pool = generate_reviews_batch(prompts, max_new_tokens=max_new_tokens)
    print(f"  [LLM] Pool ready. Example: '{pool[0][:60]}'")
    return pool


# ---------------------------------------------------------------------------
# VERSIONE MODIFICATA — _build_fake_interactions_item_centric
# Aggiunge parametro `review_pool`: se fornito, pesca dal pool invece di chiamare LLM
# ---------------------------------------------------------------------------

def _build_fake_interactions_item_centric(
    rng, item_pool, item_popularity, seq_len,
    deg_lo, deg_hi, df, prefix, rating_fn,
    item_review_index=None, negative=False,
    review_pool=None,          # ← NUOVO: lista di review pre-generate
):
    """
    Se `review_pool` è fornito (lista di stringhe), le review vengono pescate
    dal pool con rng.choice invece di generarle con l'LLM.
    Questo riduce la generazione LLM da O(n_archi) a O(1) chiamate batch.
    """
    t_lo = int(df["timestamp:float"].min())
    t_hi = int(df["timestamp:float"].max())
    if t_hi <= t_lo:
        t_hi = t_lo + 1

    # Step 1: per ogni item genera n_inject assegnazioni
    item_assignments = []
    for item in item_pool:
        degree = int(item_popularity[item])
        lo = max(1, int(degree * deg_lo))
        hi = max(lo, int(degree * deg_hi))
        n_inject = int(rng.integers(lo, hi + 1))
        item_assignments.extend([item] * n_inject)

    rng.shuffle(item_assignments)
    total_edges = len(item_assignments)

    n_fake = max(1, int(np.ceil(total_edges / seq_len)))
    target_len = n_fake * seq_len
    if total_edges < target_len:
        extra = rng.choice(item_pool, size=target_len - total_edges, replace=True)
        item_assignments.extend(extra.tolist())

    # Step 2: costruisci record
    u_col, i_col, r_col, t_col = [], [], [], []

    for idx in range(n_fake):
        fake_id = f"{prefix}{idx:08d}"
        chunk = item_assignments[idx * seq_len: (idx + 1) * seq_len]
        ts = np.sort(rng.integers(t_lo, t_hi, size=len(chunk), endpoint=True))
        for item, t in zip(chunk, ts):
            rating = rating_fn()
            u_col.append(fake_id)
            i_col.append(item)
            r_col.append(rating)
            t_col.append(t)

    total_archi = len(u_col)

    # Step 3: assegna review
    if review_pool is not None:
        # ✅ FAST PATH: pesca dal pool pre-generato (O(1) chiamate LLM)
        pool_arr = np.array(review_pool)
        idx_sampled = rng.integers(0, len(pool_arr), size=total_archi)
        texts = pool_arr[idx_sampled].tolist()
        print(f"  [Pool] Assigned {total_archi} reviews from pool (no LLM calls).")
    else:
        # SLOW PATH: genera prompt per ogni arco e chiama LLM (comportamento originale)
        prompts = []
        for item, rating in zip(i_col, r_col):
            ref_reviews = (
                _get_ref_reviews_for_item(item, rating, item_review_index, n=2, negative=negative)
                if item_review_index is not None else []
            )
            prompt = _build_prompt_nuke(rating, ref_reviews)
            prompts.append(prompt)

        print(f"  [LLM] Generating {len(prompts)} reviews in batch (slow path)...")
        texts = generate_reviews_batch(prompts, max_new_tokens=40)

    return pd.DataFrame({
        "user_id:token": u_col,
        "item_id:token": i_col,
        "rating:float":  r_col,
        "timestamp:float": np.array(t_col, dtype=np.int64),
        "text:token_seq": texts,
    }), n_fake, total_archi


# ---------------------------------------------------------------------------
# VERSIONE MODIFICATA — random_burst_attack
# ---------------------------------------------------------------------------

def random_burst_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    deg_lo=1 / 2,
    deg_hi=1.0,
    seed=42,
    pool_size: int = 50000,      # ← quante review pre-generare
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    # NON carichiamo più review_df / item_review_index (non servono col pool)
    seq_len = _compute_median_seq_len(df)
    budget  = int(len(df) * portion)

    shuffled = rng.permutation(all_items)
    item_pool = []
    estimated = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in shuffled:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else all_items[:1]

    def rating_fn():
        return int(rng.integers(1, 3))  # 1 o 2
    st = time.time()
    # ✅ Genera il pool UNA VOLTA prima del loop sugli archi
    print('generating pool',flush=True)
    review_pool = _generate_nuke_pool(rng, pool_size=pool_size, dataset = dataset)
    print('pool generated',flush=True)
    end = time.time()
    print(end-st)

    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="BURST_",
        rating_fn=rating_fn,
        item_review_index=None,   # non serve più
        negative=True,
        review_pool=review_pool,  # ← passa il pool
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[RandomBurst] Real edges:        {len(df)}")
    print(f"[RandomBurst] Portion:           {portion:.0%}")
    print(f"[RandomBurst] Fake users:        {n_fake}")
    print(f"[RandomBurst] Injected edges:    {len(injected_df)} ({len(injected_df) / len(df):.1%})")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# VERSIONE MODIFICATA — targeted_injection_attack
# ---------------------------------------------------------------------------

def targeted_injection_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    top_k_frac=0.1,
    deg_lo=1 / 2,
    deg_hi=1.0,
    seed=42,
    pool_size: int = 50000,      # ← quante review pre-generare
):
    rng, df, item_popularity, real_users, _ = _base_setup(input_path, seed)

    seq_len = _compute_median_seq_len(df)
    budget  = int(len(df) * portion)

    n_popular = max(1, int(len(item_popularity) * top_k_frac))
    popular_all = item_popularity.head(n_popular).index.to_numpy()

    item_pool = []
    estimated = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in popular_all:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else popular_all[:1]

    def rating_fn():
        return int(rng.integers(1, 3))  # 1 o 2

    # ✅ Genera il pool UNA VOLTA prima del loop sugli archi
    st = time.time()
    print('generate pool')
    review_pool = _generate_nuke_pool(rng, pool_size=pool_size, dataset = dataset)
    end = time.time()
    print('pool generated')
    print(end-st)
    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="TARGET_",
        rating_fn=rating_fn,
        item_review_index=None,
        negative=True,
        review_pool=review_pool,  # ← passa il pool
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[TargetedInjection] Real edges:      {len(df)}")
    print(f"[TargetedInjection] Portion:         {portion:.0%}")
    print(f"[TargetedInjection] Fake users:      {n_fake}")
    print(f"[TargetedInjection] Injected edges:  {len(injected_df)} ({len(injected_df) / len(df):.1%})")
    return corrupted_df, injected_df


def get_llm_vlmm(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global _vllm_model
    if _vllm_model is None:
        print(f"[vLLM] Loading {model_name}...")
        _vllm_model = LLM(
            model=model_name,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            max_model_len=512,
        )
        print("[vLLM] Model loaded.", flush=True)
    return _vllm_model


def generate_reviews_batch_vllm(prompts, max_new_tokens=80, batch_size=None):
    llm = get_llm()
    sampling = SamplingParams(
        temperature=0.8,
        top_p=0.9,
        max_tokens=max_new_tokens,
    )
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]

def get_llm_o(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global _llm_model, _llm_tokenizer
    if _llm_model is None:
        print(f"[LLM] Loading {model_name}...")
        _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
        _llm_tokenizer.padding_side = "left"
        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            #attn_implementation="flash_attention_2",  # ← aggiungi solo questo
        )
        _llm_model.eval()
        _llm_model = torch.compile(_llm_model, mode="reduce-overhead")  # ← aggiungi questo

        print("[LLM] Model loaded.",flush=True)
    return _llm_tokenizer, _llm_model


def get_llm(model_name="meta-llama/Llama-3.1-8B-Instruct"):
    global _llm_model, _llm_tokenizer

    if _llm_model is None:
        print(f"[LLM] Loading {model_name}...")

        _llm_tokenizer = AutoTokenizer.from_pretrained(model_name)
        _llm_tokenizer.pad_token = _llm_tokenizer.eos_token
        _llm_tokenizer.padding_side = "left"

        _llm_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
            # ⚡ opzionale (solo se flash-attn installato bene)
            attn_implementation="flash_attention_2",
        )

        _llm_model.eval()

        print("[LLM] Model loaded.", flush=True)

    return _llm_tokenizer, _llm_model


def generate_reviews_batch(prompts, max_new_tokens=80, batch_size=64):
    tokenizer, model = get_llm()
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]

        # chat formatting (robusto)
        try:
            messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]

            formatted = [
                tokenizer.apply_chat_template(
                    m,
                    tokenize=False,
                    add_generation_prompt=True
                )
                for m in messages_batch
            ]

        except Exception:
            formatted = batch_prompts

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
                use_cache=True,
            )

        # 🔥 FIX: rimuovere prompt correttamente a livello token
        input_ids = inputs["input_ids"]
        generated_only = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            generated_only,
            skip_special_tokens=True
        )

        results.extend([t.strip() for t in decoded])

        if i % (5 * batch_size) == 0:
            print(f"[LLM] Batch {i // batch_size + 1}")

    return results
# def generate_reviews_batch(prompts, max_new_tokens=80, batch_size=64):
#     tokenizer, model = get_llm()
#     results = []
#
#     for i in range(0, len(prompts), batch_size):
#         batch_prompts = prompts[i:i + batch_size]
#
#         # chat formatting (robusto)
#         try:
#             messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
#             formatted = tokenizer.apply_chat_template(
#                 messages_batch,
#                 tokenize=False,
#                 add_generation_prompt=True
#             )
#         except Exception:
#             formatted = batch_prompts
#
#         inputs = tokenizer(
#             formatted,
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#             max_length=512,
#         )
#
#         # ⚠️ importante con device_map="auto"
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
#
#         with torch.inference_mode():
#             outputs = model.generate(
#                 **inputs,
#                 max_new_tokens=max_new_tokens,
#                 do_sample=True,
#                 temperature=0.8,
#                 top_p=0.9,
#                 pad_token_id=tokenizer.eos_token_id,
#                 use_cache=True,
#             )
#
#         # ✅ FIX IMPORTANTE: decoding corretto batch-wise
#         decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
#
#         # rimuovi prompt in modo sicuro
#         for text, prompt in zip(decoded, batch_prompts):
#             results.append(text[len(prompt):].strip())
#
#         if i % (5 * batch_size) == 0:
#             print(f"[LLM] Batch {i // batch_size + 1}")
#
#     return results


def generate_reviews_batch_o(prompts, max_new_tokens=80, batch_size=128):
    """
    Genera review in batch usando il modello LLM locale.
    Restituisce una lista di stringhe, una per ogni prompt.
    """
    tokenizer, model = get_llm()
    results = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i: i + batch_size]

        # Applica chat template se disponibile
        try:
            messages_batch = [[{"role": "user", "content": p}] for p in batch_prompts]
            formatted = tokenizer.apply_chat_template(
                messages_batch, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            formatted = batch_prompts
        # formatted = []
        # for p in batch_prompts:
        #     try:
        #         msgs = [{"role": "user", "content": p}]
        #         text = tokenizer.apply_chat_template(
        #             msgs, tokenize=False, add_generation_prompt=True
        #         )
        #     except Exception:
        #         text = p
        #     formatted.append(text)

        inputs = tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=tokenizer.eos_token_id,
            )

        # Decodifica solo i token generati (escludi il prompt)
        input_len = inputs["input_ids"].shape[1]
        for out in outputs:
            generated = out[input_len:]
            text = tokenizer.decode(generated, skip_special_tokens=True).strip()
            results.append(text)

        if (i // batch_size) % 5 == 0:
            print(f"  [LLM] Batch {i // batch_size + 1}/{(len(prompts) + batch_size - 1) // batch_size}",flush=True)

    return results

def _build_prompt_sequence_disruption(rating, ref_reviews=None):
    """
    Sequence disruption: utente reale che esplora item fuori profilo.
    Rating coerente con la distribuzione dell'utente.
    Se ci sono review di riferimento per quell'item, genera una review simile.
    """
    rating = int(rating)
    if rating >= 4:
        sentiment_hint = "positive"
    elif rating == 3:
        sentiment_hint = "neutral"
    else:
        sentiment_hint = "negative"

    if ref_reviews:
        ref_block = "\n".join([f'- "{r}"' for r in ref_reviews[:2]])
        prompt = (
            f"Write a short review ({sentiment_hint} tone, {rating} stars) "
            f"inspired by the style of these existing reviews:\n{ref_block}\n"
            f"Write only the review text, nothing else."
        )
    else:


            prompt = (
                f"Write a short review with a {sentiment_hint} tone and {rating} stars. "
                f"Write only the review text, nothing else."
            )
    return prompt


def _build_prompt_nuke(rating, ref_reviews=None):
    """
    Burst / Targeted injection: fake user bot, review breve e generica negativa.
    Rating 1 o 2 stelle. Stile bot: breve, generica, senza dettagli specifici.
    Se ci sono review negative di riferimento per quell'item, si ispira allo stile.
    """
    rating = int(rating)
    if ref_reviews:
        ref_block = "\n".join([f'- "{r}"' for r in ref_reviews[:2]])
        prompt = (
            f"Write a very short, generic negative review ({rating} stars), "
            f"inspired by the style of these reviews:\n{ref_block}\n"
            f"Keep it under 20 words. Write only the review text, nothing else."
        )
    else:
        prompt = (
            f"Write a very short, generic negative review ({rating} stars). "
            f"Keep it under 20 words. Write only the review text, nothing else."
        )
    return prompt


def _build_prompt_sentiment_flip(original_text, original_rating):
    """
    Sentiment flip: ribalta il sentiment di una review esistente.
    """
    original_rating = int(original_rating)
    if original_rating >= 4:
        new_sentiment = "negative"
    else:
        new_sentiment = "positive"

    prompt = (
        f"Rewrite the following review to have a {new_sentiment} sentiment, "
        f"keeping the same context and style:\n\"{original_text}\"\n"
        f"Write only the rewritten review text, nothing else."
    )
    return prompt


def _build_prompt_context_swap(original_text, dataset_name):
    """
    Context swap: cambia il contesto dell'item nella review mantenendo struttura e stile.
    """
    if dataset_name == 'yelp':
        prompt = (
            f"Rewrite the following restaurant review by keeping the same restaurant "
            f"but changing the context of use (e.g. type of meal, occasion, time of day, "
            f"type of visit), while keeping the same tone and sentence structure:\n"
            f"\"{original_text}\"\n"
            f"Write only the rewritten review text, nothing else."
        )
    else:
        prompt = (
            f"Rewrite the following product review by keeping the same product "
            f"but changing the context of use (e.g. season, activity, setting, purpose), "
            f"while keeping the same tone and sentence structure:\n"
            f"\"{original_text}\"\n"
            f"Write only the rewritten review text, nothing else."
        )
    return prompt





# ---------------------------------------------------------------------------
# Review index helpers
# ---------------------------------------------------------------------------

def _load_reviews(dataset):
    """
    Carica il CSV delle review per il dataset dato.
    Path: {BASE_DIR}/data_review/{dataset}/train_{dataset}.csv
    Restituisce un DataFrame con colonne: user_id, item_id, rating, text, timestamp
    """
    path = BASE_DIR / "data_review" / dataset / f"train_{dataset}.csv"
    df = pd.read_csv(path, sep="\t")
    df["rating:float"] = pd.to_numeric(df["rating:float"], errors="coerce")
    df = df.dropna(subset=["text:token_seq", "rating:float"])
    return df


def _build_item_review_index(review_df):
    """
    Costruisce un dizionario {item_id: DataFrame delle sue review}.
    """
    return {item: grp for item, grp in review_df.groupby("item_id:token", sort=False)}


def _get_ref_reviews_for_item(item_id, target_rating, item_review_index, n=2, negative=False):
    """
    Cerca fino a n review dell'item con rating simile a target_rating.
    Se negative=True cerca solo review con rating <= 2.
    Restituisce una lista di stringhe (testi).
    """
    grp = item_review_index.get(item_id)
    if grp is None or grp.empty:
        return []

    if negative:
        candidates = grp[grp["rating:float"] <= 2]["text:token_seq"].dropna().tolist()
    else:
        # Cerca review con rating più vicino al target
        grp = grp.copy()
        grp["dist"] = (grp["rating:float"] - target_rating).abs()
        candidates = (
            grp[grp["dist"] <= 1]
            .sort_values("dist")["text:token_seq"]
            .dropna()
            .tolist()
        )
    return candidates[:n]


# ---------------------------------------------------------------------------
# Shared helpers (originali)
# ---------------------------------------------------------------------------

def _save(df, injected_df, removed_df, output_dir, portion, review_df=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{portion:.2f}"

    surviving = df.drop(index=removed_df.index) if removed_df is not None else df
    corrupted_df = (
        pd.concat([surviving, injected_df], ignore_index=True)
        if injected_df is not None
        else surviving.copy()
    )
    inter_path = output_dir / f"corrupted_{tag}.inter"

    corrupted_df = corrupted_df.sort_values(["user_id:token", "timestamp:float"]).reset_index(drop=True)
    corrupted_df.to_csv(inter_path, sep="\t", index=False)

    if injected_df is not None:
        injected_df.to_csv(output_dir / f"injected_{tag}.csv", index=False)
    if removed_df is not None:
        removed_df.to_csv(output_dir / f"removed_{tag}.csv", index=False)

    # # Salva corrupted in formato RecBole .inter
    # has_text = "text:token_seq" in corrupted_df.columns
    #
    # with inter_path.open("w", encoding="utf-8") as f:
    #     if has_text:
    #         f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\ttext:token_seq\n")
    #     else:
    #         f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
    #     for _,row in corrupted_df.iterrows():
    #         d = row
    #         user_id = d["user_id:token"]
    #         item_id = d["item_id:token"]
    #         rating = d["rating:float"]
    #         ts = int(row.get("timestamp:float", 0))
    #         if has_text:
    #
    #             text = str(getattr(row, "text:token_seq", "")).replace("\t", " ").replace("\n", " ")
    #             f.write(f"{user_id}\t{item_id}\t{rating}\t{ts}\t{text}\n")
    #         else:
    #             f.write(f"{row.user_id}\t{row.item_id}\t{row.rating}\t{ts}\n")
    #
    # if injected_df is not None:
    #     injected_df.to_csv(output_dir / f"injected_{tag}.csv", index=False)
    # if removed_df is not None:
    #     removed_df.to_csv(output_dir / f"removed_{tag}.csv", index=False)

    return corrupted_df


def _base_setup(input_path, seed):
    rng = np.random.default_rng(seed)
    df = pd.read_csv(input_path, sep='\t')
    df["timestamp:float"] = pd.to_numeric(df["timestamp:float"], errors="coerce")
    df = df.sort_values(["user_id:token", "timestamp:float"]).reset_index(drop=True)

    item_popularity = df["item_id:token"].value_counts()
    real_users = df["user_id:token"].unique()
    all_items = df["item_id:token"].unique()

    return rng, df, item_popularity, real_users, all_items


def _build_user_time_bounds(all_df, train_df):
    last2 = (
        all_df.sort_values(["user_id:token", "timestamp:float"])
              .groupby("user_id:token")
              .tail(2)
    )
    user_tval = last2.groupby("user_id:token")["timestamp:float"].min()
    user_tmax_train = train_df.groupby("user_id:token")["timestamp:float"].max()
    return user_tval, user_tmax_train


def _build_item_users_index(forb_set):
    index = {}
    for u, it in forb_set:
        index.setdefault(it, set()).add(u)
    return index


def _compute_median_seq_len(df):
    seq_lens = df.groupby("user_id:token").size()
    return max(2, int(np.median(seq_lens)))


def _build_fake_interactions_item_centric_o(
    rng, item_pool, item_popularity, seq_len,
    deg_lo, deg_hi, df, prefix, rating_fn,
    item_review_index=None, negative=False,
):
    """
    Costruisce il DataFrame delle interazioni fake con logica item-centrica.
    rating_fn: callable() -> int, chiamata per ogni arco per campionare il rating.
    """
    t_lo = int(df["timestamp:float"].min())
    t_hi = int(df["timestamp:float"].max())
    if t_hi <= t_lo:
        t_hi = t_lo + 1

    # Step 1: per ogni item genera n_inject assegnazioni
    item_assignments = []
    for item in item_pool:
        degree = int(item_popularity[item])
        lo = max(1, int(degree * deg_lo))
        hi = max(lo, int(degree * deg_hi))
        n_inject = int(rng.integers(lo, hi + 1))
        item_assignments.extend([item] * n_inject)

    rng.shuffle(item_assignments)
    total_edges = len(item_assignments)

    n_fake = max(1, int(np.ceil(total_edges / seq_len)))
    target_len = n_fake * seq_len
    if total_edges < target_len:
        extra = rng.choice(item_pool, size=target_len - total_edges, replace=True)
        item_assignments.extend(extra.tolist())

    # Step 2: costruisci record con rating e prompt per review
    u_col, i_col, r_col, t_col, prompts = [], [], [], [], []

    for idx in range(n_fake):
        fake_id = f"{prefix}{idx:08d}"
        chunk = item_assignments[idx * seq_len: (idx + 1) * seq_len]
        ts = np.sort(rng.integers(t_lo, t_hi, size=len(chunk), endpoint=True))
        for item, t in zip(chunk, ts):
            rating = rating_fn()
            ref_reviews = (
                _get_ref_reviews_for_item(item, rating, item_review_index, n=2, negative=negative)
                if item_review_index is not None else []
            )
            prompt = _build_prompt_nuke(rating, ref_reviews)
            u_col.append(fake_id)
            i_col.append(item)
            r_col.append(rating)
            t_col.append(t)
            prompts.append(prompt)

    # Step 3: genera review in batch
    print(f"  [LLM] Generating {len(prompts)} reviews in batch...")
    texts = generate_reviews_batch(prompts, max_new_tokens=40)
    #texts = ["" for _ in prompts]
    return pd.DataFrame({
        "user_id:token": u_col,
        "item_id:token": i_col,
        "rating:float": r_col,
        "timestamp:float": np.array(t_col, dtype=np.int64),
        "text:token_seq": texts,
    }), n_fake, total_edges


# ---------------------------------------------------------------------------
# 1. Random Burst — fake users con item casuali, nuke (rating 1-2)
# ---------------------------------------------------------------------------

def random_burst_attack_o(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    deg_lo=1 / 2,
    deg_hi=1.0,
    seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    review_df = _load_reviews(dataset) if dataset else None
    item_review_index = _build_item_review_index(review_df) if review_df is not None else None
    seq_len = _compute_median_seq_len(df)
    budget = int(len(df) * portion)

    shuffled = rng.permutation(all_items)
    item_pool = []
    estimated = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in shuffled:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else all_items[:1]

    def rating_fn():
        return int(rng.integers(1, 3))  # 1 o 2

    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="BURST_",
        rating_fn=rating_fn,
        item_review_index=item_review_index,
        negative=True,
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[RandomBurst] Real edges:              {len(df)}")
    print(f"[RandomBurst] Portion:                 {portion:.0%}")
    print(f"[RandomBurst] Fake users creati:       {n_fake}")
    print(f"[RandomBurst] Injected edges:          {len(injected_df)} ({len(injected_df) / len(df):.1%})")
    #print(f"[RandomBurst] Unique items in attack:  {injected_df['item_id'].nunique()}")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# 2. Targeted Injection — fake users con item popolari, nuke (rating 1-2)
# ---------------------------------------------------------------------------

def targeted_injection_attack_o(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    top_k_frac=0.1,
    deg_lo=1 / 2,
    deg_hi=1.0,
    seed=42,
):
    rng, df, item_popularity, real_users, _ = _base_setup(input_path, seed)

    review_df = _load_reviews(dataset) if dataset else None
    item_review_index = _build_item_review_index(review_df) if review_df is not None else None

    seq_len = _compute_median_seq_len(df)
    budget = int(len(df) * portion)

    n_popular = max(1, int(len(item_popularity) * top_k_frac))
    popular_all = item_popularity.head(n_popular).index.to_numpy()

    item_pool = []
    estimated = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in popular_all:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else popular_all[:1]

    def rating_fn():
        return int(rng.integers(1, 3))  # 1 o 2

    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="TARGET_",
        rating_fn=rating_fn,
        item_review_index=item_review_index,
        negative=True,
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[TargetedInjection] Real edges:              {len(df)}")
    print(f"[TargetedInjection] Portion:                 {portion:.0%}")
    print(f"[TargetedInjection] Fake users creati:       {n_fake}")
    print(f"[TargetedInjection] Injected edges:          {len(injected_df)} ({len(injected_df) / len(df):.1%})")
   # print(f"[TargetedInjection] Unique items in attack:  {injected_df['item_id'].nunique()}")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# 3. Sequence Disruption — utenti reali, item casuali, distribuzione utente
# ---------------------------------------------------------------------------

def _inject_for_users_with_reviews(
    rng, target_users, user_groups, all_items,
    forb_set, item_review_index,
    deg_lo_frac=1 / 2, deg_hi_frac=3 / 4,
):
    all_items_arr = np.array(list(all_items))
    all_items_set = set(all_items)

    user_items_index = {}
    for u, it in forb_set:
        user_items_index.setdefault(u, set()).add(it)

    u_col, i_col, r_col, t_col, prompts = [], [], [], [], []
    skipped_users = 0

    for idx, user in enumerate(target_users):
        if idx % 500 == 0:
            print(f"  users processed: {idx}/{len(target_users)}")

        user_df = user_groups[user]
        degree = len(user_df)

        lo = max(1, int(degree * deg_lo_frac))
        hi = max(lo, int(degree * deg_hi_frac))
        n_inject = int(rng.integers(lo, hi + 1))

        already = user_items_index.get(user, set())
        free_items = all_items_set - already
        if not free_items:
            skipped_users += 1
            continue

        candidates = np.fromiter(free_items, dtype=all_items_arr.dtype, count=len(free_items))
        n_inject = min(n_inject, len(candidates))
        chosen_items = rng.choice(candidates, size=n_inject, replace=False)

        user_items_index.setdefault(user, set()).update(chosen_items.tolist())
        forb_set.update((user, it) for it in chosen_items)

        t_lo = int(user_df["timestamp:float"].min())
        t_hi = int(user_df["timestamp:float"].max())
        if t_hi > t_lo:
            ts_ns = rng.integers(t_lo, t_hi, size=n_inject, endpoint=True)
        else:
            ts_ns = np.full(n_inject, t_lo, dtype=np.int64)

        # Distribuzione rating dell'utente
        user_ratings = user_df["rating:float"].dropna().tolist()

        for item, t in zip(chosen_items, ts_ns):
            # Campiona rating dalla distribuzione dell'utente
            sampled_rating = float(rng.choice(user_ratings))
            # Cerca review di riferimento per quell'item con rating simile
            ref_reviews = _get_ref_reviews_for_item(
                item, sampled_rating, item_review_index, n=2, negative=False
            ) if item_review_index is not None else []
            prompt = _build_prompt_sequence_disruption(sampled_rating, ref_reviews)

            u_col.append(user)
            i_col.append(item)
            r_col.append(sampled_rating)
            t_col.append(t)
            prompts.append(prompt)

    if not u_col:
        return pd.DataFrame(columns=["user_id:token", "item_id:token", "rating:float", "timestamp:float", "text:token_seq"]), skipped_users, []

    return u_col, i_col, r_col, t_col, prompts, skipped_users


def sequence_disruption_burst(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    deg_lo_frac=1 / 2,
    deg_hi_frac=3 / 4,
    seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)
    budget = int(len(df) * portion)

    review_df = _load_reviews(dataset) if dataset else None
    item_review_index = _build_item_review_index(review_df) if review_df is not None else None

    all_df = pd.read_csv(all_path)
    #all_df =  all_df.rename(columns={"user_id": "user_id:token", "item_id": "item_id:token", "timestamp": "timestamp:float"})
    print(all_df.columns)

    all_df["timestamp"] = pd.to_numeric(all_df["timestamp"], errors="coerce")
    forb_set = set(zip(all_df["user_id"], all_df["item_id"]))

    user_groups = {user: grp for user, grp in df.groupby("user_id:token", sort=False)}

    shuffled_users = rng.permutation(real_users)
    target_users = []
    estimated_total = 0
    for user in shuffled_users:
        degree = len(user_groups[user])
        avg_inject = int(degree * (deg_lo_frac + deg_hi_frac) / 2)
        if avg_inject == 0:
            continue
        if estimated_total + avg_inject > budget:
            break
        target_users.append(user)
        estimated_total += avg_inject

    result = _inject_for_users_with_reviews(
        rng, target_users, user_groups, all_items,
        forb_set, item_review_index,
        deg_lo_frac=deg_lo_frac, deg_hi_frac=deg_hi_frac,
    )

    if isinstance(result[0], pd.DataFrame):
        # Nessun arco iniettato
        injected_df = result[0]
        skipped = result[1]
    else:
        u_col, i_col, r_col, t_col, prompts, skipped = result

        print(f"  [LLM] Generating {len(prompts)} reviews for sequence disruption...")
        texts = generate_reviews_batch(prompts, max_new_tokens=80)
       # texts = ["" for _ in prompts]

        injected_df = pd.DataFrame({
            "user_id:token": u_col,
            "item_id:token": i_col,
            "rating:float": r_col,
            "timestamp:float": np.array(t_col, dtype=np.int64),
            "text:token_seq": texts,
        })

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[SeqDisruptBurst] Real edges:            {len(df)}")
    print(f"[SeqDisruptBurst] Budget (portion):      {budget}")
    print(f"[SeqDisruptBurst] Users targeted:        {len(target_users)}")
    print(f"[SeqDisruptBurst] Injected edges:        {len(injected_df)} ({len(injected_df) / len(df):.1%})")
    print(f"[SeqDisruptBurst] Skipped users:         {skipped}")

    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# 4. Rating Flip — modifica archi di sequence disruption, flippa il rating
# ---------------------------------------------------------------------------

def rating_flip_attack_0(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    deg_lo_frac=1 / 2,
    deg_hi_frac=3 / 4,
    seed=42,
):
    """
    Prima esegue sequence_disruption_burst per ottenere gli archi iniettati,
    poi flippa il rating: rating <= 2 → 4 o 5, rating >= 4 → 1 o 2, rating 3 → 1 o 5.
    Il testo della review NON viene modificato.
    """
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    # Esegui sequence disruption per ottenere injected_df
    _, injected_df = sequence_disruption_burst(
        all_path=all_path,
        input_path=input_path,
        output_dir=output_dir,
        portion=portion,
        dataset=dataset,
        deg_lo_frac=deg_lo_frac,
        deg_hi_frac=deg_hi_frac,
        seed=seed,
    )

    # Flippa il rating
    def flip_rating(r):
        r = float(r)
        if r <= 2:
            return float(rng.integers(4, 6))   # 4 o 5
        elif r >= 4:
            return float(rng.integers(1, 3))   # 1 o 2
        else:  # rating 3
            return float(rng.choice([1, 5]))

    flipped_df = injected_df.copy()
    flipped_df["rating:float"] = flipped_df["rating:float"].apply(flip_rating)

    output_dir_flip = Path(output_dir)
    output_dir_flip.mkdir(parents=True, exist_ok=True)
    corrupted_df = _save(df, flipped_df, None, output_dir_flip, portion)

    print(f"[RatingFlip] Real edges:        {len(df)}")
    print(f"[RatingFlip] Flipped edges:     {len(flipped_df)}")
    return corrupted_df, flipped_df


# ---------------------------------------------------------------------------
# 5. Sentiment Flip — modifica archi di sequence disruption, ribalta il sentiment
# ---------------------------------------------------------------------------

def sentiment_flip_attack_0(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    deg_lo_frac=1 / 2,
    deg_hi_frac=3 / 4,
    seed=42,
):
    """
    Prima esegue sequence_disruption_burst per ottenere gli archi iniettati,
    poi ribalta il sentiment della review con LLM. Il rating NON viene modificato.
    """
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    # Esegui sequence disruption per ottenere injected_df
    _, injected_df = sequence_disruption_burst(
        all_path=all_path,
        input_path=input_path,
        output_dir=output_dir,
        portion=portion,
        dataset=dataset,
        deg_lo_frac=deg_lo_frac,
        deg_hi_frac=deg_hi_frac,
        seed=seed,
    )

    if "text:token_seq" not in injected_df.columns or injected_df.empty:
        print("[SentimentFlip] No text column found in injected_df, skipping.")
        return _save(df, injected_df, None, Path(output_dir), portion), injected_df

    # Costruisci prompt per ribaltare il sentiment
    prompts = []
    for _, row in injected_df.iterrows():
        prompts.append(_build_prompt_sentiment_flip(row["text:token_seq"], row["rating:float"]))

    print(f"  [LLM] Generating {len(prompts)} sentiment-flipped reviews...")
    new_texts = generate_reviews_batch(prompts, max_new_tokens=100)
    #new_texts = ["" for _ in prompts]

    flipped_df = injected_df.copy()
    flipped_df["text:token_seq"] = new_texts

    output_dir_sent = Path(output_dir)
    output_dir_sent.mkdir(parents=True, exist_ok=True)
    corrupted_df = _save(df, flipped_df, None, output_dir_sent, portion)

    print(f"[SentimentFlip] Real edges:        {len(df)}")
    print(f"[SentimentFlip] Flipped edges:     {len(flipped_df)}")
    return corrupted_df, flipped_df


def rating_flip_attack(
    all_path, input_path, output_dir, portion, dataset=None,
    deg_lo_frac=1/2, deg_hi_frac=3/4, seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    # Leggi injected_df dal CSV già generato da sequence_disruption
    tag = f"{portion:.2f}"
    seq_dir = Path(output_dir).parent / "sequence_disruption"
    injected_path = seq_dir / f"injected_{tag}.csv"

    if not injected_path.exists():
        raise FileNotFoundError(
            f"Esegui prima sequence_disruption_burst per portion={portion}.\n"
            f"File atteso: {injected_path}"
        )

    injected_df = pd.read_csv(injected_path)

    # Flippa il rating
    def flip_rating(r):
        r = float(r)
        if r <= 2:   return float(rng.integers(4, 6))
        elif r >= 4: return float(rng.integers(1, 3))
        else:        return float(rng.choice([1, 5]))

    flipped_df = injected_df.copy()
    flipped_df["rating:float"] = flipped_df["rating:float"].apply(flip_rating)

    corrupted_df = _save(df, flipped_df, None, Path(output_dir), portion)
    print(f"[RatingFlip] Real edges:    {len(df)}")
    print(f"[RatingFlip] Flipped edges: {len(flipped_df)}")
    return corrupted_df, flipped_df


def sentiment_flip_attack(
    all_path, input_path, output_dir, portion, dataset=None,
    deg_lo_frac=1/2, deg_hi_frac=3/4, seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    # Leggi injected_df dal CSV già generato da sequence_disruption
    tag = f"{portion:.2f}"
    seq_dir = Path(output_dir).parent / "sequence_disruption"
    injected_path = seq_dir / f"injected_{tag}.csv"

    if not injected_path.exists():
        raise FileNotFoundError(
            f"Esegui prima sequence_disruption_burst per portion={portion}.\n"
            f"File atteso: {injected_path}"
        )

    injected_df = pd.read_csv(injected_path)

    if "text:token_seq" not in injected_df.columns or injected_df.empty:
        print("[SentimentFlip] No text column found, skipping.")
        return _save(df, injected_df, None, Path(output_dir), portion), injected_df

    prompts = [
        _build_prompt_sentiment_flip(row["text:token_seq"], row["rating:float"])
        for _, row in injected_df.iterrows()
    ]

    print(f"  [LLM] Generating {len(prompts)} sentiment-flipped reviews...")
    new_texts = generate_reviews_batch(prompts, max_new_tokens=100)

    flipped_df = injected_df.copy()
    flipped_df["text:token_seq"] = new_texts

    corrupted_df = _save(df, flipped_df, None, Path(output_dir), portion)
    print(f"[SentimentFlip] Real edges:    {len(df)}")
    print(f"[SentimentFlip] Flipped edges: {len(flipped_df)}")
    return corrupted_df, flipped_df

# ---------------------------------------------------------------------------
# 6. Context Swap — modifica interazioni casuali esistenti, cambia il contesto
# ---------------------------------------------------------------------------
def context_swap_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    dataset=None,
    seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    if "text:token_seq" not in df.columns:
        print("[ContextSwap] No text column in df, skipping.")
        return df, pd.DataFrame()

    has_text = df["text:token_seq"].notna()
    candidates = df[has_text]

    if candidates.empty:
        print("[ContextSwap] No rows with text found, skipping.")
        return df, pd.DataFrame()

    budget = int(len(df) * portion)
    n_swap = min(budget, len(candidates))

    chosen_idx = rng.choice(candidates.index, size=n_swap, replace=False)
    target_rows = candidates.loc[chosen_idx].copy()

    # Costruisci prompt per context swap
    prompts = [
        _build_prompt_context_swap(str(row["text:token_seq"]), dataset_name=dataset)
        for _, row in target_rows.iterrows()
    ]

    print(f"  [LLM] Generating {len(prompts)} context-swapped reviews...")
    new_texts = generate_reviews_batch(prompts, max_new_tokens=100)
    #new_texts = ["" for _ in prompts]

    # Aggiorna i testi nel df originale tramite indice (O(1), niente loop)
    corrupted_df = df.copy()
    corrupted_df.loc[chosen_idx, "text:token_seq"] = new_texts

    swapped_df = target_rows.copy()
    swapped_df["text:token_seq"] = new_texts

    output_dir_ctx = Path(output_dir)
    output_dir_ctx.mkdir(parents=True, exist_ok=True)
    tag = f"{portion:.2f}"
    swapped_df.to_csv(output_dir_ctx / f"swapped_{tag}.csv", index=False)

    # Salva il corrupted in formato .inter (vectorizzato)
    corrupted_df["text:token_seq"] = (
        corrupted_df["text:token_seq"]
        .astype(str)
        .str.replace(r"[\t\n]", " ", regex=True)
    )
    inter_path = output_dir_ctx / f"corrupted_{tag}.inter"
    corrupted_df.to_csv(inter_path, sep="\t", index=False)

    print(f"[ContextSwap] Real edges:        {len(df)}")
    print(f"[ContextSwap] Swapped edges:     {len(swapped_df)}")
    return corrupted_df, swapped_df
# def context_swap_attack(
#     all_path,
#     input_path,
#     output_dir,
#     portion,
#     dataset=None,
#     seed=42,
# ):
#     """
#     Seleziona casualmente portion * len(df) interazioni esistenti nel training set
#     e ne riscrive la review cambiando il contesto dell'item con l'LLM.
#     Il rating NON viene modificato.
#     """
#     rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)
#
#     review_df = _load_reviews(dataset) if dataset else None
#
#     if review_df is None or review_df.empty:
#         print("[ContextSwap] No review data found, skipping.")
#         return df, pd.DataFrame()
#
#     # Merge df con review per avere il testo
#     merged = df.merge(
#         review_df[["user_id:token", "item_id:token", "text:token_seq"]],
#         on=["user_id:token", "item_id:token"],
#         how="left",
#     )
#     print(merged.columns)
#     has_text = merged["text:token_seq"].notna()
#     candidates = merged[has_text]
#
#     budget = int(len(df) * portion)
#     n_swap = min(budget, len(candidates))
#
#     chosen_idx = rng.choice(candidates.index, size=n_swap, replace=False)
#     target_rows = candidates.loc[chosen_idx].copy()
#
#     # Costruisci prompt per context swap
#     prompts = [_build_prompt_context_swap(str(row["text:token_seq"]), dataset_name=dataset) for _, row in target_rows.iterrows()]
#
#     print(f"  [LLM] Generating {len(prompts)} context-swapped reviews...")
#     #new_texts = generate_reviews_batch(prompts, max_new_tokens=100)
#     new_texts = ["" for _ in prompts]
#     # Modifica il df originale
#     corrupted_df = df.copy()
#
#     # Aggiungi colonna text se non presente
#     if "text:token_seq" not in corrupted_df.columns:
#         # Unisci con review originali
#         corrupted_df = corrupted_df.merge(
#             review_df[["user_id:token", "item_id:token", "text:token_seq"]],
#             on=["user_id:token", "item_id:token"],
#             how="left",
#         )
#
#     corrupted_df = merged.copy()  # merged ha già il testo
#     corrupted_df.loc[chosen_idx, "text:token_seq"] = new_texts
#     # Aggiorna i testi delle righe selezionate
#     # for i, (orig_idx, new_text) in enumerate(zip(chosen_idx, new_texts)):
#     #     # Trova la riga corrispondente nel corrupted_df
#     #     mask = (
#     #         (corrupted_df["user_id:token"] == target_rows.loc[orig_idx, "user_id:token"]) &
#     #         (corrupted_df["item_id:token"] == target_rows.loc[orig_idx, "item_id:token"])
#     #     )
#     #     corrupted_df.loc[mask, "text:token_seq"] = new_text
#
#     swapped_df = target_rows.copy()
#     swapped_df["text:token_seq"] = new_texts
#
#     output_dir_ctx = Path(output_dir)
#     output_dir_ctx.mkdir(parents=True, exist_ok=True)
#     tag = f"{portion:.2f}"
#     swapped_df.to_csv(output_dir_ctx / f"swapped_{tag}.csv", index=False)
#
#     # Salva il corrupted in formato .inter
#     inter_path = output_dir_ctx / f"corrupted_{tag}.inter"
#     with inter_path.open("w", encoding="utf-8") as f:
#         f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\ttext:token_seq\n")
#         for _, row in corrupted_df.iterrows():
#             user_id = row["user_id:token"]
#             item_id = row["item_id:token"]
#             rating = row["rating:float"]
#             ts = int(row.get("timestamp:float", 0))
#             text = str(getattr(row, "text:token_seq", "")).replace("\t", " ").replace("\n", " ")
#             f.write(f"{user_id}\t{item_id}\t{rating}\t{ts}\t{text}\n")
#
#     print(f"[ContextSwap] Real edges:        {len(df)}")
#     print(f"[ContextSwap] Swapped edges:     {len(swapped_df)}")
#     return corrupted_df, swapped_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATASETS = ["Toys_and_Games", "yelp", "Books"]
    PORTIONS = [0.1, 0.2, 0.3]
    #DATASETS = ["Toys_and_Games"]
    PORTIONS = [0.3,0.2,0.1]
    print('start')
    FAKE_USER_ATTACKS = [
        (random_burst_attack,       "random_burst_fake",       {}),
        (targeted_injection_attack, "targeted_injection_fake", {}),
    ]
    #FAKE_USER_ATTACKS = []

    # Attacchi che partono da sequence disruption
    SEQ_BASED_ATTACKS = [
        (sequence_disruption_burst, "sequence_disruption", {}),
        (rating_flip_attack,        "rating_flip",         {}),
        (sentiment_flip_attack,     "sentiment_flip",      {}),
    ]
    SEQ_BASED_ATTACKS = []
    INDEPENDENT_ATTACKS = [
        (context_swap_attack, "context_swap", {}),
    ]
    INDEPENDENT_ATTACKS = []
    for portion in PORTIONS:
        for dataset in DATASETS:
            print(f"\n{'=' * 60}\nDataset: {dataset}\n{'=' * 60}", flush=True)
            input_path = f"{BASE_DIR}/data_review/{dataset}/train_{dataset}.csv"
            all_path   = f"{BASE_DIR}/data_review/{dataset}/{dataset}_merged.csv"



            for attack_fn, subfolder, kwargs in FAKE_USER_ATTACKS:
                output = f"{BASE_DIR}/data_review/{dataset}/{subfolder}"
                os.makedirs(output, exist_ok=True)
                print(f"\n--- {subfolder} | portion={portion} ---")
                attack_fn(
                    all_path=all_path,
                    input_path=input_path,
                    output_dir=output,
                    portion=portion,
                    dataset=dataset,
                    seed=42,
                    **kwargs,
                )


            for attack_fn, subfolder, kwargs in SEQ_BASED_ATTACKS:
                output = f"{BASE_DIR}/data_review/{dataset}/{subfolder}"
                os.makedirs(output, exist_ok=True)
                print(f"\n--- {subfolder} | portion={portion} ---")
                attack_fn(
                    all_path=all_path,
                    input_path=input_path,
                    output_dir=output,
                    portion=portion,
                    dataset=dataset,
                    seed=42,
                    **kwargs,
                )

            for attack_fn, subfolder, kwargs in INDEPENDENT_ATTACKS:
                output = f"{BASE_DIR}/data_review/{dataset}/{subfolder}"
                os.makedirs(output, exist_ok=True)
                print(f"\n--- {subfolder} | portion={portion} ---")
                attack_fn(
                    all_path=all_path,
                    input_path=input_path,
                    output_dir=output,
                    portion=portion,
                    dataset=dataset,
                    seed=42,
                    **kwargs,
                )