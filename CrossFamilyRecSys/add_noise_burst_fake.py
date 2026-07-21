import os
import random

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
BASE_DIR      = Path(__file__).resolve().parent.parent

def _save(df, injected_df, removed_df, output_dir, portion):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{portion:.2f}"

    surviving = df.drop(index=removed_df.index) if removed_df is not None else df
    corrupted_df = (
        pd.concat([surviving, injected_df], ignore_index=True)
        if injected_df is not None
        else surviving.copy()
    )
    corrupted_df = corrupted_df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    # Salva corrupted in formato RecBole .inter:
    #   separatore tab, header tipizzato, timestamp in secondi Unix (float).
    # Colonne: user_id:token  item_id:token  rating:float  timestamp:float
    inter_path = output_dir / f"corrupted_{tag}.inter"
    with inter_path.open("w", encoding="utf-8") as f:
        f.write("user_id:token\titem_id:token\trating:float\ttimestamp:float\n")
        for row in corrupted_df.itertuples(index=False):
            ts = int(pd.Timestamp(row.timestamp).value // 1_000_000_000)
            f.write(f"{row.user_id}\t{row.item_id}\t{row.rating}\t{ts}\n")

    if injected_df is not None:
        injected_df.to_csv(output_dir / f"injected_{tag}.csv", index=False)
    if removed_df is not None:
        removed_df.to_csv(output_dir / f"removed_{tag}.csv", index=False)

    return corrupted_df


def _base_setup(input_path, seed):
    rng = np.random.default_rng(seed)
    df  = pd.read_csv(input_path, parse_dates=["timestamp"])
    df  = df.sort_values(["user_id", "timestamp"]).reset_index(drop=True)

    item_popularity = df["item_id"].value_counts()          # descending
    real_users      = df["user_id"].unique()
    all_items       = df["item_id"].unique()

    return rng, df, item_popularity, real_users, all_items


def _build_user_time_bounds(all_df, train_df):
    """
    Per ogni utente restituisce due Series:
      - user_tval       : timestamp di validation (penultima interazione in all_df,
                          cioe' la piu' vecchia delle ultime 2).
      - user_tmax_train : t_max del training per quell'utente.

    I timestamp iniettati devono soddisfare: t_max_train < ts < t_val

    Returns
    -------
    user_tval       : pd.Series {user_id -> pd.Timestamp}
    user_tmax_train : pd.Series {user_id -> pd.Timestamp}
    """
    # Ultime 2 interazioni per utente in all_df (penultima=val, ultima=test)
    last2 = (
        all_df.sort_values(["user_id", "timestamp"])
              .groupby("user_id")
              .tail(2)
    )
    # t_val = il piu' vecchio dei due (penultimo = validation)
    user_tval = last2.groupby("user_id")["timestamp"].min()

    # t_max_train per utente
    user_tmax_train = train_df.groupby("user_id")["timestamp"].max()

    return user_tval, user_tmax_train


def _build_item_users_index(forb_set):
    """
    Invert forb_set {(user, item)} into a dict {item: set(users)}.
    O(|forb_set|) -- built once, used for O(1) per-item lookups.
    """
    index = {}
    for u, it in forb_set:
        index.setdefault(it, set()).add(u)
    return index


def _inject_for_items(rng, target_items, item_popularity, real_users,
                      forb_set, rating,
                      user_tval_ns, user_tmax_train_ns,
                      deg_lo=0.05, deg_hi=0.15):
    """
    For each item in target_items inject between deg_lo*degree and deg_hi*degree
    new (user, item) edges, sampling users uniformly without replacement.

    Il timestamp iniettato per ogni utente e' campionato uniformemente in
    (t_max_train, t_val), cioe' strettamente dopo l'ultimo training e
    strettamente prima del validation.
    """
    users_set = set(real_users)
    users_arr = np.array(list(users_set))

    item_users_index = _build_item_users_index(forb_set)

    u_col, i_col, t_col = [], [], []
    skipped_items = 0

    for i, item in enumerate(target_items):
        if i % 100 == 0:
            print(f"{i}/{len(target_items)}")
        degree   = int(item_popularity[item])
        lo       = max(1, int(degree * deg_lo))
        hi       = max(lo, int(degree * deg_hi))
        n_inject = int(rng.integers(lo, hi + 1))

        already = item_users_index.get(item, set())
        free    = users_set - already
        if not free:
            skipped_items += 1
            continue

        candidates   = np.fromiter(free, dtype=users_arr.dtype, count=len(free))
        n_inject     = min(n_inject, len(candidates))
        chosen_users = rng.choice(candidates, size=n_inject, replace=False)

        item_users_index.setdefault(item, set()).update(chosen_users.tolist())
        forb_set.update((u, item) for u in chosen_users)

        # Timestamp: campionato in (t_max_train, t_val) per ogni utente
        for user in chosen_users:
            t_lo = user_tmax_train_ns.get(user, 0)
            t_hi = user_tval_ns.get(user, t_lo)
            if t_hi > t_lo:
                ts = int(rng.integers(t_lo + 1, t_hi))
            else:
                # finestra collassata: usa t_lo stesso
                ts = t_lo
            u_col.append(user)
            i_col.append(item)
            t_col.append(ts)

    if not u_col:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"]), skipped_items

    injected_df = pd.DataFrame({
        "user_id":   u_col,
        "item_id":   i_col,
        "rating":    rating,
        "timestamp": pd.to_datetime(np.array(t_col, dtype=np.int64)),
    })
    return injected_df, skipped_items


def _remove_for_items(rng, target_items, df, deg_lo=0.6, deg_hi=0.9, min_remaining=2):
    item_idx_map = {
        item: grp.index.to_numpy()
        for item, grp in df.groupby("item_id", sort=False)
    }

    remove_indices = []

    for i, item in enumerate(target_items):
        if i % 100 == 0:
            print(f"{i}/{len(target_items)}")

        item_idx = item_idx_map.get(item)
        if item_idx is None:
            continue
        degree = len(item_idx)

        max_removable = max(0, degree - min_remaining)
        if max_removable == 0:
            continue

        lo       = max(1, int(degree * deg_lo))
        hi       = max(lo, int(degree * deg_hi))
        n_remove = min(int(rng.integers(lo, hi + 1)), max_removable)

        chosen = rng.choice(item_idx, size=n_remove, replace=False)
        remove_indices.extend(chosen.tolist())

    return df.loc[remove_indices].copy()


# ---------------------------------------------------------------------------
# Fake-user helpers
# ---------------------------------------------------------------------------

def _compute_median_seq_len(df: pd.DataFrame) -> int:
    """
    Restituisce la mediana del numero di interazioni per utente nel training set.
    Ogni fake user avrà esattamente questa lunghezza di sequenza.
    """
    seq_lens = df.groupby("user_id").size()
    return max(2, int(np.median(seq_lens)))


def _fake_user_ids(n: int, prefix: str = "FAKE_") -> list:
    """Genera n user_id fake che non collidono con quelli reali."""
    return [f"{prefix}{i:08d}" for i in range(n)]


def _sample_timestamps_for_fake(rng, df: pd.DataFrame, n_users: int, seq_len: int):
    """
    Per ogni fake user campiona seq_len timestamp nell'intervallo globale
    del training set, ordinati crescenti (sequenza temporalmente coerente).

    Restituisce una lista di array numpy, uno per fake user.
    """
    t_lo = df["timestamp"].min().value
    t_hi = df["timestamp"].max().value
    if t_hi <= t_lo:
        t_hi = t_lo + 1

    result = []
    for _ in range(n_users):
        ts = np.sort(rng.integers(t_lo, t_hi, size=seq_len, endpoint=True))
        result.append(ts)
    return result


def _build_fake_interactions_item_centric(
    rng,
    item_pool: np.ndarray,      # item da boostare (random o popolari)
    item_popularity: pd.Series, # degree reale di ogni item
    seq_len: int,               # minimo interazioni per fake user (= mediana reale)
    deg_lo: float,              # boost minimo per item: degree * deg_lo
    deg_hi: float,              # boost massimo per item: degree * deg_hi
    df: pd.DataFrame,
    prefix: str,
    rating: int = 5,
) -> tuple:
    """
    Costruisce il DataFrame delle interazioni fake con logica item-centrica.

    Per ogni item del pool inietta tra degree*deg_lo e degree*deg_hi archi
    fake — questo garantisce il boost reale sul degree dell'item.

    Le assegnazioni vengono poi raggruppate in fake users di esattamente
    seq_len interazioni ciascuno (= mediana delle sequenze reali), così
    nessun fake user ha una sola interazione.

    Se il totale degli archi non è divisibile per seq_len, i posti
    rimanenti vengono riempiti con item random dal pool (padding).

    Returns: (DataFrame, n_fake_users, total_boost_edges)
    """
    t_lo = df["timestamp"].min().value
    t_hi = df["timestamp"].max().value
    if t_hi <= t_lo:
        t_hi = t_lo + 1

    # Step 1: per ogni item genera n_inject assegnazioni (item-centrico)
    item_assignments = []
    for item in item_pool:
        degree   = int(item_popularity[item])
        lo       = max(1, int(degree * deg_lo))
        hi       = max(lo, int(degree * deg_hi))
        n_inject = int(rng.integers(lo, hi + 1))
        item_assignments.extend([item] * n_inject)

    rng.shuffle(item_assignments)   # mescola → fake users eterogenei
    total_edges = len(item_assignments)

    # Step 2: raggruppa in fake users di esattamente seq_len item
    n_fake     = max(1, int(np.ceil(total_edges / seq_len)))
    target_len = n_fake * seq_len
    if total_edges < target_len:
        # padding con item random dal pool
        extra = rng.choice(item_pool, size=target_len - total_edges, replace=True)
        item_assignments.extend(extra.tolist())

    # Step 3: costruisci DataFrame
    u_col, i_col, t_col = [], [], []
    for idx in range(n_fake):
        fake_id = f"{prefix}{idx:08d}"
        chunk   = item_assignments[idx * seq_len : (idx + 1) * seq_len]
        ts      = np.sort(rng.integers(t_lo, t_hi, size=len(chunk), endpoint=True))
        u_col.extend([fake_id] * len(chunk))
        i_col.extend(chunk)
        t_col.extend(ts.tolist())

    return pd.DataFrame({
        "user_id":   u_col,
        "item_id":   i_col,
        "rating":    rating,
        "timestamp": pd.to_datetime(np.array(t_col, dtype=np.int64)),
    }), n_fake, total_edges


# ---------------------------------------------------------------------------
# 1. Random Burst — fake users con item casuali
# ---------------------------------------------------------------------------
#
# LOGICA
# ------
# portion = frazione di archi da aggiungere rispetto al training set reale.
# Es. portion=0.10 → inietta il 10% di interazioni fake in più.
#
# Concentrazione (deg_lo=0.5, deg_hi=1.0):
#   Ogni fake user interagisce con item_pool estratto CASUALMENTE
#   dal catalogo. Gli item non hanno nessuna logica di selezione — simulano
#   bot non sofisticati o account compromessi con comportamento rumoroso.
#
# Numero di fake users:
#   budget = len(df) * portion
#   n_fake_users = budget // seq_len
#   Ogni fake user ha seq_len = mediana delle interazioni degli utenti reali.
#   Così nessun fake user ha una sola interazione (irrealistico).
#
# Il concentration_factor (media di deg_lo e deg_hi) descrive quante
# interazioni aggiuntive porta ogni fake user rispetto alla mediana:
#   effective_budget = budget * concentration_factor
#   ma poiché seq_len è già la mediana, il parametro serve solo
#   per scalare quanti item per fake user vengono scelti dal pool.

def random_burst_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    rating=5,
    deg_lo=1/2,     # boost minimo per item: +33% del suo degree
    deg_hi=1.0,     # boost massimo per item: +100% del suo degree
    seed=42,
):
    """
    Rumore esterno — fake users con item casuali (bot non sofisticati).

    Logica item-centrica:
      - Seleziona item casuali dal catalogo finché il budget (portion) è esaurito
      - Per ogni item inietta degree * U[deg_lo, deg_hi] archi fake
      - Gli archi vengono raggruppati in fake users di almeno seq_len interazioni
        (seq_len = mediana delle sequenze reali) → nessun fake user con 1 solo item
    """
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)

    seq_len = _compute_median_seq_len(df)
    budget  = int(len(df) * portion)

    # Seleziona item casuali dal catalogo finché il budget stimato è esaurito
    shuffled   = rng.permutation(all_items)
    item_pool  = []
    estimated  = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in shuffled:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else all_items[:1]

    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="BURST_", rating=rating,
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[RandomBurst] Real edges:              {len(df)}")
    print(f"[RandomBurst] Portion:                 {portion:.0%}")
    print(f"[RandomBurst] Item boost:              [+{deg_lo:.0%}, +{deg_hi:.0%}] del degree")
    print(f"[RandomBurst] Median seq len (real):   {seq_len}")
    print(f"[RandomBurst] Items boostati:          {len(item_pool)}")
    print(f"[RandomBurst] Fake users creati:       {n_fake}")
    print(f"[RandomBurst] Injected edges:          {len(injected_df)} ({len(injected_df)/len(df):.1%})")
    print(f"[RandomBurst] Unique items in attack:  {injected_df['item_id'].nunique()}")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# 2. Targeted Injection — fake users con item popolari
# ---------------------------------------------------------------------------
#
# LOGICA
# ------
# Stesso schema di Random Burst ma il pool di item NON è casuale:
# sono i top-k item più popolari del catalogo (top_k_frac).
#
# Concentrazione (deg_lo=0.2, deg_hi=0.5):
#   Più bassa del burst perché l'attacco è sofisticato — concentra
#   le interazioni su pochi item già forti, bastano meno fake users
#   per distorcere gli embedding di quegli item in modo significativo.
#
# Effetto atteso:
#   Gli item già popolari diventano ancora più centrali nello spazio
#   latente. Coverage scende, Gini sale, Novelty scende.
#   Gli utenti reali ricevono raccomandazioni ancora più concentrate
#   sui soliti item noti.

def targeted_injection_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    top_k_frac=0.1,     # pool = top 10% item più popolari
    rating=5,
    deg_lo=1/2,          # boost minimo per item: +33% del suo degree
    deg_hi=1.0,          # boost massimo per item: +100% del suo degree
    seed=42,
):
    """
    Rumore esterno — fake users con item popolari (shilling attack sofisticato).

    Logica item-centrica:
      - Pool = top top_k_frac item per popolarità
      - Per ogni item inietta degree * U[deg_lo, deg_hi] archi fake
      - Gli archi vengono raggruppati in fake users di almeno seq_len interazioni
        (seq_len = mediana delle sequenze reali)
      - Concentrato sui top item → degrada coverage e gini in modo marcato
    """
    rng, df, item_popularity, real_users, _ = _base_setup(input_path, seed)

    seq_len   = _compute_median_seq_len(df)
    budget    = int(len(df) * portion)

    # Pool = top item per popolarità, limitato dal budget stimato
    n_popular   = max(1, int(len(item_popularity) * top_k_frac))
    popular_all = item_popularity.head(n_popular).index.to_numpy()

    item_pool  = []
    estimated  = 0
    avg_factor = (deg_lo + deg_hi) / 2
    for item in popular_all:
        avg_inject = max(1, int(item_popularity[item] * avg_factor))
        if estimated + avg_inject > budget:
            break
        item_pool.append(item)
        estimated += avg_inject
    item_pool = np.array(item_pool) if item_pool else popular_all[:1]

    injected_df, n_fake, total_boost = _build_fake_interactions_item_centric(
        rng, item_pool, item_popularity, seq_len,
        deg_lo, deg_hi, df, prefix="TARGET_", rating=rating,
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[TargetedInjection] Real edges:              {len(df)}")
    print(f"[TargetedInjection] Portion:                 {portion:.0%}")
    print(f"[TargetedInjection] Item boost:              [+{deg_lo:.0%}, +{deg_hi:.0%}] del degree")
    print(f"[TargetedInjection] Popular pool (top {top_k_frac:.0%}): {len(popular_all)} items")
    print(f"[TargetedInjection] Items boostati:          {len(item_pool)}")
    print(f"[TargetedInjection] Median seq len (real):   {seq_len}")
    print(f"[TargetedInjection] Fake users creati:       {n_fake}")
    print(f"[TargetedInjection] Injected edges:          {len(injected_df)} ({len(injected_df)/len(df):.1%})")
    print(f"[TargetedInjection] Unique items in attack:  {injected_df['item_id'].nunique()}")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# 3. Targeted Removal
# ---------------------------------------------------------------------------

def targeted_removal_attack(
    all_path,
    input_path,
    output_dir,
    portion,
    bottom_k_frac=0.5,
    deg_lo=0.05,
    deg_hi=0.15,
    min_remaining=1,
    seed=42,
):
    rng, df, item_popularity, _, _ = _base_setup(input_path, seed)
    budget = int(len(df) * portion)
    if portion == 0.1:
        bottom_k_frac = 0.3
    if portion == 0.2:
        bottom_k_frac = 0.4
    if portion == 0.3:
        bottom_k_frac = 0.5
    n_longtail     = max(1, int(len(item_popularity) * bottom_k_frac))
    longtail_items = item_popularity.tail(n_longtail).index.to_numpy()
    longtail_items = list(longtail_items)

    target_items    = []
    estimated_total = 0
    for item in longtail_items:
        if longtail_items.index(item) % 100 == 0:
            print(f"{longtail_items.index(item)}/{len(longtail_items)}")
        degree        = int(item_popularity[item])
        max_removable = max(0, degree - min_remaining)
        avg_remove    = int(min(degree * (deg_lo + deg_hi) / 2, max_removable))
        if avg_remove == 0:
            continue
        if estimated_total + avg_remove > budget:
            break
        target_items.append(item)
        estimated_total += avg_remove

    removed_df   = _remove_for_items(rng, target_items, df, 0.6, 0.9, min_remaining)
    corrupted_df = _save(df, None, removed_df, output_dir, portion)

    print(f"[TargetedRemoval] Real edges before:      {len(df)}")
    print(f"[TargetedRemoval] Budget (portion):       {budget}")
    print(f"[TargetedRemoval] Long-tail pool:         {len(longtail_items)}")
    print(f"[TargetedRemoval] Items targeted:         {len(target_items)}")
    print(f"[TargetedRemoval] Edges removed:          {len(removed_df)} ({len(removed_df)/len(df):.1%})")
    print(f"[TargetedRemoval] Unique items affected:  {removed_df['item_id'].nunique()}")
    print(f"[TargetedRemoval] Real edges after:       {len(df) - len(removed_df)}")
    return corrupted_df, removed_df


# ---------------------------------------------------------------------------
# 4. Sequence-Disruption Burst
# ---------------------------------------------------------------------------

def _inject_for_users(rng, target_users, user_groups, all_items,
                      forb_set, rating,
                      deg_lo_frac=1/3, deg_hi_frac=1/2):
    """
    Per ogni utente inietta n_inject item con timestamp in
    (t_max_train, t_val): strettamente dopo l'ultimo training,
    strettamente prima del validation.
    """
    all_items_arr = np.array(list(all_items))
    all_items_set = set(all_items)

    user_items_index: dict[object, set] = {}
    for u, it in forb_set:
        user_items_index.setdefault(u, set()).add(it)

    u_col, i_col, t_col = [], [], []
    skipped_users = 0

    for idx, user in enumerate(target_users):
        if idx % 500 == 0:
            print(f"  users processed: {idx}/{len(target_users)}")

        user_df = user_groups[user]
        degree  = len(user_df)

        lo       = max(1, int(degree * deg_lo_frac))
        hi       = max(lo, int(degree * deg_hi_frac))
        n_inject = int(rng.integers(lo, hi + 1))

        already    = user_items_index.get(user, set())
        free_items = all_items_set - already
        if not free_items:
            skipped_users += 1
            continue

        candidates   = np.fromiter(free_items, dtype=all_items_arr.dtype,
                                   count=len(free_items))
        n_inject     = min(n_inject, len(candidates))
        chosen_items = rng.choice(candidates, size=n_inject, replace=False)

        user_items_index.setdefault(user, set()).update(chosen_items.tolist())
        forb_set.update((user, it) for it in chosen_items)

        # Finestra: [t_min_train, t_max_train] — dentro la sequenza di training
        times = user_df["timestamp"].values.tolist()
        times = random.sample(times,2)

        t_lo = user_df["timestamp"].min().value
        t_hi = user_df["timestamp"].max().value

        if t_hi > t_lo:
            ts_ns = rng.integers(t_lo, t_hi, size=n_inject, endpoint=True)
        else:
            # utente con un solo timestamp: usa t_lo
            ts_ns = np.full(n_inject, t_lo, dtype=np.int64)

        u_col.extend([user] * n_inject)
        i_col.extend(chosen_items.tolist())
        t_col.extend(ts_ns.tolist())

    if not u_col:
        return pd.DataFrame(columns=["user_id", "item_id", "rating", "timestamp"]), skipped_users

    injected_df = pd.DataFrame({
        "user_id":   u_col,
        "item_id":   i_col,
        "rating":    rating,
        "timestamp": pd.to_datetime(np.array(t_col, dtype=np.int64)),
    })
    return injected_df, skipped_users


def sequence_disruption_burst(
    all_path,
    input_path,
    output_dir,
    portion,
    rating=5,
    deg_lo_frac=1/2,
    deg_hi_frac=3/4,
    seed=42,
):
    rng, df, item_popularity, real_users, all_items = _base_setup(input_path, seed)
    budget = int(len(df) * portion)

    all_df   = pd.read_csv(all_path, parse_dates=["timestamp"], usecols=["user_id", "item_id", "timestamp"])
    forb_set = set(zip(all_df["user_id"], all_df["item_id"]))

    user_tval, user_tmax_train = _build_user_time_bounds(all_df, df)
    user_tval_ns        = {u: t.value for u, t in user_tval.items()}
    user_tmax_train_ns  = {u: t.value for u, t in user_tmax_train.items()}

    user_groups = {user: grp for user, grp in df.groupby("user_id", sort=False)}

    shuffled_users  = rng.permutation(real_users)
    target_users    = []
    estimated_total = 0
    for user in shuffled_users:
        degree     = len(user_groups[user])
        avg_inject = int(degree * (deg_lo_frac + deg_hi_frac) / 2)
        if avg_inject == 0:
            continue
        if estimated_total + avg_inject > budget:
            break
        target_users.append(user)
        estimated_total += avg_inject

    injected_df, skipped = _inject_for_users(
        rng, target_users, user_groups, all_items,
        forb_set, rating,
        deg_lo_frac=deg_lo_frac, deg_hi_frac=deg_hi_frac,
    )

    corrupted_df = _save(df, injected_df, None, output_dir, portion)

    print(f"[SeqDisruptBurst] Real edges:            {len(df)}")
    print(f"[SeqDisruptBurst] Budget (portion):      {budget}")
    print(f"[SeqDisruptBurst] Users targeted:        {len(target_users)}")
    print(f"[SeqDisruptBurst] Injected edges:        {len(injected_df)} ({len(injected_df)/len(df):.1%})")
    print(f"[SeqDisruptBurst] Skipped users:         {skipped}")
    print(f"[SeqDisruptBurst] Unique items injected: {injected_df['item_id'].nunique()}")
    return corrupted_df, injected_df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    DATASETS = ["Toys_and_Games","yelp", "Books"]
    PORTIONS = [0.1, 0.2, 0.3]

    # ── Attacchi con fake users (rumore esterno) ──────────────────────────────
    # all_path non serve più: i fake user non leggono le sequenze reali
    FAKE_USER_ATTACKS = [
        (random_burst_attack,       "random_burst_fake",       {}),
        (targeted_injection_attack, "targeted_injection_fake", {}),
    ]

    # ── Attacco su utenti reali (rumore interno) ──────────────────────────────
    REAL_USER_ATTACKS = [
        (sequence_disruption_burst, "sequence_disruption", {}),
    ]

    for dataset in DATASETS:
        print(f"\n{'='*60}\nDataset: {dataset}\n{'='*60}",flush=True)
        input_path = f"{BASE_DIR}/data/{dataset}/train_{dataset}.orig.csv"
        all_path   = f"{BASE_DIR}/data/{dataset}/{dataset}_merged.orig.csv"

        for portion in PORTIONS:

            # Fake-user attacks: non passano all_path
            for attack_fn, subfolder, kwargs in FAKE_USER_ATTACKS:
                output = f"{BASE_DIR}/data/{dataset}/{subfolder}"
                os.makedirs(output, exist_ok=True)
                print(f"\n--- {subfolder} | portion={portion} ---")
                attack_fn(
                    all_path=all_path,      # usato solo per _base_setup interno
                    input_path=input_path,
                    output_dir=output,
                    portion=portion,
                    seed=42,
                    **kwargs,
                )

            # Real-user attacks: passano all_path per i time bounds
            for attack_fn, subfolder, kwargs in REAL_USER_ATTACKS:
                output = f"{BASE_DIR}/data/{dataset}/{subfolder}"
                os.makedirs(output, exist_ok=True)
                print(f"\n--- {subfolder} | portion={portion} ---")
                attack_fn(
                    all_path=all_path,
                    input_path=input_path,
                    output_dir=output,
                    portion=portion,
                    seed=42,
                    **kwargs,
                )