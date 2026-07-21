import os.path

import pandas as pd
from pathlib import Path


import pandas as pd
import time
import pandas as pd

def extract_kcore(df: pd.DataFrame, k: int = 10) -> pd.DataFrame:
    """Estrae il k-core dal dataframe"""

    df.drop_duplicates(inplace=True)
    if k > 1:
        # Iterativamente rimuovi utenti e item con grado < k
        while True:
            print('kcore...')
            initial_len = len(df)
            user_counts = df['user_id'].value_counts()
            item_counts = df['item_id'].value_counts()

            df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
            df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]

            if len(df) == initial_len:
                break  # Nessuna rimozione, esci

    # self._log_stats(df)
    return df




def load_jsonl(path: str, k: int = 5) -> pd.DataFrame:
    # Usa chunks per file grandi
    st = time.time()
    chunks = pd.read_json(path, lines=True, chunksize=100_000)
    df = pd.concat(chunks, ignore_index=True)
    end = time.time()
    print(f'file read in {end-st}')
    df = df[['user_id','parent_asin','rating','timestamp']]
    df.rename(columns={"parent_asin": "item_id"}, inplace=True)
    print('ready to extract kcore')
    df = df.drop_duplicates(subset=['user_id', 'item_id','rating','timestamp'])
    df = extract_kcore(df,k)
    print('kcore extracted')
    df.to_csv(path.replace('.jsonl',f'_{k}.csv'), index=False)
    return df


def merge_csvs(path, dataset,k):

    # cartella Books/
    folder = Path(path) / dataset
    # tutti i csv nella cartella
    csv_files = folder.glob("*.csv")
    # lettura file
    dfs = [pd.read_csv(file) for file in csv_files]
    # merge (concat verticale)
    df = pd.concat(dfs, ignore_index=True)
    # salvataggio nuovo csv
    output_file = folder / f"{dataset}_merged.csv"
    df = df.drop_duplicates(subset=['user_id', 'item_id','rating','timestamp'])
    df = extract_kcore(df,k)

    df.to_csv(output_file, index=False)
    return df


def temporal_leave_one_out_split(df,users='all'):
    # Assicurati che il timestamp sia datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Ordina per utente e tempo
    df = df.sort_values(by=['user_id', 'timestamp'])

    train_list = []
    val_list = []
    test_list = []

    df = df.sort_values(['user_id', 'timestamp'])  # o la colonna temporale corretta

    # indice progressivo per ogni utente
    df['pos'] = df.groupby('user_id').cumcount()
    # dimensione del gruppo
    df['size'] = df.groupby('user_id')['user_id'].transform('size')
    #df['item_id'] = df['parent_asin']
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

def process_yelp():
    # carica il file (se è JSON lines usa lines=True)
    df = pd.read_json("../data/yelp/yelp_academic_dataset_review.json", lines=True)

    # selezione e rinomina colonne
    df = df[["business_id", "user_id", "stars", "date"]].rename(columns={
        "business_id": "item_id",
        "stars": "rating",
        "date": "timestamp"
    })
    df.to_csv("../data/yelp/yelp_merged.csv", index=False)

    # salvataggio CSV
    df = df.drop_duplicates(subset=['user_id', 'item_id','rating','timestamp'])
    df = extract_kcore(df,10)
    df.to_csv("../data/yelp/yelp_merged.csv", index=False)
    return df


if __name__ == '__main__':
    dataset = 'Books'
    datas = ['yelp','Books','Toys_and_Games']
    for dataset in datas:
        df = pd.read_csv(f'../data/{dataset}/{dataset}_merged.csv')
        print(f'{dataset}: total length: {len(df)}')
        print(f'{dataset}: total users: {len(df["user_id"].unique().tolist())}')
        print(f'{dataset}: total items: {len(df["item_id"].unique().tolist())}')
        active_df = pd.read_csv(f'../data/{dataset}/test_{dataset}_active.csv')
        print(f'{dataset}: total users active: {len(active_df["user_id"].unique().tolist())}')
        lt_df = pd.read_csv(f'../data/{dataset}/test_{dataset}_lt.csv')
        print(f'{dataset}: total users lt: {len(lt_df["user_id"].unique().tolist())}')

    for dataset in datas:
        path = f'../data/'
        k = 10
        if dataset == 'All_Beauty':
            k = 1
        if  os.path.exists(f'../data/{dataset}/{dataset}_merged.csv'):
            df = pd.read_csv(f'../data/{dataset}/{dataset}_merged.csv')
        else:
            if dataset == 'yelp':
                df = process_yelp()
            else:
                df = merge_csvs(path = path, dataset = dataset,k=k)

        train_df, val_df, test_df = temporal_leave_one_out_split(df)

        path_train = path + dataset + f'/train_{dataset}.csv'
        path_vali = path + dataset + f'/vali_{dataset}.csv'
        path_test = path + dataset + f'/test_{dataset}.csv'
        path_vali_active = path + dataset + f'/vali_{dataset}_active.csv'
        path_test_active = path + dataset + f'/test_{dataset}_active.csv'
        path_vali_lt = path + dataset + f'/vali_{dataset}_lt.csv'
        path_test_lt = path + dataset + f'/test_{dataset}_lt.csv'
        train_df.to_csv(path_train,index=False)
        val_df.to_csv(path_vali,index=False)
        test_df.to_csv(path_test,index=False)

        _, val_df_active, test_df_active = temporal_leave_one_out_split(df,'active')
        val_df_active.to_csv(path_vali_active, index=False)
        test_df_active.to_csv(path_test_active, index=False)

        _, val_df_lt, test_df_lt = temporal_leave_one_out_split(df,'lt')
        val_df_lt.to_csv(path_vali_lt,index=False)
        test_df_lt.to_csv(path_test_lt,index=False)


