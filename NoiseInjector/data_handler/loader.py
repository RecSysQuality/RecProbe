import os
import sys
import pandas as pd
import json
import pandas as pd
from sklearn.utils import shuffle

from NoiseInjector.dataModel.reviewModel import ReviewModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from datetime import datetime
from config.config import load_config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cartella del modulo
CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")    # cartella config
config_path = os.path.join(CONFIG_DIR, "files/config_base.yaml")
rating_path = os.path.join(CONFIG_DIR, "files/config_rating.yaml")
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

class DatasetLoader:
    def __init__(self, logger,config=None):
        # Carica la config
        self.config = config
        # Inizializza logger
        self.logger = logger

    def save_csv(self, df: pd.DataFrame):
        """Salva un DataFrame in CSV"""
        separator = self.config.output.reviews['separator']
        self._log_stats(df)
        # todo manage the items part
        if not os.path.exists(f"{BASE_DIR}/data/output/{self.config.dataset}/"):
            # crea la cartella
            os.makedirs(f"{BASE_DIR}/data/output/{self.config.dataset}/",exist_ok=True)
        path = f"{BASE_DIR}/data/output/{self.config.dataset}/{self.config.input.reviews['file_name']}.csv"
        self.logger.info(f"Saving noisy DataFrame to CSV at {path}")
        if self.config.output.split == 0.0:
            if 'noise' in df.columns:
                df = df.drop(columns=['noise'])
            df.to_csv(path, index=False, sep=separator)
            self.logger.info(f"CSV saved successfully at {path}")
        else:
            train,test = self.split_dataset_total(df,self.config.output.split,self.config.random_seed)
            path_train = f"{BASE_DIR}/data/output/{self.config.dataset}/train.csv"
            path_test = f"{BASE_DIR}/data/output/{self.config.dataset}/test.csv"
            if 'noise' in train.columns:
                train = train.drop(columns=['noise'])
            if 'noise' in test.columns:
                test = test.drop(columns=['noise'])
            self.logger.info(f"Train set with {len(train)} rows")
            self.logger.info(f"Test set with {len(test)} rows")

            train.to_csv(path_train, index=False, sep=separator)
            test.to_csv(path_test, index=False, sep=separator)

    def load_csv(self, path: str, sep: str = ',') -> pd.DataFrame:
        df = pd.read_csv(path, sep=sep)
        df = loader.normalize(df, 'interactions', config.dataset)
        df = loader.extract_kcore(df)
        return df

    def load_json(self, path: str) -> pd.DataFrame:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        df = loader.normalize(df, 'interactions', config.dataset)
        df = loader.extract_kcore(df)
        return df

    def load_jsonl(self, path: str) -> pd.DataFrame:
        # Usa chunks per file grandi
        chunks = pd.read_json(path, lines=True, chunksize=100_000)
        self.logger.info(f"Loading JSONL file in chunks from {path}")
        df = pd.concat(chunks, ignore_index=True)
        df = self.normalize(df, 'interactions', self.config.dataset)
        df = self.extract_kcore(df)
        self.logger.info(f"Finished loading JSONL file from {path}")
        return df


    def normalize(self, df: pd.DataFrame, data_type: str, dataset: str) -> pd.DataFrame:
        """Uniforma le colonne in base al tipo di dato (ratings o items)"""
        if data_type == 'interactions':
            df_normalized = ReviewModel.from_dataframe(df, dataset=dataset)
            #self._log_stats(df_normalized)
            return df_normalized



    def extract_kcore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estrae il k-core dal dataframe"""
        k = self.config.kcore
        if k > 1:
            self.logger.info(f"Extracting {k}-core from the dataset")
            # Iterativamente rimuovi utenti e item con grado < k
            while True:
                initial_len = len(df)
                user_counts = df['user_id'].value_counts()
                item_counts = df['item_id'].value_counts()

                df = df[df['user_id'].isin(user_counts[user_counts >= k].index)]
                df = df[df['item_id'].isin(item_counts[item_counts >= k].index)]

                if len(df) == initial_len:
                    break  # Nessuna rimozione, esci

            self.logger.info(f"Finished extracting {k}-core. Remaining records: {len(df)}")
        self._log_stats(df)
        return df

    def _log_stats(self, df: pd.DataFrame):
        self.logger.info(f"Loaded df with {len(df)} records")
        if 'user_id' in df.columns and 'item_id' in df.columns:
            self.logger.info(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}")
            n_users = df['user_id'].nunique()
            n_items = df['item_id'].nunique()
            n_interactions = len(df)
            # Sparsity = 1 - (interazioni osservate / interazioni possibili)
            sparsity = 1 - (n_interactions / (n_users * n_items))
            self.logger.info(f"Rating stats:\n{df['rating'].describe()}")
            self.logger.info(f"Rating value counts:\n{df['rating'].value_counts().sort_index()}")
            self.logger.info(f"Average items per user:\n{n_interactions / n_users:.2f}")
            self.logger.info(f"Average users per item:\n{n_interactions / n_items:.2f}")
            self.logger.info(f"Sparsity of the dataset: \n{sparsity:.6f}")




    def split_dataset_total(self,df: pd.DataFrame, split: float = 0.8, seed: int = 42):
        """
        Shuffle dataset e split in train/test basato sul totale,
        garantendo che il test non contenga righe con noise=True.
        Se ci sono troppe righe noise, il train può essere < split del totale.

        Args:
            df: DataFrame originale
            split: proporzione totale da mettere nel train
            seed: random seed

        Returns:
            train_df, test_df
        """
        df = shuffle(df, random_state=seed).reset_index(drop=True)

        # Separiamo le righe con noise
        if 'noise' in df.columns:
            noise_rows = df[df['noise'] == True]
            clean_rows = df[df['noise'] != True]
        else:
            noise_rows = pd.DataFrame(columns=df.columns)
            clean_rows = df.copy()

        n_total = len(df)
        n_train = int(n_total * split)

        # Tutte le righe con noise nel train
        train_noise = noise_rows.copy()

        # Numero di righe pulite da aggiungere al train
        n_clean_needed = n_train - len(train_noise)
        if n_clean_needed <= 0:
            # troppe righe noise: train = solo noise
            train_clean = pd.DataFrame(columns=df.columns)
            test_clean = clean_rows
        else:
            train_clean = clean_rows.iloc[:n_clean_needed]
            test_clean = clean_rows.iloc[n_clean_needed:]

        # Combina
        train_df = pd.concat([train_clean, train_noise])
        test_df = test_clean

        # Shuffle finale
        train_df = shuffle(train_df, random_state=seed).reset_index(drop=True)
        test_df = shuffle(test_df, random_state=seed).reset_index(drop=True)

        return train_df, test_df


if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # punta a NoiseInjector/
    CONFIG_PATH = os.path.join(BASE_DIR, "config/files", "config_base.yaml")



    config = load_config(CONFIG_PATH,profile='rating')

    loader = DatasetLoader(config=config)
    # Esempio con JSONL
    df = loader.load_jsonl("../data/input/amazon_All_Beauty/All_Beauty.jsonl")

   # df = loader.load_jsonl("../data/meta_All_Beauty.jsonl")
   # df = loader.normalize_columns_items(df)

