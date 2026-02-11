import os
import sys
import pandas as pd
import json

from NoiseInjector.dataModel.reviewModel import ReviewModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from datetime import datetime
from logger import get_logger, logging
from config.config import load_config


BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # cartella del modulo
CONFIG_DIR = os.path.join(BASE_DIR, "..", "config")    # cartella config
config_path = os.path.join(CONFIG_DIR, "files/config_base.yaml")
rating_path = os.path.join(CONFIG_DIR, "files/config_rating.yaml")


class DatasetLoader:
    def __init__(self, log_file=f"../logs/noise_injector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log",config=None):
        # Carica la config
        self.config = config
        # Inizializza logger
        self.logger = get_logger(log_file=log_file, level=logging.INFO)

    def load_csv(self, path: str, sep: str = ',') -> pd.DataFrame:
        df = pd.read_csv(path, sep=sep)
        return df

    def load_json(self, path: str) -> pd.DataFrame:
        with open(path) as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        return df

    def load_jsonl(self, path: str) -> pd.DataFrame:
        # Usa chunks per file grandi
        chunks = pd.read_json(path, lines=True, chunksize=100_000)
        self.logger.info(f"Loading JSONL file in chunks from {path}")
        df = pd.concat(chunks, ignore_index=True)
        self.logger.info(f"Finished loading JSONL file from {path}")
        return df


    def normalize(self, df: pd.DataFrame, data_type: str, dataset: str) -> pd.DataFrame:
        """Uniforma le colonne in base al tipo di dato (ratings o items)"""
        if data_type == 'interactions':
            df_normalized = ReviewModel.from_dataframe(df, dataset=dataset)
            loader._log_stats(df_normalized)
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

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # punta a NoiseInjector/
    CONFIG_PATH = os.path.join(BASE_DIR, "config/files", "config_base.yaml")


    config = load_config(CONFIG_PATH, path_rating=rating_path)

    loader = DatasetLoader(config=config)
    # Esempio con JSONL
    df = loader.load_jsonl("../data/All_Beauty.jsonl")
    df = loader.normalize(df,'interactions',config.dataset)
    df = loader.extract_kcore(df)
   # df = loader.load_jsonl("../data/meta_All_Beauty.jsonl")
   # df = loader.normalize_columns_items(df)

