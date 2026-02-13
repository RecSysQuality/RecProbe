import os
import sys
import pandas as pd
import json
import pandas as pd
from sklearn.utils import shuffle

from src.data_model.reviewModel import ReviewModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from datetime import datetime
from src.config.config import load_config
from src.data_handler.splitter import *

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

    def load_data(self):
        path_train = f"{BASE_DIR}/data/output/{self.config.dataset}/train.csv"
        path_validation = f"{BASE_DIR}/data/output/{self.config.dataset}/validation.csv"
        path_test = f"{BASE_DIR}/data/output/{self.config.dataset}/test.csv"
        if not os.path.exists(path_train):
            if self.config.input.reviews['format'] == 'jsonl':
                df = self.load_jsonl(
                    f"{BASE_DIR}/data/input/{self.config.dataset}/{self.config.input.reviews['file_name']}.{self.config.input.reviews['format']}")
            elif self.config.input.reviews['format'] == 'json':
                df = self.load_jsonl(
                    f"{BASE_DIR}/data/input/{self.config.dataset}/{self.config.input.reviews['file_name']}.{self.config.input.reviews['format']}")
            elif self.config.input.reviews['format'] == 'csv':
                df = self.load_csv(
                    f"{BASE_DIR}/data/input/{self.config.dataset}/{self.config.input.reviews['file_name']}.{self.config.input.reviews['format']}")

            df = self.normalize(df, 'interactions', self.config.dataset)
            df = self.extract_kcore(df)
            df = self.filter_by_rating_review(df)
            df,df_val,df_test = self.save_csv(df=df,modified=pd.DataFrame(),clean=True)
        else:
            df = pd.read_csv(path_train)
            df_val = pd.read_csv(path_validation)
            df_test = pd.read_csv(path_test)
            if self.config.split.noise_in_test:
                df = pd.concat([df,df_val,df_test])

        return df,df_val,df_test

    def save_csv(self, df: pd.DataFrame, modified: pd.DataFrame,clean=True):
        """Salva un DataFrame in CSV"""
        separator = self.config.output.reviews['separator']


        if self.config.drop_duplicates:
            df.drop_duplicates(inplace=True)

        # todo manage the items part

        if not os.path.exists(f"{BASE_DIR}/data/output/{self.config.dataset}/"):
            os.makedirs(f"{BASE_DIR}/data/output/{self.config.dataset}/",exist_ok=True)



        path = f"{BASE_DIR}/data/output/{self.config.dataset}/train_noisy_{self.config.noise_profile}_{self.config.noise_context}.csv"
        path_test = f"{BASE_DIR}/data/output/{self.config.dataset}/test.csv"
        path_validation = f"{BASE_DIR}/data/output/{self.config.dataset}/validation.csv"
        path_mod = f"{BASE_DIR}/data/output/{self.config.dataset}/modified_{self.config.noise_profile}_{self.config.noise_context}.csv"
        if modified is not None and len(modified) > 0:
            self.logger.info(f"Saving noisy DataFrame to CSV at {path}")
            modified.to_csv(path_mod, index=False, sep=separator)

        if clean: # clean means before noise injection
            self.logger.info(f"Computing stats on clean data...")
            self._log_stats(df)
            path_train = f"{BASE_DIR}/data/output/{self.config.dataset}/train.csv"
            self.logger.info(f"Saving clean training DataFrame to CSV at {path_train}")
            self.logger.info(f"Saving clean test DataFrame to CSV at {path_test}")
            self.logger.info(f"Saving clean validation DataFrame to CSV at {path_validation}")
            train,validation,test = split_dataset_total(df,self.config.split.training, self.config.split.validation, self.config.split.test, self.config.split.strategy,self.config.split.noise_in_test, self.config.random_seed)
            train.to_csv(path_train, index=False, sep=separator)
            test.to_csv(path_test, index=False, sep=separator)
            validation.to_csv(path_validation, index=False, sep=separator)
            self.logger.info(f"Train set with {len(train)} rows")
            self._log_split(train)
            self.logger.info(f"Test set with {len(test)} rows")
            self._log_split(test)
            self.logger.info(f"Validation set with {len(validation)} rows")
            self._log_split(validation)
            self.logger.info(f"Computing stats on clean split data...")
            if self.config.split.noise_in_test:
                # se voglio sporcare training e validation allora devo ritornare tutto insieme e faccio un altro split dopo
                return pd.concat([train,validation,test]),None,None
            else:
                # se no torno solo il train
                return train,validation,test

        else:
            # salvo dopo l'injection assicurandomi che quello che è in training non ci sia anche in test in termini di user,item interaction
            if self.config.split.noise_in_test:
                path = f"{BASE_DIR}/data/output/{self.config.dataset}/train_noisy_{self.config.noise_profile}_{self.config.noise_context}.csv"
                path_test = f"{BASE_DIR}/data/output/{self.config.dataset}/test_noisy_{self.config.noise_profile}_{self.config.noise_context}.csv"
                path_validation = f"{BASE_DIR}/data/output/{self.config.dataset}/validation_noisy_{self.config.noise_profile}_{self.config.noise_context}.csv"
                train, validation, test = split_dataset_total(df, self.config.split.training,
                                                              self.config.split.validation, self.config.split.test,
                                                              self.config.split.strategy,
                                                              self.config.split.noise_in_test, self.config.random_seed)
                train.to_csv(path, index=False, sep=separator)
                test.to_csv(path_test, index=False, sep=separator)
                validation.to_csv(path_validation, index=False, sep=separator)
                self._log_split(train)
                self._log_split(validation)
                self._log_split(test)
                return
            else:
                path_test = f"{BASE_DIR}/data/output/{self.config.dataset}/test.csv"
                test = pd.read_csv(path_test)
                validation = pd.read_csv(path_validation)

                df['pair'] = list(zip(df['user_id'], df['item_id']))
                test_pairs = set(zip(test['user_id'], test['item_id']))
                validation_pairs = set(zip(validation['user_id'], validation['item_id']))

                df = df[
                    ~((df['noise']) & (df['pair'].isin(test_pairs)) & (df['pair'].isin(validation_pairs)))
                ].drop(columns='pair')
                train = df.drop(columns=['noise'])

                train.to_csv(path, index=False, sep=separator)
                self._log_split(train)
                self._log_split(validation)
                self._log_split(test)
                return


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
            #self._log_stats(df_normalized)
            return df_normalized

    def filter_by_rating_review(self, df: pd.DataFrame) -> pd.DataFrame:
        """Filter by rating and review length"""

        rating_to_filter = self.config.min_rating
        review_to_filter = self.config.min_review_length

        df = df[
            (df['rating'] >= rating_to_filter) &
            (df['review_text'].str.len() >= review_to_filter)
            ]

        return df




    def extract_kcore(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estrae il k-core dal dataframe"""

        if self.config.drop_duplicates:
            df.drop_duplicates(inplace=True)
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
        #self._log_stats(df)
        return df

    def _log_stats(self, df: pd.DataFrame):
        self.logger.info(f"Loaded df with {len(df)} records")
        if 'user_id' in df.columns and 'item_id' in df.columns:
            self.logger.info(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}\n\n")
            n_users = df['user_id'].nunique()
            n_items = df['item_id'].nunique()
            max_items_per_user = df.groupby('user_id')['item_id'].count().max()
            max_users_per_item = df.groupby('item_id')['user_id'].count().max()
            n_interactions = len(df)
            self.logger.info(f"Total interactions: {n_interactions}\n\n")

            # Sparsity = 1 - (interazioni osservate / interazioni possibili)
            sparsity = 1 - (n_interactions / (n_users * n_items))
            self.logger.info(f"Rating stats:\n{df['rating'].describe()}\n\n")
            self.logger.info(f"Rating value counts:\n{df['rating'].value_counts().sort_index()}\n\n")
            self.logger.info(f"Average items per user:\n{n_interactions / n_users:.2f}\n\n")
            self.logger.info(f"Average users per item:\n{n_interactions / n_items:.2f}\n\n")
            self.logger.info(f"Max items per user:\n{max_items_per_user:.2f}\n\n")
            self.logger.info(f"Max users per item:\n{max_users_per_item:.2f}\n\n")
            self.logger.info(f"Sparsity of the dataset: \n{sparsity:.6f}\n\n")

    def _log_split(self, df: pd.DataFrame):
        self.logger.info(f"Loaded df with {len(df)} records")
        if 'user_id' in df.columns and 'item_id' in df.columns:
            self.logger.info(f"Unique users: {df['user_id'].nunique()}, Unique items: {df['item_id'].nunique()}\n\n")

            n_interactions = len(df)
            self.logger.info(f"Total interactions: {n_interactions}\n\n")
            if 'noise' in df.columns:
                noisy = df[df['noise'] == True]
                self.logger.info(f"Total noisy interactions: {len(noisy)}\n\n")

            else:
                self.logger.info(f"Total noisy interactions: 0\n\n")




if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # punta a NoiseInjector/
    CONFIG_PATH = os.path.join(BASE_DIR, "config/files", "config_base.yaml")



    config = load_config(CONFIG_PATH,profile='rating')

    loader = DatasetLoader(config=config)
    # Esempio con JSONL
    df = loader.load_jsonl("../data/input/amazon_All_Beauty/All_Beauty.jsonl")

   # df = loader.load_jsonl("../data/meta_All_Beauty.jsonl")
   # df = loader.normalize_columns_items(df)

