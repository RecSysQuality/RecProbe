import os
import sys
import pandas as pd
import json
import pandas as pd
from sklearn.utils import shuffle
from src.utils import *
from src.data_model.reviewModel import ReviewModel

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__))))
from datetime import datetime
from src.config.config import load_config

def split_dataset_total(df: pd.DataFrame, training: float = 0.8, validation: float = 0.1, test: float = 0.1, strategy: str = "random holdout", noise_in_test: bool = False, seed: int = 42):
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
    if strategy == 'random holdout':
        train_df, validation_df, test_df = random_holdout(df,training,validation,test,noise_in_test,seed)
    elif strategy == 'temporal leave-one-out':
        train_df, validation_df, test_df = temporal_leave_one_out(df,seed)
    elif strategy == 'random leave-one-out':
        train_df, validation_df, test_df = random_leave_one_out_per_user(df,seed)
    else:
        return ValueError("strategy unknown")
    return  train_df, validation_df, test_df


def random_holdout(df: pd.DataFrame, training: float = 0.8, validation: float = 0.1, test: float = 0.1,noise_in_test: bool = False, seed: int = 42):
    df = shuffle(df, random_state=seed).reset_index(drop=True)
    if training + validation + test != 1.0:
        raise ValueError("training + validation + test must be equal to 1.0")
    # Separiamo le righe con noise

    # if 'noise' in df.columns:
    #     noise_rows = df[df['noise'] == True]
    #     clean_rows = df[df['noise'] != True]
    # else:
    #     noise_rows = pd.DataFrame(columns=df.columns)
    #     clean_rows = df.copy()

    n_total = len(df)
    n_train = int(n_total * training)
    n_validation = int(n_total * validation)
    n_test = int(n_total * test)

    # if not noise_in_test:
    #     # Tutte le righe con noise nel train
    #     train_noise = noise_rows.copy()
    #
    #     # Numero di righe pulite da aggiungere al train
    #     n_clean_needed = n_train - len(train_noise)
    #     if n_clean_needed <= 0:
    #         # troppe righe noise: train = solo noise
    #         train_clean = pd.DataFrame(columns=df.columns)
    #         test_clean = clean_rows
    #     else:
    #         train_clean = clean_rows.iloc[:n_clean_needed]
    #         test_clean = clean_rows.iloc[n_clean_needed:]
    #
    #     # Combina
    #     train_df = pd.concat([train_clean, train_noise])
    #     validation_df = test_clean[:n_validation]
    #     test_df = test_clean[n_test:]
    # else:
        # Tutte le righe con noise nel train
        # train_df = df[:n_train]
        # validation_df = df[n_train:n_train+n_validation]
        # test_df = df[:-n_test]
    train_df = df[:n_train]
    validation_df = df[n_train:n_train + n_validation]
    test_df = df[-n_test:]
    #assert len(train_df) + len(validation_df) + len(test_df) == len(df)

    # Shuffle finale
    train_df = shuffle(train_df, random_state=seed).reset_index(drop=True)
    validation_df = shuffle(validation_df, random_state=seed).reset_index(drop=True)
    test_df = shuffle(test_df, random_state=seed).reset_index(drop=True)

    return train_df, validation_df, test_df



def temporal_leave_one_out(df: pd.DataFrame,seed: int = 42):
    df = df.copy()
    time_col = 'timestamp'
    user_col = 'user_id'
    df[time_col] = df[time_col].apply(parse_timestamp)
    df = df.sort_values([user_col, time_col])

    train_list = []
    val_list = []
    test_list = []

    for user, user_df in df.groupby(user_col):

        if len(user_df) < 3:
            continue  # oppure gestisci diversamente

        train_list.append(user_df.iloc[:-2])
        val_list.append(user_df.iloc[-2:-1])
        test_list.append(user_df.iloc[-1:])

    train_df = pd.concat(train_list).reset_index(drop=True)
    validation_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, validation_df, test_df

def random_leave_one_out_per_user(
        df: pd.DataFrame,
        seed: int = 42
):
    np.random.seed(seed)

    train_list = []
    val_list = []
    test_list = []
    user_col = 'user_id'
    for user, user_df in df.groupby(user_col):

        if len(user_df) < 3:
            continue  # oppure gestisci a parte

        user_df = shuffle(user_df, random_state=seed)

        test_row = user_df.iloc[0:1]
        val_row = user_df.iloc[1:2]
        train_rows = user_df.iloc[2:]

        test_list.append(test_row)
        val_list.append(val_row)
        train_list.append(train_rows)

    train_df = pd.concat(train_list).reset_index(drop=True)
    validation_df = pd.concat(val_list).reset_index(drop=True)
    test_df = pd.concat(test_list).reset_index(drop=True)

    return train_df, validation_df, test_df