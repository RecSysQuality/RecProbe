import argparse
import pandas as pd
from data_handler.loader import DatasetLoader
from orchestrator import NoiseOrchestrator
from baselines_orchestrator import BaselinesOrchestrator
from config.config import load_config
import os
from logger import get_logger, logging
from datetime import datetime
import random
import numpy as np
import torch
import yaml
from utils import *

def set_seed(seed: int = 42):
    """Setta il seed per Python, Numpy, Torch e CUDA per riproducibilità"""
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch CPU
    torch.manual_seed(seed)

    # PyTorch GPU
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # se ci sono più GPU

    # Imposta comportamenti deterministici (potrebbe rallentare un po')
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Random seed set to {seed} for Python, NumPy, Torch, and CUDA.")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/
CONFIG_PATH = os.path.join(BASE_DIR, "config/files", "config_base.yaml")
CONFIG_PATH_STREAMLIT = os.path.join(BASE_DIR, "streamlit/generated_configs", "config.yaml")

if os.path.exists(CONFIG_PATH_STREAMLIT):
    config = load_streamlit_config()


def main():
    parser = argparse.ArgumentParser(description="Noise Injector for Reviews")

    parser.add_argument(
        "--baselines",
        action="store_true",
        help="compute the baselines",
    )

    parser.add_argument(
        "--profile",
        choices=["rating", "review", "hybrid",""],
        default="rating",
        help="Noise profile",
    )

    parser.add_argument(
        "--noise_injection",
        default="user_burst_noise",
        help='Noise injection (called "context" in the config files)',
    )

    args = parser.parse_args()

    rating_options = {
        "realistic_noise",
        "user_burst_noise",
        "item_burst_noise",
        "timestamp_corruption",
    }

    review_options = {
        "remove_reviews",
        "review_burst_noise",
        "sentence_noise",
    }

    combined_options = {
        "rating_review_burst",
        "semantic_drift",
    }

    profile = args.profile
    noise_injection = args.noise_injection
    baselines = args.baselines

    valid = (
        (profile == "rating" and noise_injection in rating_options)
        or (profile == "review" and noise_injection in review_options)
        or (profile == "hybrid" and noise_injection in combined_options)
        or (profile == '' and noise_injection == '' and baselines == True)
    )

    if not valid:
        parser.error(
            f"Invalid noise_injection '{noise_injection}' for profile '{profile}'.\n"
            f"Valid options:\n"
            f"  rating:   {sorted(rating_options)}\n"
            f"  review:   {sorted(review_options)}\n"
            f"  hybrid: {sorted(combined_options)}"
            "you can also leave the two params unset and compute the baselines."
        )

    log_file=f"{BASE_DIR}/logs/noise_injector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = get_logger(log_file=log_file, level=logging.INFO)
    config = load_config(CONFIG_PATH, profile=profile, context=noise_injection)
    set_seed(config.random_seed)

    loader = DatasetLoader(logger=logger,config=config)
    df,df_val,df_test = loader.load_data()
    if profile != '' and noise_injection != '':
        # todo: do it also with items
        # empty dataframe
        orchestrator = NoiseOrchestrator(logger, config)
        df, modified = orchestrator.apply(df,df_val,df_test)
        # the split is after the injection
        loader.save_csv(df, modified,clean=False) # qua è noisy quindi clean è false
        logger.info(f"noisy df saved.")

    if baselines:
        path_train = f"{BASE_DIR}/data/output/{config.dataset}/train.csv"
        path_train_noisy = f"{BASE_DIR}/data/output/{config.dataset}/train_noisy_{profile}_{noise_injection}.csv"
        path_test_noisy = f"{BASE_DIR}/data/output/{config.dataset}/test_noisy.csv"
        path_test = f"{BASE_DIR}/data/output/{config.dataset}/test.csv"
        path_validation_noisy = f"{BASE_DIR}/data/output/{config.dataset}/validation_noisy.csv"
        path_validation = f"{BASE_DIR}/data/output/{config.dataset}/validation.csv"
        logger.info("Checking file existence...")

        config_path = f"{BASE_DIR}/baselines/config.yaml"
        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        baseline_config = cfg_dict
        baseline_config['dataset'] = config.dataset
        baseline_config['profile'] = profile

        baselines_orchestrator = BaselinesOrchestrator(logger, baseline_config)

        if os.path.exists(path_train) and os.path.exists(path_test) and os.path.exists(path_validation):
            logger.info("Baselines on clean data")
            results = baselines_orchestrator.apply(path_train,path_test,clean=True)
            print(results)
        if os.path.exists(path_train_noisy) and os.path.exists(path_test) and os.path.exists(path_validation):
            logger.info("Baselines on noisy data")
            results = baselines_orchestrator.apply(path_train_noisy, path_test)
            print(results)
        elif os.path.exists(path_train_noisy) and os.path.exists(path_test_noisy) and os.path.exists(path_validation_noisy):
            logger.info("Baselines on noisy data")
            results = baselines_orchestrator.apply(path_train_noisy, path_test_noisy)
            print(results)
        else:
            raise ValueError("The train.csv and test.csv are mandatory in baselines mode. Please, set a split parameter in the config/config_base.yaml file")
    # 3. Create orchestrator e applica rumore
        # create a unique table
        create_unique_table(config.dataset)

    # 4. Salva output

    exit(0)

if __name__ == "__main__":

    main()
