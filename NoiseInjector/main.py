import argparse
import pandas as pd
from data_handler.loader import DatasetLoader
#from data_handler.saver import save_dataset
from orchestrator import NoiseOrchestrator
from config.config import load_config
import os
from logger import get_logger, logging
from datetime import datetime
import random
import numpy as np
import torch


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

def main():
    parser = argparse.ArgumentParser(description="Noise Injector for Reviews")

    parser.add_argument(
        "--profile",
        choices=["rating", "review", "combined"],
        default="rating",
        help="Noise profile",
    )

    parser.add_argument(
        "--noise_injection",
        default='timestamp_corruption',
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

    valid = (
        (profile == "rating" and noise_injection in rating_options)
        or (profile == "review" and noise_injection in review_options)
        or (profile == "combined" and noise_injection in combined_options)
    )

    if not valid:
        parser.error(
            f"Invalid noise_injection '{noise_injection}' for profile '{profile}'.\n"
            f"Valid options:\n"
            f"  rating:   {sorted(rating_options)}\n"
            f"  review:   {sorted(review_options)}\n"
            f"  combined: {sorted(combined_options)}"
        )

    log_file=f"{BASE_DIR}/logs/noise_injector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = get_logger(log_file=log_file, level=logging.INFO)
    config = load_config(CONFIG_PATH, profile=profile, context=noise_injection)
    set_seed(config.random_seed)

    loader = DatasetLoader(logger=logger,config=config)

    # todo: do it also with items
    # empty dataframe
    df = pd.DataFrame()
    if config.input.reviews['format'] == 'jsonl':
        df = loader.load_jsonl(f"{BASE_DIR}/data/input/{config.dataset}/{config.input.reviews['file_name']}.{config.input.reviews['format']}")
    elif config.input.reviews['format'] == 'json':
        df = loader.load_json(f"{BASE_DIR}/data/input/{config.dataset}/{config.input.reviews['file_name']}.{config.input.reviews['format']}")
    elif config.input.reviews['format'] == 'csv':
        df = loader.load_csv(f"{BASE_DIR}/data/input/{config.dataset}/{config.input.reviews['file_name']}.{config.input.reviews['format']}")


    # 3. Create orchestrator e applica rumore
    orchestrator = NoiseOrchestrator(logger,config)
    df_noisy,modified = orchestrator.apply(df)

    # 4. Salva output
    loader.save_csv(df_noisy,modified)
    logger.info(f"noisy df saved.")
    exit(0)

if __name__ == "__main__":
    main()
