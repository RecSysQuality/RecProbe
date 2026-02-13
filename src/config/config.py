import yaml
from dataclasses import dataclass
from typing import Optional, Union

from tokenizers.pre_tokenizers import Split

from .config_rating import RatingConfig, load_rating_config
from .config_reviews import ReviewConfig, load_review_config
from .config_combined import RatingReviewConfig, load_hybrid_config
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------ DATACLASSES ------------------

@dataclass
class SplitConfig:
    training: float
    validation: float
    test: float
    strategy: str
    noise_in_test: bool


@dataclass
class SingleFileConfig:
    name: str
    format: str
    separator: str = ","

@dataclass
class InputConfig:
    reviews: Optional[SingleFileConfig] = None
    items: Optional[SingleFileConfig] = None


@dataclass
class OutputConfig:
    reviews: Optional[SingleFileConfig] = None
    items: Optional[SingleFileConfig] = None


@dataclass
class Config:
    input: InputConfig
    output: OutputConfig
    split: SplitConfig
    drop_duplicates: bool = True
    noise_profile: str = "rating"
    noise_context: str = "realistic_noise"
    kcore: Optional[int] = 5
    min_rating: Optional[int] = 1
    min_review_length: Optional[int] = 0
    random_seed: int = 42
    verbose: bool = False
    dataset: str = "amazon_All_Beauty"
    noise_config: Optional[
        Union[RatingConfig, ReviewConfig, RatingReviewConfig]
    ] = None


# ------------------ LOADER ------------------
import os
def load_config(
    path: str = "files/config_base.yaml",
    profile: str = "rating",
    context: str = "realistic_noise"
) -> Config:


    with open(path) as f:
        cfg_dict = yaml.safe_load(f)

    input_cfg = InputConfig(**cfg_dict["input"])
    output_cfg = OutputConfig(**cfg_dict["output"])
    split_cfg = SplitConfig(**cfg_dict["split"])
    if profile:
        noise_profile = profile
    else:
        noise_profile = cfg_dict.get("noise_profile", profile)

    if noise_profile == "rating":
        path_noise = f"{BASE_DIR}/files/config_rating.yaml"
        noise_config = load_rating_config(path_noise)
    elif noise_profile == "review":
        path_noise = f"{BASE_DIR}/files/config_review.yaml"
        noise_config = load_review_config(path_noise)
    elif noise_profile == "hybrid":
        path_noise = f"{BASE_DIR}/files/config_hybrid.yaml"
        noise_config = load_hybrid_config(path_noise)
    else:
        raise ValueError(f"Unknown noise_profile: {noise_profile}")
    if context:
        noise_config.context = context

    return Config(
        input=input_cfg,
        output=output_cfg,
        noise_profile=noise_profile,
        noise_context=context,
        split=split_cfg,
        random_seed=cfg_dict.get("random_seed", 42),
        kcore=cfg_dict.get("kcore", 5),
        min_rating=cfg_dict.get("min_rating", 1),
        min_review_length=cfg_dict.get("min_review_length", 0),
        verbose=cfg_dict.get("verbose", False),
        dataset=cfg_dict.get("dataset"),
        noise_config=noise_config,
    )

def load_streamlit_config(
    path: str = "streamlit/generated_configs/config.yaml",
    profile: str = "rating",
    context: str = "realistic_noise"
) -> Config:


    with open(path) as f:
        cfg_dict = yaml.safe_load(f)

    input_cfg = InputConfig(**cfg_dict["input"])
    output_cfg = OutputConfig(**cfg_dict["output"])
    split_cfg = SplitConfig(**cfg_dict["split"])
    if profile:
        noise_profile = profile
    else:
        noise_profile = cfg_dict.get("noise_profile", profile)

    if noise_profile == "rating":
        noise_config = load_rating_config(path,streamlit=True)
    elif noise_profile == "review":
        noise_config = load_review_config(path,streamlit=True)
    elif noise_profile == "hybrid":
        noise_config = load_hybrid_config(path,streamlit=True)
    else:
        raise ValueError(f"Unknown noise_profile: {noise_profile}")
    if context:
        noise_config.context = context

    return Config(
        input=input_cfg,
        output=output_cfg,
        noise_profile=noise_profile,
        noise_context=context,
        split=split_cfg,
        random_seed=cfg_dict.get("random_seed", 42),
        kcore=cfg_dict.get("kcore", 5),
        min_rating=cfg_dict.get("min_rating", 1),
        min_review_length=cfg_dict.get("min_review_length", 0),
        verbose=cfg_dict.get("verbose", False),
        dataset=cfg_dict.get("dataset"),
        noise_config=noise_config,
    )

# ------------------ DEBUG ------------------

if __name__ == "__main__":
    config = load_config()
    print(config)
