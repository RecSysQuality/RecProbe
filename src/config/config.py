import yaml
from dataclasses import dataclass
from typing import Optional, Union

from .config_rating import RatingConfig, load_rating_config
from .config_reviews import ReviewConfig, load_review_config
from .config_combined import RatingReviewConfig, load_combined_config
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# ------------------ DATACLASSES ------------------

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
    target: Optional[str] = None
    split: Optional[float] = 0.8


@dataclass
class Config:
    input: InputConfig
    output: OutputConfig
    drop_duplicates: bool = True
    noise_profile: str = "rating"
    noise_context: str = "realistic_noise"
    kcore: Optional[int] = 5
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
    elif noise_profile == "combined":
        path_noise = f"{BASE_DIR}/files/config_combined.yaml"
        noise_config = load_combined_config(path_noise)
    else:
        raise ValueError(f"Unknown noise_profile: {noise_profile}")
    if context:
        noise_config.context = context

    return Config(
        input=input_cfg,
        output=output_cfg,
        noise_profile=noise_profile,
        noise_context=context,
        random_seed=cfg_dict.get("random_seed", 42),
        kcore=cfg_dict.get("kcore", 5),
        verbose=cfg_dict.get("verbose", False),
        dataset=cfg_dict.get("dataset"),
        noise_config=noise_config,
    )


# ------------------ DEBUG ------------------

if __name__ == "__main__":
    config = load_config()
    print(config)
