import yaml
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dataclasses import dataclass, field
from typing import Optional, List, Union
from .config_rating import RatingConfig, load_rating_config

@dataclass
class SingleFileConfig:
    path: str
    format: str
    separator: str = ","  # default per CSV

@dataclass
class InputConfig:
    reviews: Optional[SingleFileConfig] = None
    items: Optional[SingleFileConfig] = None  # opzionale, se vuoi caricare items

@dataclass
class OutputConfig:
    reviews: Optional[SingleFileConfig] = None
    items: Optional[SingleFileConfig] = None
    target: Optional[str] = None  # percorso di output opzionale


@dataclass
class Config:
    input: InputConfig
    output: OutputConfig
    noise_profile: str = 'rating' # choices rating, review, combined
    kcore: Optional[int] = 5
    random_seed: int = 42
    verbose: bool = False
    dataset: Optional[str] = None
    noise_config: Optional[Union['RatingConfig']] = None


def load_config(path: str = "config_base.yaml",path_rating: str = "files/config_rating.yaml") -> Config:
    with open(path) as f:
        cfg_dict = yaml.safe_load(f)

    input_cfg = InputConfig(**cfg_dict["input"])
    output_cfg = OutputConfig(**cfg_dict["output"])

    if cfg_dict.get("noise_profile", 'rating') == 'rating':
        noise_config = load_rating_config(path_rating)

    elif cfg_dict.get("noise_profile") == 'review':
        noise_config = None
    elif cfg_dict.get("noise_profile") == 'combined':
        noise_config = None

    return Config(
        input=input_cfg,
        output=output_cfg,
        noise_profile=cfg_dict.get("noise_profile", 'rating'),
        random_seed=cfg_dict.get("random_seed", 42),
        kcore=cfg_dict.get("kcore", 5),
        verbose=cfg_dict.get("verbose", False),
        dataset=cfg_dict.get("dataset"),
        noise_config=noise_config
    )


if __name__ == "__main__":
    config = load_config("files/config_base.yaml")
    print(config)
