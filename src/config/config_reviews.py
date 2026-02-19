from dataclasses import dataclass, field
from typing import Optional
import yaml
from dacite import from_dict
#from numpy.core.numeric import infty

from typing import List

@dataclass
class RatingBehaviorConfig:
    min_rating: int = 1
    max_rating: int = 5
    sampling_strategy: str = "gaussian"  # gaussian | uniform


@dataclass
class NearDuplicatesConfig:
    model: str = 'ramsrigouthamg/t5_paraphraser'
    rating: float = 4
    review: Optional[str] = None
    title: Optional[str] = None




# =========================
# Temporal interval
# =========================

@dataclass
class TemporalIntervalConfig:
    start_timestamp: int = 1609459200       #  uniform | forward | backward
    end_timestamp: int = 1640995200              # low | medium | high

@dataclass
class RealisticNoiseConfig:
    target: str = "user"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    operation: str = "remove"           # remove | add
    preserve_degree_distribution: bool = True
    max_reviews_per_node: float = float("inf")
    min_reviews_per_node: float = 1
    min_length_of_review: float = 10
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    near_duplicates_configuration: NearDuplicatesConfig = field(default_factory=NearDuplicatesConfig)

# =========================
# Item Burst Noise
# =========================
@dataclass
class ReviewBurstNoiseConfig:
    target: str = "item"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    operation: str = "add"           # remove | add
    model: str = "t5-base"
    max_reviews_per_node: float = float("inf")
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)
    near_duplicates_configuration: NearDuplicatesConfig = field(default_factory=NearDuplicatesConfig)

# =========================
# Timestamp corruption config
# =========================
@dataclass
class SentecneNoiseConfig:
    target: str = "item"                # user | item | NA
    operation: str = "corrupt"                # user | item | NA
    model: str = "t5-base"
    selection_strategy: str = "uniform" # uniform | popularity_based
          # remove | add
    max_reviews_per_node: float = "inf"
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    noise_type: str = 'shuffle'
    intensity: str = 'low'
   # vocabulary_file: str = None
    vocabulary: List[str] = field(default_factory=lambda: [
        "shipping", "breakfast", "home", "night", "sun",
        "dinner", "lunch", "dog", "pets"
    ])
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)

# =========================
# Main Rating Config
# =========================
@dataclass
class ReviewConfig:
    context: str = "random_inconsistencies"   # realistic_noise | user_burst_noise | item_burst_noise | timestamp_corruption
    budget: int = 5000
    avoid_duplicates: bool = True

    random_inconsistencies: Optional[RealisticNoiseConfig] = None
    review_burst: Optional[ReviewBurstNoiseConfig] = None
    sentence_noise: Optional[SentecneNoiseConfig] = None

# =========================
# Loader YAML
# =========================

def load_review_config(path: str = "files/config_review.yaml",context: str = 'random_inconsistencies',streamlit: bool = False) -> ReviewConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)
        cfg_dict['context'] = context
    if streamlit:
        # Trova l'indice di noise_profile nell'ordine originale
        keys = list(cfg_dict.keys())
        start_index = keys.index("noise_profile")

        # Crea un nuovo dict solo dalla chiave noise_profile in poi
        filtered_dict = {
            k: cfg_dict[k] for k in keys[start_index:]
        }

        cfg_dict = filtered_dict
    if streamlit:
        # Trova l'indice di noise_profile nell'ordine originale
        keys = list(cfg_dict.keys())
        start_index = keys.index("noise_profile")

        # Crea un nuovo dict solo dalla chiave noise_profile in poi
        filtered_dict = {
            k: cfg_dict[k] for k in keys[start_index:]
        }

        cfg_dict = filtered_dict

    return from_dict(
        data_class=ReviewConfig,
        data=cfg_dict
    )

# =========================
# Test
# =========================
if __name__ == "__main__":
    rating_cfg = load_review_config("files/config_review.yaml")
    print(rating_cfg)
