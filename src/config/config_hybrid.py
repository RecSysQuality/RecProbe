from dataclasses import dataclass, field
from typing import Optional
import yaml
from dacite import from_dict
#from numpy.core.numeric import infty


@dataclass
class RatingBehaviorConfig:
    min_rating: int = 1
    max_rating: int = 5
    sampling_strategy: str = "gaussian"  # gaussian | uniform


# =========================
# Temporal interval
# =========================

@dataclass
class TemporalIntervalConfig:
    start_timestamp: int = 1609459200       #  uniform | forward | backward
    end_timestamp: int = 1640995200              # low | medium | high


# =========================
# Timestamp corruption config
# =========================
@dataclass
class RatingReviewNoiseConfig:
    target: str = "item"                # user | item | NA
    operation: str = "corrupt"
    modify: str = "rating"              # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    max_reviews_per_node: float = 50
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

@dataclass
class RatingReviewNoiseConfig:
    target: str = "item"                # user | item | NA
    operation: str = "corrupt"
    modify: str = "rating"              # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    max_reviews_per_node: float = 50
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    model: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

@dataclass
class SemanticDriftNoiseConfig:
    target: str = "item"                # user | item | NA
    operation: str = "corrupt"
    selection_strategy: str = "uniform" # uniform | popularity_based
    max_reviews_per_node: float = 50
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    model: str = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_behavior: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

# ===
# =========================
# Main Rating Config
# =========================
@dataclass
class RatingReviewConfig:
    context: str = "realistic_noise"   # realistic_noise | user_burst_noise | item_burst_noise | timestamp_corruption
    budget: int = 5000
    avoid_duplicates: bool = True
    semantic_drift: Optional[SemanticDriftNoiseConfig] = None
    random_inconsistencies: Optional[RatingReviewNoiseConfig] = None
    hybrid_burst: Optional[RatingReviewNoiseConfig] = None


# =========================
# Loader YAML
# =========================
def load_hybrid_config(path: str = "files/config_hybrid.yaml",context: str = 'random_inconsistencies',streamlit: bool = False) -> RatingReviewConfig:
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

    return from_dict(
        data_class=RatingReviewConfig,
        data=cfg_dict
    )

# =========================
# Test
# =========================
if __name__ == "__main__":
    rating_cfg = load_hybrid_config("files/config_hybrid.yaml")
    print(rating_cfg)
