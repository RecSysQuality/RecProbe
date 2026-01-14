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
# Temporal behavior
# =========================
@dataclass
class TemporalBehaviorConfig:
    start_timestamp: int = 1609459200       #  uniform | forward | backward
    end_timestamp: int = 1640995200             # low | medium | high

# =========================
# Temporal interval
# =========================

@dataclass
class TemporalIntervalConfig:
    start_timestamp: int = 1609459200       #  uniform | forward | backward
    stop_timestamp: int = 1640995200              # low | medium | high

@dataclass
class RemoveRevNoiseConfig:
    target: str = "user"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    operation: str = "remove"           # remove | add
    preserve_degree_distribution: bool = True
    max_reviews_per_node: float = float("inf")
    min_reviews_per_node: float = 1
    min_length_of_review: float = 10
    #rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_interval: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

# =========================
# Item Burst Noise
# =========================
@dataclass
class ReviewBurstNoiseConfig:
    target: str = "item"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    operation: str = "add"           # remove | add
    max_reviews_per_node: float = float("inf")
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_interval: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

# =========================
# Timestamp corruption config
# =========================
@dataclass
class SentecneNoiseConfig:
    target: str = "item"                # user | item | NA
    operation: str = "corrupt"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
          # remove | add
    max_reviews_per_node: float = "inf"
    min_reviews_per_node: float = 1
    min_length_of_review: int = 10
    noise_type: str = 'shuffle'
    vocabulary_file: str = None
    temporal_behavior: TemporalBehaviorConfig = field(default_factory=TemporalBehaviorConfig)

# =========================
# Main Rating Config
# =========================
@dataclass
class RatingConfig:
    context: str = "realistic_noise"   # realistic_noise | user_burst_noise | item_burst_noise | timestamp_corruption
    total_budget: int = 5000
    avoid_duplicates: bool = True

    remove_reviews: Optional[RemoveRevNoiseConfig] = None
    review_burst_noise: Optional[ReviewBurstNoiseConfig] = None
    sentence_noise: Optional[SentecneNoiseConfig] = None

# =========================
# Loader YAML
# =========================
def load_rating_config(path: str = "files/config_rating.yaml") -> RatingConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    return from_dict(
        data_class=RatingConfig,
        data=cfg_dict
    )

# =========================
# Test
# =========================
if __name__ == "__main__":
    rating_cfg = load_rating_config("files/config_review.yaml")
    print(rating_cfg)
