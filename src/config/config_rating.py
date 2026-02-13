from dataclasses import dataclass, field
from typing import Optional
import yaml
from dacite import from_dict
#from numpy.core.numeric import infty


# =========================
# Node selection / Distribution
# =========================
@dataclass
class NodeSelectionConfig:
    target: str = "user"                # user | item | NA
    strategy: str = "uniform"           # uniform | top | least
    popularity_bias: str = "uniform"    # uniform | top | least
    k: int = 100

@dataclass
class NodeLimitsConfig:
    max_ratings_per_node: int = -1      # -1 = no limit
    min_ratings_per_node: int = 1

@dataclass
class DistributionConfig:
    node_selection: NodeSelectionConfig = field(default_factory=NodeSelectionConfig)
    per_node_limits: NodeLimitsConfig = field(default_factory=NodeLimitsConfig)

# =========================
# Rating behavior
# =========================

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
    corruption_mode: str = "uniform"       #  uniform | forward | backward
    intensity: str = "low"              # low | medium | high

# =========================
# Temporal interval
# =========================

@dataclass
class TemporalIntervalConfig:
    start_timestamp: int = 1609459200       #  uniform | forward | backward
    end_timestamp: int = 1640995200              # low | medium | high
# =========================
# User Burst Noise
# =========================
@dataclass
class UserBurstNoiseConfig:
    target: str = "user"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    #hubs: int = 100
    operation: str = "add"           # remove | add
    max_ratings_per_node: float = float("inf")
    min_ratings_per_node: float = 1
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_interval: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)


@dataclass
class RealisticNoiseConfig:
    target: str = "user"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    #k: int = 100
    operation: str = "add"           # remove | add
    preserve_degree_distribution: bool = True
    max_ratings_per_node: float = float("inf")
    min_ratings_per_node: float = 1
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_interval: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

# =========================
# Item Burst Noise
# =========================
@dataclass
class ItemBurstNoiseConfig:
    target: str = "item"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    #hubs: int = 100
    operation: str = "add"           # remove | add
    max_ratings_per_node: float = float("inf")
    min_ratings_per_node: float = 1
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_interval: TemporalIntervalConfig = field(default_factory=TemporalIntervalConfig)

# =========================
# Timestamp corruption config
# =========================
@dataclass
class TimestampCorruptionConfig:
    target: str = "item"                # user | item | NA
    selection_strategy: str = "uniform" # uniform | popularity_based
    #k: int = 100
    operation: str = "add"           # remove | add
    max_ratings_per_node: float = "inf"
    min_ratings_per_node: float = 1
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_behavior: TemporalBehaviorConfig = field(default_factory=TemporalBehaviorConfig)

# =========================
# Main Rating Config
# =========================
@dataclass
class RatingConfig:
    context: str = "realistic_noise"   # realistic_noise | user_burst_noise | item_burst_noise | timestamp_corruption
    budget: int = 5000
    avoid_duplicates: bool = True

    realistic_noise: Optional[RealisticNoiseConfig] = None
    user_burst_noise: Optional[UserBurstNoiseConfig] = None
    item_burst_noise: Optional[ItemBurstNoiseConfig] = None
    timestamp_corruption: Optional[TimestampCorruptionConfig] = None

# =========================
# Loader YAML
# =========================
def load_rating_config(path: str = "files/config_rating.yaml",streamlit: bool = False) -> RatingConfig:
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

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
        data_class=RatingConfig,
        data=cfg_dict
    )

# =========================
# Test
# =========================
if __name__ == "__main__":
    rating_cfg = load_rating_config("files/config_rating.yaml")
    print(rating_cfg)
