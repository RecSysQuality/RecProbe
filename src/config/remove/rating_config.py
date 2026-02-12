from dataclasses import dataclass, field
from typing import Optional
import yaml
from dacite import from_dict

# =========================
# Distribution
# =========================
@dataclass
class NodeSelectionConfig:
    target: str = "user"                # user | item
    strategy: str = "uniform"          # uniform | popularity_based
    popularity_bias: str = "uniform"   # uniform | top | least
    k: int = 100

@dataclass
class NodeLimitsConfig:
    max_ratings_per_node: int = 3
    min_ratings_per_node: int = 1


@dataclass
class DistributionRatingConfig:
    node_selection: NodeSelectionConfig = field(default_factory=NodeSelectionConfig)
    per_node_limits: NodeLimitsConfig = field(default_factory=NodeLimitsConfig)


# =========================
# Structure (bot-only)
# =========================
@dataclass
class StructureRatingConfig:
    hub_strategy: str = "none"          # none | few_hubs | many_hubs
    user_hubs: int = 0
    item_hubs: int = 0
    hub_concentration: float = 0.0


# =========================
# Rating behavior
# =========================
@dataclass
class ValueDistributionConfig:
    type: str = "gaussian"              # gaussian | uniform | empirical
    clip: bool = True


@dataclass
class RatingBehaviorConfig:
    mean_rating: float = 3.6
    std_rating: float = 0.7
    min_rating: int = 1
    max_rating: int = 5
    value_distribution: ValueDistributionConfig = field(default_factory=ValueDistributionConfig)


# =========================
# Temporal behavior
# =========================
@dataclass
class TemporalBehaviorConfig:
    timestamp_noise: str = "low"        # none | low | high
    temporal_spread: str = "wide"       # wide | narrow


# =========================
# Constraints
# =========================
@dataclass
class RatingConstraintsConfig:
    avoid_duplicates: bool = True
    preserve_degree_distribution: bool = True
    respect_user_activity_profile: bool = True
    respect_item_popularity_distribution: bool = True


# =========================
# Main Rating Config
# =========================
@dataclass
class RatingConfig:
#    target: str = "user"                # user | item | global
    user_item_ratio: float = 0.0        # only if target == global
    profile: str = "human"              # human | bot

    operation: str = "add"              # add | remove
    timestamp_corruption: bool = False
    total_budget: int = 5000

    distribution: DistributionRatingConfig = field(default_factory=DistributionRatingConfig)
    structure: StructureRatingConfig = field(default_factory=StructureRatingConfig)
    rating_behavior: RatingBehaviorConfig = field(default_factory=RatingBehaviorConfig)
    temporal_behavior: TemporalBehaviorConfig = field(default_factory=TemporalBehaviorConfig)
    constraints: RatingConstraintsConfig = field(default_factory=RatingConstraintsConfig)


# =========================
# Loader YAML
# =========================
def load_rating_config(path: str = "config_rating.yaml") -> RatingConfig:
    """
    Carica un config YAML per il noise injector dei rating
    e restituisce un oggetto RatingConfig completamente tipizzato.
    """
    with open(path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Conversione automatica da dict a dataclass annidata
    return from_dict(
        data_class=RatingConfig,
        data=cfg_dict
    )


# =========================
# Test
# =========================
if __name__ == "__main__":
    rating_cfg = load_rating_config("config_rating.yaml")
    print(rating_cfg)
