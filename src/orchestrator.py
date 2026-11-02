from injectors.general_injector import GeneralNoiseInjector
from injectors.interactions.interaction_injector import InteractionNoiseInjector
from injectors.interactions.rating_injector import RatingNoiseInjector
from injectors.interactions.review_injector import ReviewNoiseInjector
from injectors.registry import get_perturbation

INJECTORS = {
    "interaction": InteractionNoiseInjector,   # Table 1, row A
    "rating": RatingNoiseInjector,             # Table 1, row B
    "review": ReviewNoiseInjector,             # Table 1, row C
    "general": GeneralNoiseInjector,           # removal / metadata (outside the taxonomy)
}


class NoiseOrchestrator:
    """Selects the row injector of the configured perturbation and applies it."""

    def __init__(self, logger, config, items=None, rating_scale=None):
        self.config = config
        self.logger = logger
        self.items = items
        self.rating_scale = rating_scale

    def apply(self, df, df_val=None, df_test=None):
        signal = get_perturbation(self.config.injection.perturbation).signal
        injector = INJECTORS[signal](self.config, self.logger, self.items, self.rating_scale)
        return injector.apply_noise(df, df_val, df_test)
