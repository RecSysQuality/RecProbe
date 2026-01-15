from NoiseInjector.injectors.interactions.rating_injector import RatingNoiseInjector
#from NoiseInjector.injectors.interactions.review_injector import ReviewNoiseInjector
#from NoiseInjector.injectors.interactions.combined_injector import CombinedNoiseInjector
from logger import get_logger, logging
from datetime import datetime

class NoiseOrchestrator:
    def __init__(self,logger, config):
        self.config = config
        self.logger = logger

    def apply(self, df):
        obj = RatingNoiseInjector(self.logger,self.config)
        np = self.config.noise_profile
        if np == 'rating':
            obj = RatingNoiseInjector(self.logger,self.config)
        elif np == 'review':
            obj = RatingNoiseInjector(self.config,self.logger)
        elif np == 'combined':
            obj = RatingNoiseInjector(self.config,self.logger)
        df,modified = obj.apply_noise(df)

        return df