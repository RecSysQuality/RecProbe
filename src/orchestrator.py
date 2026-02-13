from injectors.interactions.hybrid_injector import HybridNoiseInjector
from injectors.interactions.rating_injector import RatingNoiseInjector
from injectors.interactions.review_injector import ReviewNoiseInjector
#from NoiseInjector.injectors.interactions.review_injector import ReviewNoiseInjector
#from NoiseInjector.injectors.interactions.combined_injector import CombinedNoiseInjector
from logger import get_logger, logging
from datetime import datetime

class NoiseOrchestrator:
    def __init__(self,logger, config):
        self.config = config
        self.logger = logger

    def apply(self, df, df_val, df_test):
        obj = RatingNoiseInjector(self.logger,self.config)
        np = self.config.noise_profile

        if np == 'rating':
            obj = RatingNoiseInjector(self.logger,self.config)
        elif np == 'review':
            obj = ReviewNoiseInjector(self.config,self.logger)
        elif np == 'hybrid':
            obj = HybridNoiseInjector(self.config,self.logger)
        df,modified = obj.apply_noise(df,df_val, df_test)

        return df,modified