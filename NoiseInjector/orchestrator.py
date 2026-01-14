from NoiseInjector.injectors.rating.rating_injector import RatingNoiseInjector
from logger import get_logger, logging
from datetime import datetime

class NoiseOrchestrator:
    def __init__(self, config):
        self.config = config
        log_file = f"../logs/noise_injector_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        self.logger = get_logger(log_file=log_file, level=logging.INFO)

# ricordaris kcore
    def apply(self, df):
        obj = RatingNoiseInjector(self.config,self.logger)
        if self.config["noise_profile"] == 'rating':
            obj = RatingNoiseInjector(self.config,self.logger)
        if self.config["noise_profile"] == 'review':
            obj = ReviewNoiseInjector(self.config,self.logger)
        if self.config["noise_profile"] == 'combined':
            obj = RatingNoiseInjector(self.config,self.logger)
        df = obj.apply_noise(df)

        return df