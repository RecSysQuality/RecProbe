from NoiseInjector.baselines.cornac.cornac_runner import CornacExperimentRunner
from NoiseInjector.injectors.interactions.combined_injector import CombinedNoiseInjector
from NoiseInjector.injectors.interactions.rating_injector import RatingNoiseInjector
from NoiseInjector.injectors.interactions.review_injector import ReviewNoiseInjector
#from NoiseInjector.injectors.interactions.review_injector import ReviewNoiseInjector
#from NoiseInjector.injectors.interactions.combined_injector import CombinedNoiseInjector
from logger import get_logger, logging
from datetime import datetime

class BaselinesOrchestrator:
    def __init__(self,logger, config):
        self.config = config
        self.logger = logger

    def apply(self, df_train_path,df_test_path,clean=False):

        cornac_obj = CornacExperimentRunner(self.config,clean)
        results = cornac_obj.run(df_train_path,df_test_path)

        return results