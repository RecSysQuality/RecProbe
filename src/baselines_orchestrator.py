from src.baselines.cornac.cornac_runner import CornacExperimentRunner
from src.baselines.recbole.recbole_runner import RecBoleExperimentRunner
#from src.baselines.cornac.custom_runner import CustomExperimentRunner
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/
import yaml

class BaselinesOrchestrator:
    def __init__(self,logger,config_path = f"{BASE_DIR}/cornac/config/config.yaml",profile='rating',dataset='yelp',framework='cornac'):

        with open(config_path) as f:
            cfg_dict = yaml.safe_load(f)
        config = cfg_dict
        config['dataset'] = dataset
        config['profile'] = profile
        self.config = config
        self.logger = logger
        self.fw = framework

    def apply(self, df_train_path,df_validation_path,df_test_path,clean=False):
        if self.fw == 'cornac':
            cornac_obj = CornacExperimentRunner(self.config,clean)
            results = cornac_obj.run(df_train_path,df_validation_path,df_test_path)
        elif self.fw == 'recbole':
            recbole_obj = RecBoleExperimentRunner(self.config, clean)
            results = recbole_obj.run(df_train_path, df_validation_path,df_test_path)
        elif self.fw == 'custom':
            custom_obj = RecBoleExperimentRunner(self.config,clean)
            results = custom_obj.run(df_train_path,df_validation_path,df_test_path)
        return results
