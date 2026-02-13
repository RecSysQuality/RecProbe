from src.baselines.cornac.cornac_runner import CornacExperimentRunner
# from src.baselines.cornac.recbole_runner import RecBoleExperimentRunner
# from src.baselines.cornac.custom_runner import CustomExperimentRunner


class BaselinesOrchestrator:
    def __init__(self,logger, config):
        self.config = config
        self.logger = logger

    def apply(self, df_train_path,df_validation_path,df_test_path,framework,clean=False):
        cornac_obj = CornacExperimentRunner(self.config,clean)
        results_cornac = cornac_obj.run(df_train_path,df_test_path)
        results_custom, results_recbole = None,None
        #recbole_obj = RecBoleExperimentRunner(self.config, clean)
        #results_recbole = recbole_obj.run(df_train_path, df_test_path)

        #custom_obj = CustomExperimentRunner(self.config,clean)
        #results_custom = custom_obj.run(df_train_path,df_test_path)
        return results_cornac,results_recbole,results_custom