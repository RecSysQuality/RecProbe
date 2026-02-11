from abc import ABC, abstractmethod
import pandas as pd

class BaseBaselineRunner(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def run(self,training_path,test_path) -> pd.DataFrame:
        """Applica rumore a livello metadata"""
        pass


    # @abstractmethod
    # def save_results(self,training_path,test_path) -> pd.DataFrame:
    #     """Applica rumore a livello metadata"""
    #     pass

