from abc import ABC, abstractmethod
import pandas as pd

class BaseNoiseInjector(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def apply_noise(self, df: pd.DataFrame, df_val: pd.DataFrame, df_test: pd.DataFrame) -> pd.DataFrame:
        """Applica rumore a livello metadata"""
        pass


