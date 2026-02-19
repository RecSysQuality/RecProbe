from cornac.data import Dataset, Reader
import certifi
from cornac.eval_methods import RatioSplit, BaseMethod
from cornac.metrics import *
from collections import Counter
import os
import cornac
import pandas as pd

from src.baselines.base import BaseBaselineRunner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/

class CustomExperimentRunner(BaseBaselineRunner):
    def __init__(self,config, clean):
        self.config = config
        self.clean = clean
        self.dataset = self.config['dataset']
        self.quality = config['quality']

    def run(self):
        results_df = pd.DataFrame()


        # SAVE IN SRC/BASELINES/RESULTS A CSV WITH THE RESULTS WITH MODEL IN ROWS AND METRICS IN COLUMNS

        return results_df



    def _save_results(self):
        return



