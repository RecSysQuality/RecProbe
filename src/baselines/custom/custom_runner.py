from cornac.data import Dataset, Reader
import certifi
from cornac.eval_methods import RatioSplit, BaseMethod
from cornac.metrics import *
from collections import Counter
import os
import cornac
from NoiseInjector.baselines.quality_metrics import *
from NoiseInjector.baselines.cornac.mapper import *
import pandas as pd
from NoiseInjector.baselines.base import BaseBaselineRunner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/

class CustomExperimentRunner(BaseBaselineRunner):
    def __init__(self,config, clean):
        self.config = config
        self.clean = clean
        self.dataset = self.config['dataset']
        self.quality = config['quality']

    def run(self):
        results = {}

        return results



    def save_results(self):
        return

    def _compute_quality_metrics(self,model,train_set,all_items,interaction_counts):
        # Costruisci raccomandazioni top-10 per utente
        recommendations_model = {}
        for user_id in model.user_ids:
            recs = model.recommend(
                user_id=user_id,
                k=10,
                remove_seen=True,
                train_set=train_set  # assicurati che train_data sia disponibile
            )
            recommendations_model[user_id] = recs

        coverage = compute_coverage(recommendations_model, all_items)
        novelty = compute_novelty(recommendations_model, interaction_counts, all_items)
        return coverage,novelty

