from cornac.data import Dataset, Reader
import certifi
from cornac.eval_methods import RatioSplit, BaseMethod
from cornac.metrics import *
from collections import Counter
import pandas as pd
from cornac.data import ReviewModality,TextModality
from cornac.data.text import BaseTokenizer
import os
import cornac
from datetime import datetime
from cornac.data import ReviewModality
from NoiseInjector.baselines.quality_metrics import *
from NoiseInjector.baselines.cornac.mapper import *
import pandas as pd
from NoiseInjector.baselines.base import BaseBaselineRunner

BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/

class CornacExperimentRunner(BaseBaselineRunner):
    def __init__(self,config, clean):
        self.config = config
        self.clean = clean
        self.dataset = self.config['dataset']
        self.profile = self.config['profile']
        self.quality = config['quality']

    def run(self,training_path,test_path):
        train_data = Reader().read(training_path, sep=',', skip_lines=1)
        test_data = Reader().read(test_path, sep=',', skip_lines=1)
        all_items = pd.read_csv(training_path)['item_id'].unique().tolist() + pd.read_csv(test_path)['item_id'].unique().tolist()
        all_items = list(set(all_items))# o lista di tutti gli item nel dataset
        interaction_counts = Counter(pd.read_csv(training_path)['item_id'].tolist() + pd.read_csv(test_path)['item_id'].tolist())


        eval_method = BaseMethod.from_splits(
            train_data=train_data,
            test_data=test_data,
            fmt='UIR',  # or your actual format
            exclude_unknowns=False,  # or True, depending on your setup
            verbose=True
        )

        if self.profile in ['review','hybrid']:
            review_modality_training, review_modality_test = self._load_reviews(training_path, test_path)
            item_modality_training, item_modality_test = self._load_items(training_path, test_path)
            eval_method.train_set.add_modalities(review_text=review_modality_training)
            eval_method.train_set.add_modalities(review_text=review_modality_training)
            eval_method.test_set.add_modalities(item_text=item_modality_training)
            eval_method.test_set.add_modalities(item_text=item_modality_test)

        metrics = self._load_metrics()
        methods = self._load_models()

        train_set = eval_method.train_set
        exp = cornac.Experiment(
            eval_method=eval_method,
            models=methods,
            metrics=metrics,
            user_based=True
        )
        exp.run()
        rows = []
        recommendations = {}

        for r in exp.result:  # r is a Result
            row = {'model': r.model_name}
            # r.metric_avg_results is an OrderedDict: {metric_name: value}
            row.update(r.metric_avg_results)
            model = next(m for m in exp.models if m.name == r.model_name)
            if self.quality: # quality solo su ranking
                coverage, novelty = self._compute_quality_metrics(model,train_set,all_items,interaction_counts)
                # Aggiorna la row con le nuove metriche
                row.update({'coverage': coverage, 'novelty': novelty})
            rows.append(row)

        # Now convert to DataFrame
        df = pd.DataFrame(rows).set_index('model')
        df = df.round(4)
        # Salviamo come TSV
        out_name = f"results_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv" if self.clean else f"results_noisy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
        if not os.path.exists(f"{BASE_DIR}/../results/{self.dataset}/{out_name}"):
            os.makedirs(f"{BASE_DIR}/../results/{self.dataset}",exist_ok=True)
        df.to_csv(f"{BASE_DIR}/../results/{self.dataset}/{out_name}", sep='\t', index=True)


        return df

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



    def _load_data(self,df_path):
        return Reader().read(df_path, sep=',',skip_lines=1)

    def _load_models(self):
        models = []
        for m in self.config['cornac']['models']:
            cls = MODEL_MAP[m['name']]
            params = m.get('params', {})
            try:
                models.append(cls(**params))
            except: # if the model does not exsit in cornac
                pass
        return models


    def _load_metrics(self):
        metrics = []
        for m in self.config['cornac'].get('metrics', []):
            for metric_name, params in m.items():
               # if metric_name in ['Precision','NDCG','Recall','NCRR','MRR','MAP','HitRatio','FMeasure','AUC']:
                #    self.quality = True
                try:
                    metrics.append(METRIC_MAP[metric_name](**params))
                except: # if the metrics does not exist in cornac
                    pass
        return metrics

    def _load_reviews(self,training_path,test_path):

        reviews = []

        train_df = pd.read_csv(training_path)
        test_df = pd.read_csv(test_path)

        # Togli le righe senza review
        #train_df = train_df.dropna(subset=['review_text'])

        # Costruisci la lista di tuple in modo vettoriale
        reviews_training = list(zip(
            train_df['user_id'].astype(str),
            train_df['item_id'].astype(str),
            train_df['review_text']
        ))
        reviews_test = list(zip(
            test_df['user_id'].astype(str),
            test_df['item_id'].astype(str),
            test_df['review_text']
        ))

        review_modality_training = ReviewModality(
            data=reviews_training,
            tokenizer=BaseTokenizer(stop_words="english"),
            max_vocab=4000,
            max_doc_freq=0.5,
        )
        review_modality_test = ReviewModality(
            data=reviews_test,
            tokenizer=BaseTokenizer(stop_words="english"),
            max_vocab=4000,
            max_doc_freq=0.5,
        )

        return review_modality_training,review_modality_test

    def _load_items(self,training_path,test_path):

        reviews = []

        train_df = pd.read_csv(training_path)
        test_df = pd.read_csv(test_path)

        aggregated_reviews_training = (
            train_df
            .groupby("item_id")["review_text"]
            .apply(lambda texts: " ".join(str(t) for t in texts))
            .reset_index()
        )
        aggregated_reviews_test = (
            test_df
            .groupby("item_id")["review_text"]
            .apply(lambda texts: " ".join(str(t) for t in texts))
            .reset_index()
        )

        # --- Step 3: crea TextModality dalle review aggregate ---
        item_text_modality_training = TextModality(
            corpus=aggregated_reviews_training["review_text"].tolist(),
            ids=aggregated_reviews_training["item_id"].tolist(),
            tokenizer=BaseTokenizer(stop_words="english"),
            max_vocab=8000,
            max_doc_freq=0.5
        )
        item_text_modality_test = TextModality(
            corpus=aggregated_reviews_test["review_text"].tolist(),
            ids=aggregated_reviews_test["item_id"].tolist(),
            tokenizer=BaseTokenizer(stop_words="english"),
            max_vocab=8000,
            max_doc_freq=0.5
        )




        return item_text_modality_training,item_text_modality_test
