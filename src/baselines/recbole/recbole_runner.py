from cornac.data import Dataset, Reader
import certifi
from cornac.eval_methods import RatioSplit, BaseMethod
from recbole.quick_start import run_recbole
import pandas as pd
import os
from collections import Counter
from cornac.data import ReviewModality,TextModality
from cornac.data.text import BaseTokenizer
import os
import cornac
from datetime import datetime
from cornac.data import ReviewModality
from src.baselines.quality_metrics import *
from src.baselines.cornac.mapper import *
import pandas as pd
from src.baselines.base import BaseBaselineRunner
import os
import re
import csv
from pathlib import Path
from datetime import datetime
import pandas as pd


class RecBoleExperimentRunner(BaseBaselineRunner):
    def __init__(self,config, clean):
        self.config = config
        self.clean = clean
        self.dataset = self.config['dataset']

    def run(self,training_path,validation_path,test_path):

        # load training, validation, test

        train_data = pd.read_csv(training_path, sep=',')
        test_data = pd.read_csv(test_path, sep=',')
        validation_data = pd.read_csv(validation_path, sep=',')
        self._adapt_for_recbole(train_data,self.dataset,type="train")
        self._adapt_for_recbole(validation_data,self.dataset,type="valid")
        self._adapt_for_recbole(test_data,self.dataset,type="test")
        # --- Trova tutti i config YAML nella cartella config ---
        config_folder = os.path.join(os.getcwd(), "baselines/recbole/config")
        yaml_files = [os.path.join(config_folder, f) for f in os.listdir(config_folder) if f.endswith(".yaml")]
        parameter_dict = {
            # dataset config
            'benchmark_filename': ['train', 'valid', 'test'],
        }
        print('RUNNING')
        for yaml_file in yaml_files:
            run_recbole(
                dataset=self.dataset,
                config_file_list=[yaml_file],
                config_dict=parameter_dict
            )
        print('SAVING')

        self._save_results_csv()


    def _adapt_for_recbole(
            self,
            df,
            output_root="dataset",
            type="train"
    ):
        """
        Convert for RecBole
        """
        col_types = {
            "user_id": "token",
            "item_id": "token",
            "review": "token_seq",
            "rating": "float",
            "timestamp": "float"
        }
        # 1️⃣ Leggi CSV
        #df = pd.read_csv(dataset_path)
        df = df.rename(columns={"review_text": "review"})

        df = df[list(col_types.keys())]

        typed_header = [f"{col}:{col_types[col]}" for col in df.columns]

        dataset_dir = f"./dataset/{self.dataset}"
        os.makedirs(dataset_dir, exist_ok=True)

        output_file = os.path.join(dataset_dir, f"{self.dataset}.{type}.inter")

        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\t".join(typed_header) + "\n")
            df.to_csv(f, sep="\t", index=False, header=False)

        print(f"✓ Dataset converted for RecBole: {output_file}")
        return df

    def _save_results_csv(self):

        import ast
        import re
        from pathlib import Path
        from datetime import datetime
        import pandas as pd
        import os

        BASE_DIR = Path(os.getcwd())
        LOG_DIR = BASE_DIR / "baselines/recbole/log"

        # prendi tutti i file .log ricorsivamente
        log_files = sorted(LOG_DIR.rglob("*.log"))
        if not log_files:
            raise FileNotFoundError(f"Nessun file .log trovato in {LOG_DIR.resolve()}")

        rows = []

        for LOG_PATH in log_files:
            method_name = LOG_PATH.stem  # nome file senza estensione

            last_test_metrics = None  # prendiamo l'ultimo test result nel file

            with LOG_PATH.open("r", encoding="utf-8") as f:
                for line in f:
                    if "test result:" not in line.lower():
                        continue

                    match = re.search(r"OrderedDict\((.*)\)", line)
                    if not match:
                        continue

                    try:
                        metrics = dict(ast.literal_eval(match.group(1)))
                        last_test_metrics = metrics
                    except Exception:
                        continue

            # aggiungi solo se trovato almeno un test result
            if last_test_metrics:
                row = {"model": method_name.split('-')[0]}
                row.update(last_test_metrics)
                rows.append(row)

        if not rows:
            raise ValueError("No test result in log.")

        # DataFrame finale: riga = metodo, colonne = metriche
        df = pd.DataFrame(rows)
        df = df.sort_values("model")

        # directory risultati


        out_name = (
            f"results_recbole_clean_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
            if getattr(self, "clean", False)
            else f"results_recbole_noisy_{datetime.now().strftime('%Y%m%d_%H%M%S')}.tsv"
        )
        RES_DIR = os.path.dirname(os.path.abspath(__file__))  # punta a NoiseInjector/

        if not os.path.exists(f"{RES_DIR}/../results/{self.dataset}/{out_name}"):
            os.makedirs(f"{RES_DIR}/../results/{self.dataset}", exist_ok=True)
        df.to_csv(f"{RES_DIR}/../results/{self.dataset}/{out_name}", sep='\t', index=True)



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



# import yaml
# # --- Configurazione di base ---
# BASE_DIR = '/Users/ornellairrera/PycharmProjects/RecProbe/src/data/output/amazon_All_Beauty/'
# RECBOLE_DIR = '/Users/ornellairrera/PycharmProjects/RecProbe/src/baselines/recbole/'
#
# # Percorsi dei dati
# TRAIN_PATH = os.path.join(BASE_DIR, "train.csv")
# VALIDATION_PATH = os.path.join(BASE_DIR, "validation.csv")
# TEST_PATH = os.path.join(BASE_DIR, "test.csv")
#
# # Caricamento della configurazione generale
# CONFIG_PATH = os.path.join(RECBOLE_DIR, "config", "config.yaml")
# with open(CONFIG_PATH, "r") as f:
#     config = yaml.safe_load(f)
#
# # --- Istanzio il runner ---
# runner = RecBoleExperimentRunner(config=config, clean=True)
#
# # --- Eseguo l'esperimento ---
# runner.run(
#     training_path=TRAIN_PATH,
#     validation_path=VALIDATION_PATH,
#     test_path=TEST_PATH
# )
