import argparse
import sys
from logging import basicConfig

from mpmath.libmp.libelefun import machin
from pydantic.v1.validators import max_str_int
from recbole.quick_start import run_recbole
import json
import os

from recbole.trainer import HyperTuning
from recbole.quick_start import objective_function



CONFIG_DICT = {

    "BPR": {
        "amazon_baby": {
            "embedding_size": 64,
            "learning_rate": 0.01,
            "reg_weight": 1e-4,
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": 5
            }
        },
        "yelp": {
            "embedding_size": 64,
            "learning_rate": 0.001,
            "reg_weight": 1e-4,
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": 5
            }
        }
    },

    "SASRec": {
        "amazon_baby": {
            "hidden_size": 64,
            "num_blocks": 1,
            "num_heads": 1,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4,
            "MAX_ITEM_LIST_LENGTH": 50,

        },
        "yelp": {
            "hidden_size": 128,
            "num_blocks": 2,
            "num_heads": 2,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4,
            "MAX_ITEM_LIST_LENGTH": 100,

        }
    },

    "LightGCN": {
        "amazon_baby": {
            "embedding_size": 32,
            "n_layers": 1,
            "learning_rate": 0.001,
            "reg_weight": 1e-4,
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": 5
            }
        },
        "yelp": {
            "embedding_size": 64,
            "n_layers": 2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4,
            "train_neg_sample_args": {
                "distribution": "uniform",
                "sample_num": 5
            }
        }
    },

    "AutoInt": {
        "amazon_baby": {
            # fortemente sconsigliato, ma configurazione minima
            "embedding_size": 16,
            "attn_layer_num": 1,
            "attn_head_num": 2,
            "attn_head_size": 16,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        },
        "yelp": {
            # SOLO se dataset ridotto e feature limitate
            "embedding_size": 16,
            "attn_layer_num": 2,
            "attn_head_num": 2,
            "attn_head_size": 16,
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        }
    },

    "DCN": {
        "amazon_baby": {
            "embedding_size": 16,
            "cross_layer_num": 2,
            "mlp_hidden_size": [64, 32],
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        },
        "yelp": {
            "embedding_size": 32,
            "cross_layer_num": 3,
            "mlp_hidden_size": [128, 64],
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        }
    },

    "DeepFM": {
        "amazon_baby": {
            "embedding_size": 16,
            "mlp_hidden_size": [64, 32],
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        },
        "yelp": {
            "embedding_size": 32,
            "mlp_hidden_size": [128, 64],
            "dropout_prob": 0.2,
            "learning_rate": 0.001,
            "reg_weight": 1e-4
        }
    }
}



base_dir = os.path.dirname(os.path.abspath(__file__))
CONTEXT_MODELS = ['AutoInt','FFM','xDeepFM','NeuMF','DeepFM','LR','AFM','DCN']
#CONTEXT_MODELS = ['AFM','DCN']
GENERAL_MODELS = ['ItemKNN','Pop', 'BPR','LightGCN','NGCF']
SEQUENTIAL_MODELS = ['GRU4Rec','SASRec','BERT4Rec']
FILTERED_MODELS = ['Pop','AutoInt','DeepFM','DCN','SASRec','BPR','LightGCN']
#FILTERED_MODELS = ['SASRec','Pop','BPR','LightGCN']
#FILTERED_MODELS = ['NGCF','SASRec','AutoInt','DeepFM','DCN']
#FILTERED_MODELS = ['NGCF']
#SEQUENTIAL_MODELS = ['BERT4Rec']
VALID_DATASETS = ['amazon_beauty','yelp','amazon_baby']
#VALID_DATASETS = ['amazon_baby']
ALL_MODELS = GENERAL_MODELS + SEQUENTIAL_MODELS + CONTEXT_MODELS






def main(tuning=True,dataset=None,model=None,load_conf=False):



    print(f"Running model: {model} on dataset: {dataset}")
    best_file = f'{base_dir}/tuning/output_tuning/amazon_baby_{model.lower()}.best_params.json'

    try:
        # Hyperparameter tuning per tutti i modelli tranne 'Pop'
        conf = "config_files"
        if dataset == 'yelp':
            conf = 'yelp_config_files'
        config_dict = None
        if model != 'Pop' and tuning:
            max_evals = 70
            if model.lower() in ['bpr','neumf']:
                max_evals = 50
            elif model.lower() == ['itemknn','lr','ffm']:
                max_evals = 20

            hp = HyperTuning(objective_function=objective_function, algo='random', early_stop=10,
                             max_evals=max_evals, params_file=f'{base_dir}/{conf}/params_files/{model.lower()}.hyper',
                             fixed_config_file_list=[f'{base_dir}/{conf}/yaml_tuning/{model.lower()}.yaml'])
            # run
            hp.run()
            # export result to the file
            hp.export_result(output_file=f'{base_dir}/tuning/results/amazon_baby_{model.lower()}.result')
            # print best parameters
            config_dict = hp.best_params
            print('best params: ', hp.best_params)
            # print best result
            print('best result: ')
            print(hp.params2result[hp.params2str(hp.best_params)])
            # salva i best params in JSON (compatto)
            try:
                with open(best_file, 'w') as f:
                    json.dump(hp.best_params, f, default=str, indent=2)
            except Exception as save_err:
                print(f"Warning: non è stato possibile salvare {best_file}: {save_err}", file=sys.stderr)
        config_dict = {}
        if load_conf == False:
            if dataset == 'amazon_beauty':
                d = 'amazon_baby'
            else:
                d = dataset
            config_dict = CONFIG_DICT.get(model, {}).get(d, {})
            if model != 'Pop':
                config_dict['epochs'] = 100
                config_dict['eval_step'] = 100
                config_dict['stopping_step'] = 3

            if dataset == 'amazon_baby':
                config_dict['user_inter_num_interval'] = "[10,inf)"
                config_dict['item_inter_num_interval'] = "[10,inf)"
            elif dataset == 'amazon_beauty':
                config_dict['user_inter_num_interval'] = "[5,inf)"
                config_dict['item_inter_num_interval'] = "[5,inf)"
            elif dataset == 'yelp':
                config_dict['user_inter_num_interval'] = "[20,inf)"
                config_dict['item_inter_num_interval'] = "[20,inf)"
        if dataset == 'yelp':
            config_dict['seq_separator'] = " "
            if model != 'Pop':
                config_dict['epochs'] = 100
                config_dict['eval_step'] = 100
                config_dict['stopping_step'] = 5


        # se esiste un file con i best params lo carico e lo passo a run_recbole
        if os.path.exists(best_file) and load_conf:
            try:
                with open(best_file, 'r') as f:
                    config_dict = json.load(f)

                    config_dict['eval_step'] = 100
                    if dataset == 'yelp':
                        config_dict['user_inter_num_interval'] = "[20,inf)"
                        config_dict['item_inter_num_interval'] = "[20,inf)"
                        config_dict['seq_separator'] = " "

                    print(config_dict)
            except Exception as load_err:
                print(f"Warning: errore caricando {best_file}: {load_err}", file=sys.stderr)
                config_dict = None

        result = run_recbole(
            model=model,
            dataset=dataset,
            config_file_list=[f'{base_dir}/{conf}/yaml/{model.lower()}.yaml'],
            config_dict=config_dict
        )


        # result è un dict con metriche di test
        print('RESULT')
        print(result)

        # Salvarlo manualmente
        res_path = f'{base_dir}/tuning/final_results/loaded_final_{dataset}_{model.lower()}.json'
        try:
            with open(res_path, 'w') as ff:
                json.dump(result, ff, indent=4)
                print('written')
            ff.close()
        except Exception as save_err:
            print(f"Warning: non è stato possibile salvare {res_path}: {save_err}", file=sys.stderr)
        # run_recbole(
        #     model=model,
        #     dataset=dataset,
        #     config_file_list=[f'yaml/{model.lower()}.yaml']
        # )
    except Exception as e:
        print(f"Error executing {model} on {dataset}: {e}", file=sys.stderr)
        sys.exit(1)


def delete_files(checkpoint_dir, model):
    for filename in os.listdir(checkpoint_dir):
        if model in filename and "dataloader" not in filename:
            file_path = os.path.join(checkpoint_dir, filename)
            try:
                os.remove(file_path)
                print(f"Rimosso: {file_path}")
            except Exception as e:
                print(f"Errore rimuovendo {file_path}: {e}")
if __name__ == "__main__":
    # POP: alla me del futuro: se runno amazon_beauty_5 con [0,inf] sul yaml, dà lo stesso risultato di [5,inf] con amazon_beauty

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'model',
        nargs='?',  # <--- rende l'argomento posizionale opzionale
        default='BPR',  # usato se non viene passato da CLI
    )

    parser.add_argument(
        'dataset',
        nargs='?',  # <--- anche questo opzionale
        default='amazon_beauty',
        help='Dataset'
    )
    parser.add_argument(
        'type',
        nargs='?',  # <--- anche questo opzionale
        default='all',
        help='Type'
    )
    args = parser.parse_args()
    model = args.model
    dataset = args.dataset
    type = args.type
    VALID_MODELS = ALL_MODELS
    if type == 'general':
        VALID_MODELS = GENERAL_MODELS
    elif type == 'context':
        VALID_MODELS = CONTEXT_MODELS
    elif type == 'sequential':
        VALID_MODELS = SEQUENTIAL_MODELS
    elif type == 'filtered':
        VALID_MODELS = FILTERED_MODELS

    if model != 'all' and dataset != 'all':
        print(f"Starting run for model: {model}, dataset: {dataset}")
        main(tuning=False, dataset=dataset, model=model)
        checkpoint_dir = f"{base_dir}/checkpoint"
        delete_files(checkpoint_dir, model)

    elif model == 'all' and dataset != 'all':
        for m in VALID_MODELS:
            print(f"Starting run for model: {m}, dataset: {dataset}")
            main(tuning=False, dataset=dataset, model=m)
            checkpoint_dir = f"{base_dir}/checkpoint"
            delete_files(checkpoint_dir, m)
    elif model != 'all' and dataset == 'all':
        for d in VALID_DATASETS:
            print(f"Starting run for model: {model}, dataset: {d}")
            main(tuning=False, dataset=d, model=model)
            checkpoint_dir = f"{base_dir}/checkpoint"
            delete_files(checkpoint_dir, model)

    else:
        for dataset in VALID_DATASETS:
            for model in VALID_MODELS:

                print(f"Starting run for model: {model}, dataset: {dataset}")
                main(tuning=False, dataset=dataset, model=model)
                checkpoint_dir = f"{base_dir}/checkpoint"
                delete_files(checkpoint_dir, model)
    #BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    #config_path = os.path.join(BASE_DIR, f'config_files/yaml/pop.yaml')
    #run_recbole(model='BPR', dataset='amazon_beauty', config_file_list=[f'config_files/yaml/pop.yaml'])
    #run_recbole(model='Pop', dataset='ml-100k')
    # hp = HyperTuning(objective_function=objective_function, algo='exhaustive', early_stop=10,
    #                  max_evals=100, params_file='model.hyper', fixed_config_file_list=['example.yaml'])
    #
    # # run
    # hp.run()
    # # export result to the file
    # hp.export_result(output_file='hyper_example.result')
    # # print best parameters
    # print('best params: ', hp.best_params)
    # # print best result
    # print('best result: ')
    # print(hp.params2result[hp.params2str(hp.best_params)])


