import argparse
import inspect

import torch


class Config:
    device = torch.device("cuda:0")
    #device = torch.device("cpu")
    model = 'NARRE'
    dataset = 'yelp'
    word2vec_file = 'embedding/glove.6B.50d.txt'
    train_file = f'data/{dataset}/train.csv'
    valid_file = f'data/{dataset}/valid.csv'
    test_file = f'data/{dataset}/test.csv'
    model_file = f'model/best_model_{dataset}.pt'
    save_path = f'data/{dataset}/results.json'
    path_inter_train = f'data/{dataset}/train.pt'
    path_inter_vali = f'data/{dataset}/vali.pt'
    path_inter_test = f'data/{dataset}/test.pt'
    path_inter_train_narre = f'data/{dataset}/train_narre.pt'
    path_inter_vali_narre = f'data/{dataset}/vali_narre.pt'
    path_inter_test_narre = f'data/{dataset}/test_narre.pt'
    if model == 'DeepCoNN':
        if dataset == 'amazon_beauty':
            review_count = 30  # pochi item per utente
            review_length = 40
            lowest_review_count = 2

            train_epochs = 50  # più epoche per compensare pochi dati
            batch_size = 32
            learning_rate = 0.001
            learning_rate_decay = 0.98

            l2_regularization = 5e-2  # forte regolarizzazione
            dropout_prob = 0.8
            early_stop = 10
            kernel_count = 8
            kernel_size = 3
            cnn_out_dim = 8

        elif dataset == 'amazon_baby':
            review_count = 20
            review_length = 50
            lowest_review_count = 2

            train_epochs = 30
            batch_size = 128
            learning_rate = 0.0001
            learning_rate_decay = 0.99

            l2_regularization = 1e-4
            dropout_prob = 0.3
            early_stop = 10
            kernel_count = 32
            kernel_size = 3
            cnn_out_dim = 32
        else:
            review_count = 20
            review_length = 100
            lowest_review_count = 5

            batch_size = 128
            learning_rate = 1e-4
            learning_rate_decay = 0.995

            kernel_size = 3
            kernel_count = 64
            cnn_out_dim = 64

            dropout_prob = 0.4
            l2_regularization = 1e-4

            train_epochs = 30
            early_stop = 10
    else:
        if model == 'NARRE':
            save_path = f'data/{dataset}/results_narre.json'

            word_dim = 50  # dimensione embedding parole (GloVe 50d)

            vocab_size = 1291148
            early_stop=10
            if dataset == 'amazon_beauty':
                review_count = 10
                kernel_count = 16
                kernel_deep = 4
                learning_rate = 0.005
                batch_size = 16
                early_stop = 5
                kernel_size = 3
                l2_regularization = 1e-2
                learning_rate_decay = 0.99
                train_epochs = 30
                review_length = 30  # ↓ da 40 (focus su review brevi)
                id_emb_size = 32  # ↓ da 64 → 32 (meno parametri)
                filters_num = 64  # ↓ da 128 → 64 (sufficiente per piccolo dataset)
                drop_out = 0.3  # ↓ da 0.4 → 0.3 (ancora meno regolarizzazione)
                #learning_rate = 0.0005  # ↓ da 0.001 → 0.0005 (convergenza lenta ma stabile)
                #early_stop = 15  # Più pazienza
                #l2_regularization = 1e-5  # ↓ da 5e-5 → 1e-5 (molto leggero)
                #train_epochs = 30  # Molte epoche, early stopping salverà
            elif dataset == 'amazon_baby':
                #review_count = 20  # ↑ da 10 (più contesto)
                #review_length = 50  # ↑ da 30 (review più lunghe)
                id_emb_size = 32  # ↑ da 32 → 64 (più espressività)
                #filters_num = 128  # ↑ da 64 → 128 (più feature CNN)
                #vocab_size = 10000
                drop_out = 0.4  # ↑ da 0.3 → 0.4 (leggermente più regolarizzazione)
                kernel_size = 3
                kernel_count = 16  # Puoi usarlo per multi-kernel
                kernel_deep = 4  # Idem
                learning_rate = 0.002  # ↓ da 0.005 → 0.002 (stabile con rete più grande)
                #batch_size = 64  # ↑ da 16 → 64 (efficienza)
                early_stop = 12  # Più pazienza
                l2_regularization = 5e-5  # ↓ da 1e-2 → 5e-5 (molto leggero)
                learning_rate_decay = 0.98  # Leggermente più aggressivo
                train_epochs = 60  # Più epoche
                review_count = 20
                review_length = 30
                filters_num = 64
                batch_size = 64
            else:
                # review_count = 20  # ↑ da 10 (più contesto)
                # review_length = 50  # ↑ da 30 (review più lunghe)
                id_emb_size = 32  # ↑ da 32 → 64 (più espressività)
                # filters_num = 128  # ↑ da 64 → 128 (più feature CNN)
                # vocab_size = 10000
                drop_out = 0.4  # ↑ da 0.3 → 0.4 (leggermente più regolarizzazione)
                kernel_size = 3
                kernel_count = 16  # Puoi usarlo per multi-kernel
                kernel_deep = 4  # Idem
                learning_rate = 0.0001  # ↓ da 0.005 → 0.002 (stabile con rete più grande)
                # batch_size = 64  # ↑ da 16 → 64 (efficienza)
                early_stop = 12  # Più pazienza
                l2_regularization = 1e-5  # ↓ da 1e-2 → 5e-5 (molto leggero)
                learning_rate_decay = 0.99  # Leggermente più aggressivo
                train_epochs = 30  # Più epoche
                review_count = 20
                review_length = 30
                filters_num = 64
                batch_size = 128


    PAD_WORD = '<UNK>'



    def __init__(self):
        # By the way, we can customize parameters in the command line parameters.
        # For example:
        # python main.py --device cuda:0 --train_epochs 50
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))

        parser = argparse.ArgumentParser()
        for key, val in attributes:
            parser.add_argument('--' + key, dest=key, type=type(val), default=val)
        for key, val in parser.parse_args().__dict__.items():
            self.__setattr__(key, val)

    def __str__(self):
        attributes = inspect.getmembers(self, lambda a: not inspect.isfunction(a))
        attributes = list(filter(lambda x: not x[0].startswith('__'), attributes))
        to_str = ''
        for key, val in attributes:
            to_str += '{} = {}\n'.format(key, val)
        return to_str
