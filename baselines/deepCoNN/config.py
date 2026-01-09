import argparse
import inspect

import torch


class Config:
    device = torch.device("cuda:0")
    # device = torch.device("cpu")

    dataset = 'amazon_beauty'
    word2vec_file = 'embedding/glove.6B.50d.txt'
    train_file = f'data/{dataset}/train.csv'
    valid_file = f'data/{dataset}/valid.csv'
    test_file = f'data/{dataset}/test.csv'
    model_file = f'model/best_model_{dataset}.pt'
    save_path = f'data/{dataset}/results.json'
    path_inter_train = f'data/{dataset}/train.pt'
    path_inter_vali = f'data/{dataset}/vali.pt'
    path_inter_test = f'data/{dataset}/test.pt'
    model = 'NARRE'
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
            word_dim = 50  # dimensione embedding parole (GloVe 50d)
            id_dim = 32  # dimensione embedding user/item
            dropout_prob = 0.4
            if dataset == 'amazon_beauty':
                review_count = 30
                review_length = 40
                lowest_review_count = 2

                train_epochs = 50
                batch_size = 32
                learning_rate = 0.001
                learning_rate_decay = 0.98

                l2_regularization = 5e-2
                dropout_prob = 0.8

                kernel_deep = 8  # NARRE usa kernel_deep
                kernel_size = 3


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

                kernel_deep = 32
                kernel_size = 3
            else:
                review_count = 20
                review_length = 100
                lowest_review_count = 5

                batch_size = 128
                train_epochs = 30
                learning_rate = 1e-4
                learning_rate_decay = 0.995

                l2_regularization = 1e-4
                dropout_prob = 0.4

                kernel_deep = 64
                kernel_size = 3

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
