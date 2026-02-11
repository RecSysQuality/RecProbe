from cornac.models import *
from cornac.metrics import *

# Mapping dei nomi a classi
MODEL_MAP = {
    'BPR': BPR,
    'MostPop': MostPop,
    'LightGCN': LightGCN,
    'HRDR': HRDR,
    'NARRE': NARRE,
    'ConvMF': ConvMF,

    # add methods
}

METRIC_MAP = {
    'Recall': Recall,
    'NDCG': NDCG,
    'RMSE':RMSE
    # add eval
}
