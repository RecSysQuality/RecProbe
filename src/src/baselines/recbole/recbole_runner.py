"""
RecBole integration.

The splits produced by RecProbe are written as RecBole atomic files
(<dataset>.train.inter / .valid.inter / .test.inter) and loaded with
``benchmark_filename``, so RecBole trains and evaluates on exactly the same
clean or perturbed data. Settings come from the ``recbole`` block of
evaluation.yaml; a native RecBole YAML file can also be given with
``native_config`` (library-specific options, Section 4.1).

Accuracy is computed by RecBole on the whole population. RecBole does not
expose per-user values, so segment-level accuracy is computed by RecProbe on
the top-K lists exported with ``full_sort_topk``.
"""

from __future__ import annotations

import os

import pandas as pd

from src.baselines.base_runner import BaseRunner, RunOutput
from src.evaluation.metrics import parse_metric_list
from utils import ITEM, RATING, TS, USER


def _numpy2_shim():
    """RecBole 1.2 references aliases removed in NumPy 2 (np.float_, np.unicode_, ...)."""
    import numpy as np

    for alias, target in (("float_", "float64"), ("complex_", "complex128"), ("unicode_", "str_"),
                          ("string_", "bytes_")):
        if alias not in np.__dict__:
            setattr(np, alias, getattr(np, target))


class RecBoleExperimentRunner(BaseRunner):
    framework = "recbole"

    def _write_atomic(self, parts: dict, folder: str, name: str):
        os.makedirs(folder, exist_ok=True)
        for split, df in parts.items():
            cols = {USER: "user_id:token", ITEM: "item_id:token"}
            if RATING in df.columns and df[RATING].notna().any():
                cols[RATING] = "rating:float"
            if TS in df.columns and df[TS].notna().any():
                cols[TS] = "timestamp:float"
            out = df[list(cols)].rename(columns=cols)
            out.to_csv(os.path.join(folder, f"{name}.{split}.inter"), sep="\t", index=False)

    def run(self, model_cfg, train, val, test, seed):
        _numpy2_shim()
        import torch
        from recbole.config import Config
        from recbole.data import create_dataset, data_preparation
        from recbole.utils import get_model, get_trainer, init_seed
        from recbole.utils.case_study import full_sort_topk

        name = "recprobe"
        data_root = os.path.join(self.work_dir, "recbole_data")
        if val is None or not len(val):
            self.logger.warning("RecBole needs a validation split: using the test split for validation")
            val = test
        self._write_atomic({"train": train, "valid": val, "test": test}, os.path.join(data_root, name), name)

        metrics, topk = [], set()
        for m, p in parse_metric_list(self.fw_cfg.get("metrics") or [{"NDCG": {"k": 10}}]):
            metrics.append(m)
            if "k" in p:
                topk.add(int(p["k"]))
        topk = sorted(topk or {int(self.eval_cfg.top_k)})
        load_cols = ["user_id", "item_id"] + (["rating"] if RATING in train.columns and train[RATING].notna().any() else []) \
            + (["timestamp"] if TS in train.columns and train[TS].notna().any() else [])
        config_dict = {
            "data_path": data_root, "dataset": name,
            "benchmark_filename": ["train", "valid", "test"],
            "load_col": {"inter": load_cols},
            "seed": int(seed), "reproducibility": True,
            "metrics": metrics, "topk": topk,
            "valid_metric": self.fw_cfg.get("valid_metric", f"NDCG@{topk[0]}"),
            "eval_args": {"group_by": "user", "order": "RO", "mode": "full"},
            "checkpoint_dir": os.path.join(self.work_dir, "recbole_saved"),
            "show_progress": False,
            **(self.fw_cfg.get("params") or {}),
            **(model_cfg.get("params") or {}),
        }
        native = [self.fw_cfg["native_config"]] if self.fw_cfg.get("native_config") else None
        config = Config(model=model_cfg["name"], dataset=name, config_file_list=native, config_dict=config_dict)
        init_seed(config["seed"], config["reproducibility"])
        dataset = create_dataset(config)
        train_data, valid_data, test_data = data_preparation(config, dataset)
        model = get_model(config["model"])(config, train_data._dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, model)
        trainer.fit(train_data, valid_data, saved=True, show_progress=False)
        # best checkpoint: written by this same run a moment ago. Loaded here because
        # torch >= 2.6 defaults to weights_only=True, which RecBole 1.2 checkpoints do not support.
        ckpt_file = getattr(trainer, "saved_model_file", None)
        if ckpt_file and os.path.exists(ckpt_file):
            ckpt = torch.load(ckpt_file, map_location=config["device"], weights_only=False)
            model.load_state_dict(ckpt["state_dict"])
            model.load_other_parameter(ckpt.get("other_parameter"))
        result = trainer.evaluate(test_data, load_best_model=False, show_progress=False)
        lib = {self._metric_name(k): float(v) for k, v in dict(result).items()}

        # export top-K lists (history items are masked by RecBole)
        tokens = [u for u in test[USER].astype(str).unique() if u in dataset.field2token_id[dataset.uid_field]]
        rows = []
        model.eval()
        for start in range(0, len(tokens), 1024):
            chunk = tokens[start:start + 1024]
            uids = dataset.token2id(dataset.uid_field, chunk)
            with torch.no_grad():
                scores, iids = full_sort_topk(uids, model, test_data, k=self.k_max, device=config["device"])
            items = dataset.id2token(dataset.iid_field, iids.cpu())
            scores = scores.cpu().numpy()
            for u, its, sc in zip(chunk, items, scores):
                rows.extend((u, str(i), r + 1, float(s)) for r, (i, s) in enumerate(zip(its, sc)))
        ranked = pd.DataFrame(rows, columns=[USER, ITEM, "rank", "score"])
        return RunOutput(lib, ranked, None, None)

    @staticmethod
    def _metric_name(k: str) -> str:
        """recbole 'ndcg@10' -> 'NDCG@10' (same naming as the other runners)."""
        base, _, cut = k.partition("@")
        names = {"ndcg": "NDCG", "recall": "Recall", "precision": "Precision", "hit": "HitRatio",
                 "mrr": "MRR", "map": "MAP", "rmse": "RMSE", "mae": "MAE"}
        return names.get(base.lower(), base) + (f"@{cut}" if cut else "")
