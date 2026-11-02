"""
Cornac integration. Models, hyperparameters and metrics come from the
``cornac`` block of evaluation.yaml and are passed to Cornac as they are;
accuracy (overall and per user) is computed by Cornac with its own protocol.

Review-aware Cornac models receive the review text of the TRAINING split (so
review perturbations reach the model):
  * item_text   - one document per item with its concatenated reviews (HFT, ConvMF, CTR, CDL, ...)
  * review_text - one document per (user, item) review (NARRE, HRDR, ...)
The modality is chosen from the model name, or with ``text_modality`` on the model.
"""

from __future__ import annotations

import inspect

import numpy as np
import pandas as pd

from src.baselines.base_runner import BaseRunner, RunOutput, topk_from_scores
from src.evaluation.metrics import parse_metric_list
from utils import ITEM, RATING, REVIEW, USER

TEXT_MODALITY = {
    "HFT": "item_text", "ConvMF": "item_text", "CDL": "item_text", "CTR": "item_text",
    "CDR": "item_text", "CVAE": "item_text",
    "NARRE": "review_text", "HRDR": "review_text",
}


class CornacExperimentRunner(BaseRunner):
    framework = "cornac"

    def _uir(self, df: pd.DataFrame) -> list:
        r = df[RATING].fillna(1.0) if RATING in df.columns else pd.Series(1.0, index=df.index)
        return list(zip(df[USER].astype(str), df[ITEM].astype(str), r.astype(float)))

    def _metrics(self):
        import cornac.metrics as M

        out = []
        for name, params in parse_metric_list(self.fw_cfg.get("metrics") or self.eval_cfg.get("metrics")):
            if ":" in name:
                continue
            out.append(getattr(M, name)(**params))
        return out

    def run(self, model_cfg, train, val, test, seed):
        import cornac
        from cornac.eval_methods import BaseMethod

        name = model_cfg["name"]
        params = dict(model_cfg.get("params") or {})
        kwargs = {}
        modality = model_cfg.get("text_modality") or TEXT_MODALITY.get(name)
        if modality and REVIEW not in train.columns:
            raise ValueError(f"Cornac model {name} needs review text, which the dataset does not have")
        if modality:
            from cornac.data import ReviewModality, TextModality
            from cornac.data.text import BaseTokenizer

            rev = train[[USER, ITEM, REVIEW]].dropna()
            rev = rev[rev[REVIEW].astype(str).str.strip() != ""]
            tok = BaseTokenizer(stop_words="english")
            vocab = int(model_cfg.get("max_vocab", 4000))
            if modality == "item_text":
                docs = rev.groupby(ITEM)[REVIEW].apply(lambda s: " ".join(map(str, s)))
                # items without any review still need a (empty) document
                all_items = train[ITEM].astype(str).unique()
                docs = docs.reindex(all_items, fill_value="")
                kwargs["item_text"] = TextModality(corpus=docs.tolist(), ids=docs.index.tolist(),
                                                   tokenizer=tok, max_vocab=vocab, max_doc_freq=0.5)
            elif modality == "review_text":
                kwargs["review_text"] = ReviewModality(
                    data=list(zip(rev[USER].astype(str), rev[ITEM].astype(str), rev[REVIEW].astype(str))),
                    tokenizer=tok, max_vocab=vocab, max_doc_freq=0.5)
            else:
                raise ValueError(f"text_modality must be item_text or review_text, got '{modality}'")

        em = BaseMethod.from_splits(
            train_data=self._uir(train), test_data=self._uir(test),
            val_data=self._uir(val) if val is not None and len(val) else None,
            exclude_unknowns=True, seed=seed, verbose=False,
            rating_threshold=float(self.fw_cfg.get("rating_threshold", 1.0)), **kwargs)

        cls = getattr(cornac.models, name)
        sig = inspect.signature(cls.__init__).parameters
        if "seed" in sig and "seed" not in params:
            params["seed"] = seed
        if "verbose" in sig:
            params.setdefault("verbose", False)
        model = cls(**params)

        metrics = self._metrics()
        test_res, _ = em.evaluate(model, metrics, user_based=True, show_validation=False)
        lib = {k: float(v) for k, v in test_res.metric_avg_results.items() if not k.endswith("(s)")}

        idx2user = {v: k for k, v in em.train_set.uid_map.items()}
        user_metrics = pd.DataFrame({
            m: {idx2user[u]: v for u, v in d.items()} for m, d in test_res.metric_user_results.items()})

        ranked, preds = self._export(model, em, test)
        return RunOutput(lib, ranked, user_metrics, preds)

    def _export(self, model, em, test):
        ts = em.train_set
        item_ids = np.array([None] * ts.num_items, dtype=object)
        for k, v in ts.iid_map.items():
            item_ids[v] = k
        csr = ts.csr_matrix
        users = [u for u in test[USER].unique() if u in ts.uid_map]
        rows = []
        for start in range(0, len(users), 512):
            chunk = users[start:start + 512]
            idx = [ts.uid_map[u] for u in chunk]
            scores = np.vstack([np.asarray(model.score(i), dtype=float) for i in idx])
            seen = [csr.indices[csr.indptr[i]:csr.indptr[i + 1]] for i in idx]
            rows.append(topk_from_scores(chunk, scores, item_ids, seen, self.k_max))
        ranked = pd.concat(rows, ignore_index=True) if rows else None

        preds = None
        if RATING in test.columns and test[RATING].notna().any():
            t = test[test[USER].isin(ts.uid_map) & test[ITEM].isin(ts.iid_map)]
            try:
                p = [float(model.rate(ts.uid_map[u], ts.iid_map[i])) for u, i in zip(t[USER], t[ITEM])]
                preds = t[[USER, ITEM, RATING]].assign(prediction=p)
            except Exception:                 # ranking-only models may not support rate()
                preds = None
        return ranked, preds
