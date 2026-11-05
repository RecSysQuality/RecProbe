"""
Common input-output interface of the evaluation runners (Section 4, fourth
principle). A runner trains one model with one seed and returns:

  * library_metrics : accuracy computed by the library with its own protocol
                      (whole user population);
  * user_metrics    : per-user accuracy from the library, when it exposes it
                      (used for user segments), else None;
  * ranked          : top-K lists for the test users (user_id, item_id, rank, score),
                      from which RecProbe computes beyond-accuracy metrics;
  * predictions     : rating predictions on the test pairs, for rating models.

New libraries are added by subclassing ``BaseRunner`` and registering the class
in ``RUNNERS`` (baselines_orchestrator.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from utils import ITEM, USER


@dataclass
class RunOutput:
    library_metrics: dict
    ranked: pd.DataFrame | None = None
    user_metrics: pd.DataFrame | None = None
    predictions: pd.DataFrame | None = None


class BaseRunner:
    framework = ""

    def __init__(self, fw_cfg, eval_cfg, logger, work_dir: str):
        self.fw_cfg = fw_cfg or {}
        self.eval_cfg = eval_cfg
        self.logger = logger
        self.work_dir = work_dir

    @property
    def k_max(self) -> int:
        return int(max(self.eval_cfg.cutoffs + [self.eval_cfg.top_k]))

    def run(self, model_cfg: dict, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame,
            seed: int) -> RunOutput:  # pragma: no cover - abstract
        raise NotImplementedError


def topk_from_scores(users, scores: np.ndarray, item_ids: np.ndarray, seen: list, k: int) -> pd.DataFrame:
    """scores: (n_users, n_items). Seen items are masked before ranking."""
    rows = []
    for u, s, excl in zip(users, scores, seen):
        s = s.astype(float).copy()
        if len(excl):
            s[excl] = -np.inf
        kk = min(k, int(np.isfinite(s).sum()))
        top = np.argpartition(-s, kk - 1)[:kk] if kk else np.array([], int)
        top = top[np.argsort(-s[top], kind="mergesort")]
        rows.extend((u, item_ids[i], r + 1, s[i]) for r, i in enumerate(top))
    return pd.DataFrame(rows, columns=[USER, ITEM, "rank", "score"])
