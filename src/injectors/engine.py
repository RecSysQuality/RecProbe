"""
Injection driver: applies the configured perturbation to the training split
and, if requested, to the validation and test splits (Section 4.1, phase 3:
"By default, perturbations are applied exclusively to the training set").
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from injectors.base import InjectionResult, PerturbationContext
from injectors.registry import get_perturbation
from injectors.text_generation import TextGenerator
from utils import resolve_budget


class RowNoiseInjector:
    """Base class of the three row injectors of Table 1 (and of the generic operations)."""

    signal: str | None = None

    def __init__(self, config, logger, items: pd.DataFrame | None = None, rating_scale=None):
        self.config = config
        self.logger = logger
        self.items = items
        self.rating_scale = rating_scale

    def apply_noise(self, df: pd.DataFrame, df_val: pd.DataFrame | None = None,
                    df_test: pd.DataFrame | None = None) -> InjectionResult:
        inj = self.config.injection
        cls = get_perturbation(inj.perturbation)
        if self.signal and cls.signal != self.signal:
            raise ValueError(f"Perturbation '{cls.name}' corrupts the '{cls.signal}' signal, "
                             f"not '{self.signal}'")
        split_cfg = self.config.data.split
        todo = [("train", df)]
        if split_cfg.get("noise_in_validation") and df_val is not None and len(df_val):
            todo.append(("validation", df_val))
        if split_cfg.get("noise_in_test") and df_test is not None and len(df_test):
            todo.append(("test", df_test))

        seeds = np.random.SeedSequence(int(inj.random_seed)).spawn(len(todo))
        text = None
        llm_cfg = inj.params.get("llm_configuration")
        if getattr(cls, "needs_text", lambda p: False)(inj.params):
            text = TextGenerator(llm_cfg or {}, int(inj.random_seed), self.logger)

        splits = {"train": df, "validation": df_val, "test": df_test}
        logs, fakes, per_split = [], set(), {}
        for (name, part), ss in zip(todo, seeds):
            rng = np.random.default_rng(ss)
            reference = df if name == "train" else pd.concat([df, part], ignore_index=True)
            ctx = PerturbationContext(reference, self.items, self.rating_scale, rng, self.logger, name)
            budget = resolve_budget(inj.budget, len(part))
            pert = cls(inj.params, budget, ctx, self.logger, text)
            self.logger.info(f"Injecting '{cls.name}' ({cls.signal}/{cls.observability}, {cls.operation}) "
                             f"into {name}: budget={budget} of {len(part)} records")
            out = pert.apply(part)
            splits[name] = out
            if pert.log_rows:
                logs.append(pd.DataFrame(pert.log_rows).assign(split=name))
            fakes |= pert.fake_users
            per_split[name] = {**pert.stats, "records_before": len(part), "records_after": len(out)}
            self.logger.info(f"  -> {name}: {per_split[name]['target_records']} target records, "
                             f"{per_split[name].get('filler_records', 0)} filler records, "
                             f"{len(pert.fake_users)} fake users")

        log = pd.concat(logs, ignore_index=True) if logs else pd.DataFrame(
            columns=["perturbation", "operation", "record_id", "user_id", "item_id", "field",
                     "old_value", "new_value", "role", "split"])
        summary = {
            "perturbation": cls.name, "signal": cls.signal, "observability": cls.observability,
            "operation": cls.operation, "budget": inj.budget, "random_seed": inj.random_seed,
            "params": inj.params.to_dict() if hasattr(inj.params, "to_dict") else dict(inj.params),
            "splits": per_split,
        }
        llm = summary["params"].get("llm_configuration")
        if isinstance(llm, dict) and llm.get("huggingface_token"):
            llm["huggingface_token"] = "*****"
        return InjectionResult(splits=splits, log=log, fake_users=fakes, summary=summary)
