"""
Configuration handling for RecProbe.

As described in Section 4.1 of the paper, the pipeline is configured through
three YAML files, one per phase:

    (2) data processing   -> data.yaml
    (3) defect injection  -> injection.yaml   (template per defect in Table 1)
    (4) evaluation        -> evaluation.yaml

Each file is merged with the defaults below, so a user only needs to specify
the values that differ from them.  The Streamlit GUI writes the same three files.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from typing import Any

import yaml

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FILES_DIR = os.path.join(BASE_DIR, "config", "files")
INJECTION_TEMPLATES_DIR = os.path.join(FILES_DIR, "injection")
STREAMLIT_DIR = os.path.join(BASE_DIR, "streamlit_app", "generated_configs")


# --------------------------------------------------------------------------- #
# Attribute-access dictionary                                                 #
# --------------------------------------------------------------------------- #
class ConfigNode(dict):
    """dict with attribute access (``cfg.split.training``), recursive."""

    def __init__(self, data: dict | None = None):
        super().__init__()
        for k, v in (data or {}).items():
            self[k] = _wrap(v)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError as exc:
            raise AttributeError(key) from exc

    def __setattr__(self, key, value):
        self[key] = _wrap(value)

    def __deepcopy__(self, memo):
        return ConfigNode(copy.deepcopy(self.to_dict(), memo))

    def get_path(self, dotted: str, default=None):
        node: Any = self
        for part in dotted.split("."):
            if isinstance(node, dict) and part in node and node[part] is not None:
                node = node[part]
            else:
                return default
        return node

    def to_dict(self) -> dict:
        return _unwrap(self)


def _wrap(v):
    if isinstance(v, ConfigNode):
        return v
    if isinstance(v, dict):
        return ConfigNode(v)
    if isinstance(v, list):
        return [_wrap(x) for x in v]
    return v


def _unwrap(v):
    if isinstance(v, dict):
        return {k: _unwrap(x) for k, x in v.items()}
    if isinstance(v, list):
        return [_unwrap(x) for x in v]
    return v


def deep_merge(base: dict, override: dict | None) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def as_list(value, cast=None) -> list:
    """Accept YAML lists or whitespace/comma separated strings ("0 25 75")."""
    if value is None or value == "":
        return []
    if isinstance(value, (list, tuple)):
        items = list(value)
    else:
        items = str(value).replace(",", " ").split()
    return [cast(x) for x in items] if cast else items


# --------------------------------------------------------------------------- #
# Defaults                                                                    #
# --------------------------------------------------------------------------- #
DATA_DEFAULTS: dict = {
    "dataset": "dataset",
    # amazon | yelp | custom  (presets for field names / item metadata)
    "source": "custom",
    "input": {
        "reviews": {"path": None, "file_name": None, "format": "csv", "separator": ","},
        # optional item metadata (used e.g. by context swap and segment shilling)
        "items": None,
    },
    # mapping from RecProbe fields to dataset columns. user_id and item_id are
    # mandatory, the other fields are optional.
    "fields": {},
    "output": {"reviews": {"path": None, "format": "csv", "separator": "\t"}},
    # preprocessing / filtering (applied to the raw data, before the split)
    "kcore": 0,
    "min_rating": None,
    "max_rating": None,
    "min_review_length": 0,
    "drop_duplicates": True,
    # rating scale of the dataset; inferred from the data when left empty
    "rating_scale": {"min": None, "max": None, "step": None},
    "split": {
        "training": 0.8,
        "validation": 0.1,
        "test": 0.1,
        # random_holdout | temporal_leave_one_out | global_temporal |
        # random_leave_one_out | "package.module:function" (custom)
        "strategy": "random_holdout",
        "split_timestamp": None,        # global_temporal: test = after this
        "validation_timestamp": None,   # global_temporal: validation = after this
        "ensure_train_coverage": True,  # move eval records of unseen users/items to train
        "noise_in_validation": False,
        "noise_in_test": False,
    },
    "random_seed": 42,
    "verbose": True,
}

INJECTION_COMMON_DEFAULTS: dict = {
    # how users or items are selected for perturbation
    "target": "item",                    # user | item
    "selection_strategy": "uniform",     # uniform | top | bottom
    "top_k": None,                       # restrict top/bottom selection to k nodes
    "min_per_node": 1,
    "max_per_node": 10,
    # optional explicit list of target ids (overrides selection_strategy)
    "target_ids": None,
}

EVALUATION_DEFAULTS: dict = {
    # cornac | recbole | reviews4rec | custom
    "framework": "cornac",
    "beyond_accuracy": ["coverage", "novelty", "entropy", "gini"],
    # percentile thresholds on the clean-train interaction counts.
    # 0 = whole user population.
    "user_segments": [0],
    # explicit segments: {name: [ids]} or {name: path/to/ids.txt}
    "user_segment_ids": {},
    "seeds": [42],
    "top_k": 10,
    "cutoffs": None,                     # extra list lengths for beyond-accuracy
    "reuse_clean_results": True,
    "results_dir": None,
    "cornac": {"models": [], "metrics": []},
    "recbole": {"models": [], "metrics": [], "native_config": None, "params": {}},
    "reviews4rec": {"models": [], "repo_path": None, "command": None, "params": {}},
    "custom": {"models": [], "metrics": []},
}


# --------------------------------------------------------------------------- #
# Loading                                                                     #
# --------------------------------------------------------------------------- #
def read_yaml(path: str | None) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _normalize_strategy(name: str | None) -> str | None:
    if name is None:
        return None
    return name if ":" in name else name.replace("-", "_").strip()


def load_data_config(path: str | None) -> ConfigNode:
    cfg = deep_merge(DATA_DEFAULTS, read_yaml(path))
    cfg["split"]["strategy"] = _normalize_strategy(cfg["split"]["strategy"])
    if not cfg["output"]["reviews"].get("path"):
        cfg["output"]["reviews"]["path"] = os.path.join(BASE_DIR, "data", "output", cfg["dataset"])
    return ConfigNode(cfg)


def load_injection_config(path: str | None, perturbation: str | None = None) -> ConfigNode:
    """
    The injection file follows Figure 3 (central panel): a global ``budget``
    and a block named after the perturbation (e.g. ``review_burst:``).
    ``perturbation`` can be given explicitly or inferred from the block name.
    """
    from injectors.registry import resolve_name, get_perturbation, perturbation_names

    raw = read_yaml(path)
    name = perturbation or raw.get("perturbation")
    if name is None:
        blocks = [k for k in raw if resolve_name(k, strict=False) in perturbation_names()]
        if len(blocks) != 1:
            raise ValueError(
                "Cannot infer the perturbation from the injection file: specify "
                "'perturbation: <name>' or exactly one perturbation block. "
                f"Available: {sorted(perturbation_names())}"
            )
        name = blocks[0]
    canonical = resolve_name(name)
    cls = get_perturbation(canonical)

    block = dict(raw.get(name) or raw.get(canonical) or {})
    # Figure 3 naming: max_reviews_per_node / min_reviews_per_node -> max_per_node / min_per_node
    for k in list(block):
        for bound in ("min", "max"):
            if k.startswith(f"{bound}_") and k.endswith("_per_node") and k != f"{bound}_per_node":
                block[f"{bound}_per_node"] = block.pop(k)
    params = deep_merge(INJECTION_COMMON_DEFAULTS, {"target": cls.default_target})
    params = deep_merge(params, cls.DEFAULTS)
    params = deep_merge(params, block)
    # shared blocks may also be given at top level
    for shared in ("rating_behavior", "temporal_behavior", "fake_users", "llm_configuration"):
        if shared in raw and shared not in block:
            params[shared] = deep_merge(params.get(shared) or {}, raw[shared])

    cfg = {
        "perturbation": canonical,
        "signal": cls.signal,
        "observability": cls.observability,
        "budget": raw.get("budget", 0.1),
        "random_seed": raw.get("random_seed"),
        "params": params,
    }
    return ConfigNode(cfg)


def load_evaluation_config(path: str | None) -> ConfigNode:
    cfg = deep_merge(EVALUATION_DEFAULTS, read_yaml(path))
    cfg["beyond_accuracy"] = [m.lower() for m in as_list(cfg["beyond_accuracy"])]
    cfg["user_segments"] = as_list(cfg["user_segments"], float) or [0.0]
    cfg["seeds"] = as_list(cfg["seeds"], int) or [42]
    cfg["cutoffs"] = as_list(cfg["cutoffs"], int) or [int(cfg["top_k"])]
    return ConfigNode(cfg)


class RecProbeConfig:
    """Bundle of the three phase configurations."""

    def __init__(self, data: ConfigNode, injection: ConfigNode | None, evaluation: ConfigNode | None,
                 paths: dict | None = None):
        self.data = data
        self.injection = injection
        self.evaluation = evaluation
        self.paths = paths or {}

    @property
    def dataset(self) -> str:
        return self.data.dataset

    @property
    def random_seed(self) -> int:
        return int(self.data.random_seed)

    @property
    def output_dir(self) -> str:
        return self.data.output.reviews.path

    def injection_tag(self) -> str | None:
        """Unique, readable name for the perturbed dataset produced by this injection."""
        if self.injection is None:
            return None
        payload = json.dumps(self.injection.to_dict(), sort_keys=True, default=str)
        digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
        budget = str(self.injection.budget).replace(".", "p")
        return f"{self.injection.perturbation}_b{budget}_{digest}"

    def evaluation_hash(self) -> str:
        if self.evaluation is None:
            return "none"
        data = {k: v for k, v in self.data.to_dict().items() if k not in ("output", "verbose")}
        data["split"] = {k: v for k, v in data["split"].items() if not k.startswith("noise_in")}
        ev = {k: v for k, v in self.evaluation.to_dict().items() if k != "reuse_clean_results"}
        payload = json.dumps({"eval": ev, "data": data}, sort_keys=True, default=str)
        return hashlib.sha1(payload.encode()).hexdigest()[:8]


def load_config(data_path: str | None, injection_path: str | None = None,
                evaluation_path: str | None = None, perturbation: str | None = None) -> RecProbeConfig:
    data = load_data_config(data_path)
    injection = load_injection_config(injection_path, perturbation) if (injection_path or perturbation) else None
    evaluation = load_evaluation_config(evaluation_path) if evaluation_path else None
    if injection is not None and injection.random_seed is None:
        injection.random_seed = data.random_seed
    return RecProbeConfig(data, injection, evaluation,
                          {"data": data_path, "injection": injection_path, "evaluation": evaluation_path})


def load_streamlit_config() -> RecProbeConfig | None:
    """Load the three YAML files exported by the web interface, if present."""
    paths = {k: os.path.join(STREAMLIT_DIR, f"{k}.yaml") for k in ("data", "injection", "evaluation")}
    if not os.path.exists(paths["data"]):
        return None
    return load_config(
        paths["data"],
        paths["injection"] if os.path.exists(paths["injection"]) else None,
        paths["evaluation"] if os.path.exists(paths["evaluation"]) else None,
    )


def dump_yaml(cfg: dict | ConfigNode, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = cfg.to_dict() if isinstance(cfg, ConfigNode) else cfg
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(data, fh, sort_keys=False, allow_unicode=True)
