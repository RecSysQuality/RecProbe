"""Registry of the perturbations, keyed by the defect names of Table 1."""

from __future__ import annotations

import importlib

_REGISTRY: dict[str, type] = {}
_LOADED = False

_MODULES = (
    "injectors.interactions.interaction_injector",
    "injectors.interactions.rating_injector",
    "injectors.interactions.review_injector",
    "injectors.general_injector",
)

# names used by the previous version of the code -> Table 1 defects
ALIASES = {
    "misclick": "sequence_corruption",
    "implicit_burst": "interaction_burst",
    "burst_interaction": "interaction_burst",
    "shilling_attack": "rating_shilling",
    "uninformative": "uninformative_review",
    "off_topic": "off_topic_review",
    "timestamp_invalid": "invalid_timestamp",
    "rating_invalid": "invalid_rating",
    "history_inconsistent": "history_inconsistent_rating",
    "removal": "record_removal",
    "remove": "record_removal",
}
# alias whose meaning depends on the corrupted signal (old --profile flag)
SIGNAL_ALIASES = {
    ("profile_inconsistent", "interaction"): "sequence_corruption",
    ("profile_inconsistent", "rating"): "history_inconsistent_rating",
}


def register(cls):
    _REGISTRY[cls.name] = cls
    return cls


def _ensure_loaded():
    global _LOADED
    if not _LOADED:
        _LOADED = True
        for m in _MODULES:
            importlib.import_module(m)


def perturbation_names() -> set[str]:
    _ensure_loaded()
    return set(_REGISTRY)


def resolve_name(name: str, signal: str | None = None, strict: bool = True) -> str | None:
    _ensure_loaded()
    if name in _REGISTRY:
        return name
    if signal and (name, signal) in SIGNAL_ALIASES:
        return SIGNAL_ALIASES[(name, signal)]
    if name in ALIASES:
        return ALIASES[name]
    if strict:
        hint = " (ambiguous: also pass --profile)" if any(k[0] == name for k in SIGNAL_ALIASES) else ""
        raise ValueError(f"Unknown perturbation '{name}'{hint}. Available: {sorted(_REGISTRY)}")
    return None


def get_perturbation(name: str):
    _ensure_loaded()
    return _REGISTRY[resolve_name(name)]


def taxonomy_table() -> list[dict]:
    """Rows of Table 1 as implemented (signal, observability, perturbation)."""
    _ensure_loaded()
    return [
        {"perturbation": c.name, "signal": c.signal, "observability": c.observability,
         "operation": c.operation, "description": c.description}
        for c in sorted(_REGISTRY.values(), key=lambda c: (c.signal, c.observability or "", c.name))
    ]
