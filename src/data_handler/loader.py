"""
Dataset loading (Section 4.1, phases 1-2).

Supports the Amazon Reviews and Yelp datasets and custom datasets in CSV,
JSON or JSONL format. Only ``user_id`` and ``item_id`` are required; the
timestamp, rating and review fields are optional. Different column names are
mapped onto the RecProbe fields through the ``fields`` block of data.yaml.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from data_handler import preprocessing, splitters
from utils import CORE_FIELDS, ITEM, RATING, REVIEW, TS, USER, infer_rating_scale, to_unix_seconds

# candidate column names, in order of preference, for each preset
PRESETS = {
    "amazon": {
        "reviews": {
            USER: ["user_id", "reviewerID"],
            ITEM: ["parent_asin", "asin"],
            TS: ["timestamp", "unixReviewTime", "sort_timestamp"],
            RATING: ["rating", "overall"],
            REVIEW: ["text", "reviewText"],
        },
        "items": {ITEM: ["parent_asin", "asin"], "category": ["main_category", "categories", "category"],
                  "title": ["title"]},
    },
    "yelp": {
        "reviews": {USER: ["user_id"], ITEM: ["business_id"], TS: ["date"], RATING: ["stars"], REVIEW: ["text"]},
        "items": {ITEM: ["business_id"], "category": ["categories"], "title": ["name"]},
    },
    "custom": {
        "reviews": {USER: ["user_id", "user"], ITEM: ["item_id", "item"], TS: ["timestamp", "time", "date"],
                    RATING: ["rating", "stars", "score"], REVIEW: ["review_text", "review", "text"]},
        "items": {ITEM: ["item_id", "item"], "category": ["category", "categories"], "title": ["title", "name"]},
    },
}

EXT = {"csv": "csv", "tsv": "tsv", "json": "json", "jsonl": "jsonl"}


# --------------------------------------------------------------------------- #
# I/O helpers                                                                 #
# --------------------------------------------------------------------------- #
def read_table(path: str, fmt: str | None = None, sep: str | None = None) -> pd.DataFrame:
    fmt = (fmt or os.path.splitext(path.replace(".gz", ""))[1].lstrip(".") or "csv").lower()
    if fmt in ("csv", "tsv"):
        sep = sep if sep is not None else ("\t" if fmt == "tsv" else ",")
        return pd.read_csv(path, sep=sep, dtype={USER: str, ITEM: str}, keep_default_na=True)
    if fmt == "jsonl":
        return pd.read_json(path, lines=True, dtype=False)
    if fmt == "json":
        try:
            return pd.read_json(path, dtype=False)
        except ValueError:                      # e.g. Yelp files are JSON lines
            return pd.read_json(path, lines=True, dtype=False)
    raise ValueError(f"Unsupported format '{fmt}' (csv | json | jsonl)")


def write_table(df: pd.DataFrame, path: str, fmt: str = "csv", sep: str = "\t") -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    if fmt in ("csv", "tsv"):
        df.to_csv(path, sep=sep, index=False)
    elif fmt == "jsonl":
        df.to_json(path, orient="records", lines=True, force_ascii=False)
    elif fmt == "json":
        df.to_json(path, orient="records", force_ascii=False)
    else:
        raise ValueError(f"Unsupported output format '{fmt}'")
    return path


def _resolve_input_file(spec) -> str:
    path, name, fmt = spec.get("path"), spec.get("file_name"), spec.get("format", "csv")
    if not path:
        raise ValueError("input.reviews.path is required in data.yaml")
    candidates = []
    if name:
        for ext in (EXT.get(fmt, fmt), fmt, f"{EXT.get(fmt, fmt)}.gz", ""):
            candidates.append(os.path.join(path, f"{name}.{ext}" if ext else name))
    candidates.append(path)
    for c in candidates:
        if os.path.isfile(c):
            return c
    raise FileNotFoundError(f"Input file not found. Tried: {candidates}")


def _map_fields(raw: pd.DataFrame, candidates: dict, overrides: dict) -> pd.DataFrame:
    out = pd.DataFrame(index=raw.index)
    for field, cands in candidates.items():
        source = overrides.get(field)
        if source is None:
            source = next((c for c in cands if c in raw.columns), None)
        if source is not None and source in raw.columns:
            out[field] = raw[source]
    return out


# --------------------------------------------------------------------------- #
# Loader                                                                      #
# --------------------------------------------------------------------------- #
class DatasetLoader:
    def __init__(self, logger, config):
        self.logger = logger
        self.config = config                   # RecProbeConfig
        self.cfg = config.data
        self.rng = np.random.default_rng(config.random_seed)
        self.items: pd.DataFrame | None = None
        self.rating_scale: tuple | None = None

    # ------------------------------------------------------------------ raw
    def load_raw(self) -> pd.DataFrame:
        spec = self.cfg.input.reviews
        path = _resolve_input_file(spec)
        self.logger.info(f"Loading {path}")
        raw = read_table(path, spec.get("format"), spec.get("separator"))
        preset = PRESETS.get(self.cfg.source, PRESETS["custom"])
        overrides = {k: v for k, v in (self.cfg.fields or {}).items() if k in CORE_FIELDS}
        df = _map_fields(raw, preset["reviews"], overrides)
        for req in (USER, ITEM):
            if req not in df.columns:
                raise ValueError(f"Mandatory field '{req}' not found; map it in 'fields' of data.yaml")
        df[USER] = df[USER].astype(str)
        df[ITEM] = df[ITEM].astype(str)
        if TS in df.columns:
            df[TS] = to_unix_seconds(df[TS])
        if RATING in df.columns:
            df[RATING] = pd.to_numeric(df[RATING], errors="coerce")
        if REVIEW in df.columns:
            df[REVIEW] = df[REVIEW].astype("string")
        self.items = self._load_items(preset)
        return df

    def _load_items(self, preset) -> pd.DataFrame | None:
        spec = self.cfg.input.get("items")
        if not spec or not spec.get("path"):
            return None
        raw = read_table(_resolve_input_file(spec), spec.get("format"), spec.get("separator"))
        overrides = {k: v for k, v in (self.cfg.fields or {}).items() if k in ("category", "title")}
        overrides.update({ITEM: spec.get("item_field")} if spec.get("item_field") else {})
        items = _map_fields(raw, preset["items"], overrides)
        items[ITEM] = items[ITEM].astype(str)
        if "category" in items.columns:
            items["category"] = items["category"].map(_first_category)
        return items.drop_duplicates(ITEM)

    # ---------------------------------------------------------- processing
    def process(self) -> pd.DataFrame:
        df = preprocessing.preprocess(self.load_raw(), self.cfg, self.logger)
        if df.empty:
            raise ValueError("No interactions left after preprocessing; relax the filters.")
        if RATING in df.columns and df[RATING].notna().any():
            sc = self.cfg.rating_scale
            inferred = infer_rating_scale(df[RATING])
            self.rating_scale = (
                sc.min if sc.min is not None else inferred[0],
                sc.max if sc.max is not None else inferred[1],
                sc.step if sc.step is not None else inferred[2],
            )
            self.logger.info(f"Rating scale: min={self.rating_scale[0]} max={self.rating_scale[1]} "
                             f"step={self.rating_scale[2]}")
        if self.items is not None:
            self.items = self.items[self.items[ITEM].isin(set(df[ITEM]))]
        return df

    def load_data(self):
        """Process the raw data and split it: returns (train, validation, test)."""
        df = self.process()
        return splitters.split(df, self.cfg.split, self.rng, self.logger)

    # ------------------------------------------------------------- output
    @property
    def out_dir(self) -> str:
        return self.cfg.output.reviews.path

    def _fmt(self):
        o = self.cfg.output.reviews
        return o.get("format", "csv"), o.get("separator", "\t")

    def split_path(self, split: str, noisy_tag: str | None = None) -> str:
        fmt, _ = self._fmt()
        base = os.path.join(self.out_dir, "noisy", noisy_tag) if noisy_tag else self.out_dir
        return os.path.join(base, f"{split}.{EXT.get(fmt, fmt)}")

    def save_clean(self, train, val, test) -> dict:
        fmt, sep = self._fmt()
        paths = {}
        for name, part in (("train", train), ("validation", val), ("test", test)):
            paths[name] = write_table(part, self.split_path(name), fmt, sep)
        if self.items is not None:
            write_table(self.items, os.path.join(self.out_dir, "items.csv"), "csv", "\t")
        meta = {"rating_scale": self.rating_scale, "n_train": len(train), "n_validation": len(val),
                "n_test": len(test), "split": self.cfg.split.to_dict()}
        with open(os.path.join(self.out_dir, "processed_meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2, default=str)
        self.logger.info(f"Clean splits saved to {self.out_dir}")
        return paths

    def save_noisy(self, tag: str, result, injection_cfg) -> str:
        """Store the perturbed dataset as a reusable artifact (Section 4, second principle)."""
        from config.config import dump_yaml

        fmt, sep = self._fmt()
        folder = os.path.join(self.out_dir, "noisy", tag)
        os.makedirs(folder, exist_ok=True)
        for name, part in result.splits.items():
            if part is not None:
                write_table(part, self.split_path(name, tag), fmt, sep)
        result.log.to_csv(os.path.join(folder, "injection_log.tsv"), sep="\t", index=False)
        with open(os.path.join(folder, "fake_users.txt"), "w") as fh:
            fh.write("\n".join(map(str, sorted(result.fake_users))))
        dump_yaml(injection_cfg, os.path.join(folder, "injection.yaml"))
        with open(os.path.join(folder, "manifest.json"), "w") as fh:
            json.dump(result.summary, fh, indent=2, default=str)
        self.logger.info(f"Perturbed dataset saved to {folder}")
        return folder

    def load_processed(self):
        """Read back the clean splits (evaluation-only runs)."""
        fmt, sep = self._fmt()
        parts = []
        for name in ("train", "validation", "test"):
            p = self.split_path(name)
            parts.append(read_table(p, fmt, sep) if os.path.exists(p) else None)
        items_path = os.path.join(self.out_dir, "items.csv")
        if os.path.exists(items_path):
            self.items = read_table(items_path, "csv", "\t")
        meta_path = os.path.join(self.out_dir, "processed_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path) as fh:
                rs = json.load(fh).get("rating_scale")
                self.rating_scale = tuple(rs) if rs else None
        return parts


def _first_category(value):
    """Amazon categories may be lists, Yelp categories comma-separated strings."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, (list, tuple, np.ndarray)):
        flat = [v for v in value if v]
        return str(flat[0]) if flat else None
    s = str(value)
    return s.split(",")[0].strip() if "," in s else s
