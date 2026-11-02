"""
RecProbe web interface (Section 4.1, Figure 2).

  (A) left sidebar   -> data processing (phase 2)
  (B) central panel  -> defect injection (phase 3): perturbation and its hyperparameters
  (C) right panel    -> evaluation (phase 4)

All parameters are converted into the three YAML files used by the pipeline
(streamlit_app/generated_configs/{data,injection,evaluation}.yaml), so runs
started from the interface are identical to runs started from the command line.

Run with:   streamlit run streamlit_app/app.py
"""

import os
import subprocess
import sys

import streamlit as st
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.config import (DATA_DEFAULTS, EVALUATION_DEFAULTS, INJECTION_COMMON_DEFAULTS,  # noqa: E402
                           STREAMLIT_DIR, deep_merge)
from injectors.registry import get_perturbation, taxonomy_table  # noqa: E402

CHOICES = {
    "target": ["item", "user"],
    "selection_strategy": ["uniform", "top", "bottom"],
    "sampling_strategy": ["uniform", "gaussian"],
    "fake_behavior": ["random", "mimic"],
    "direction": ["push", "nuke"],
    "filler_strategy": ["random", "popular", "segment"],
    "attack_type": ["random", "average", "bandwagon", "segment", "love_hate"],
    "backend": ["vllm", "transformers", "template"],
    "reference": ["user", "item"],
    "inconsistency": ["cooccurrence", "popularity", "category"],
    "reference_time": ["dataset_end", "now"],
    "item_choice": ["uniform", "popular"],
}
PER_PERTURBATION_CHOICES = {
    ("context_swap", "strategy"): ["donor", "llm"],
    ("sentiment_flip", "strategy"): ["llm", "donor"],
    ("off_topic_review", "mode"): ["replace", "append"],
    ("metadata_modification", "mode"): ["shift", "resample"],
}
SECRET_KEYS = {"huggingface_token"}

st.set_page_config(page_title="RecProbe", layout="wide")
st.title("RecProbe")
st.caption("Data perturbation framework for robustness analysis of recommender systems")


def _list_input(label, value, key):
    text = st.text_input(label, " ".join(map(str, value or [])), key=key,
                         help="space separated values")
    return [v for v in text.replace(",", " ").split()]


def render(params: dict, prefix: str, pert: str) -> dict:
    """Widgets generated from the defaults of the selected perturbation."""
    out = {}
    for k, v in params.items():
        key = f"{prefix}.{k}"
        choices = PER_PERTURBATION_CHOICES.get((pert, k)) or CHOICES.get(k)
        if isinstance(v, dict):
            with st.expander(k, expanded=k in ("rating_behavior", "temporal_behavior", "fake_users")):
                out[k] = render(v, key, pert)
        elif choices:
            idx = choices.index(v) if v in choices else 0
            out[k] = st.selectbox(k, choices, index=idx, key=key)
        elif isinstance(v, bool):
            out[k] = st.checkbox(k, v, key=key)
        elif isinstance(v, int):
            out[k] = int(st.number_input(k, value=v, step=1, key=key))
        elif isinstance(v, float):
            out[k] = float(st.number_input(k, value=v, key=key, format="%.4f"))
        elif isinstance(v, list):
            vals = _list_input(k, v, key)
            out[k] = [_num(x) for x in vals]
        else:
            text = st.text_input(k, "" if v is None else str(v), key=key,
                                 type="password" if k in SECRET_KEYS else "default")
            out[k] = None if text == "" else _num(text)
    return out


def _num(x):
    for cast in (int, float):
        try:
            return cast(x)
        except (TypeError, ValueError):
            pass
    return x


# ---------------------------------------------------------------- (A) data
with st.sidebar:
    st.header("A · Data processing")
    d = DATA_DEFAULTS
    dataset = st.text_input("Dataset name", "amazon_All_Beauty")
    source = st.selectbox("Source", ["amazon", "yelp", "custom"])
    in_path = st.text_input("Input folder or file", f"./data/input/{dataset}")
    in_name = st.text_input("File name (without extension)", "")
    in_fmt = st.selectbox("Input format", ["csv", "json", "jsonl"])
    in_sep = st.text_input("Separator (csv)", "\\t")
    items_name = st.text_input("Item metadata file name (optional)", "")
    items_fmt = st.selectbox("Item metadata format", ["jsonl", "json", "csv"])
    with st.expander("Field mapping (optional)"):
        fields = {f: st.text_input(f, "", key=f"field.{f}") for f in
                  ("user_id", "item_id", "timestamp", "rating", "review_text")}
    out_path = st.text_input("Output folder", f"./data/output/{dataset}")
    out_fmt = st.selectbox("Output format", ["csv", "jsonl", "json"])
    st.subheader("Filtering")
    kcore = int(st.number_input("k-core", value=3, min_value=0))
    min_rating = st.number_input("Minimum rating", value=1.0)
    min_len = int(st.number_input("Minimum review length (tokens)", value=0, min_value=0))
    dedup = st.checkbox("Drop duplicates", True)
    st.subheader("Split")
    strategy = st.selectbox("Strategy", ["random_holdout", "temporal_leave_one_out", "global_temporal",
                                         "random_leave_one_out"])
    c1, c2, c3 = st.columns(3)
    tr = c1.number_input("train", value=0.8)
    va = c2.number_input("val", value=0.1)
    te = c3.number_input("test", value=0.1)
    split_ts = st.text_input("Split timestamp (global temporal)", "")
    noise_val = st.checkbox("Apply noise to validation", False)
    noise_test = st.checkbox("Apply noise to test", False)
    data_seed = int(st.number_input("Random seed", value=42))

sep = in_sep.encode().decode("unicode_escape")
data_cfg = {
    "dataset": dataset, "source": source,
    "input": {"reviews": {"path": in_path, "file_name": in_name or None, "format": in_fmt, "separator": sep},
              "items": {"path": in_path, "file_name": items_name, "format": items_fmt} if items_name else None},
    "fields": {k: v for k, v in fields.items() if v},
    "output": {"reviews": {"path": out_path, "format": out_fmt, "separator": "\t"}},
    "kcore": kcore, "min_rating": min_rating, "min_review_length": min_len, "drop_duplicates": dedup,
    "split": {"training": tr, "validation": va, "test": te, "strategy": strategy,
              "split_timestamp": split_ts or None, "noise_in_validation": noise_val, "noise_in_test": noise_test},
    "random_seed": data_seed, "verbose": True,
}

col_b, col_c = st.columns([3, 2])

# ----------------------------------------------------------- (B) injection
with col_b:
    st.header("B · Defect injection")
    table = taxonomy_table()
    signals = ["interaction", "rating", "review", "general"]
    signal = st.radio("Corrupted signal (row of Table 1)", signals, horizontal=True)
    names = [r["perturbation"] for r in table if r["signal"] == signal]
    pert = st.selectbox("Perturbation", names)
    cls = get_perturbation(pert)
    st.caption(f"Observability: **{cls.observability or 'generic operation'}** · operation: "
               f"**{cls.operation}** — {cls.description}")
    budget_mode = st.radio("Budget", ["fraction of training edges", "absolute"], horizontal=True)
    budget = st.number_input("Budget value", value=0.1 if budget_mode.startswith("fraction") else 100.0)
    budget = float(budget) if budget_mode.startswith("fraction") else int(budget)
    inj_seed = int(st.number_input("Injection seed", value=42))
    defaults = deep_merge(deep_merge(INJECTION_COMMON_DEFAULTS, {"target": cls.default_target}), cls.DEFAULTS)
    params = render(defaults, f"inj.{pert}", pert)
    injection_cfg = {"budget": budget, "random_seed": inj_seed, pert: params}

# ---------------------------------------------------------- (C) evaluation
with col_c:
    st.header("C · Evaluation")
    run_eval = st.checkbox("Evaluate models on clean and perturbed data", True)
    framework = st.selectbox("Library", ["cornac", "recbole", "reviews4rec", "custom"])
    ba = st.multiselect("Beyond-accuracy metrics", ["coverage", "novelty", "entropy", "gini"],
                        default=["coverage", "novelty", "entropy", "gini"])
    segs = st.text_input("User segments (percentiles)", "0 25 75")
    seeds = st.text_input("Seeds", "0 10 42")
    top_k = int(st.number_input("Top-k", value=10, min_value=1))
    default_models = EVALUATION_DEFAULTS[framework]["models"] or []
    example = {"cornac": "- name: BPR\n  params: {k: 50, max_iter: 100, learning_rate: 0.01, lambda_reg: 0.001}\n"
                         "- name: MostPop\n  params: {}",
               "recbole": "- name: BPR\n  params: {embedding_size: 64}\n- name: LightGCN\n  params: {}",
               "reviews4rec": "- name: DeepCoNN\n  params: {}",
               "custom": "- name: MostPop\n- name: Random"}[framework]
    models_yaml = st.text_area("Models and hyperparameters (YAML)", example, height=140)
    metrics_yaml = st.text_area("Accuracy metrics (YAML)", "- NDCG: {k: 10}\n- Recall: {k: 10}", height=80)
    extra_yaml = st.text_area(f"Other '{framework}' options (YAML)", "{}", height=70,
                              help="e.g. native_config for RecBole, repo_path/command for Reviews4Rec")
    try:
        fw_block = {"models": yaml.safe_load(models_yaml) or [], "metrics": yaml.safe_load(metrics_yaml) or [],
                    **(yaml.safe_load(extra_yaml) or {})}
        yaml_ok = True
    except yaml.YAMLError as exc:
        st.error(f"Invalid YAML: {exc}")
        yaml_ok = False
    evaluation_cfg = {"framework": framework, "beyond_accuracy": ba, "user_segments": segs,
                      "seeds": seeds, "top_k": top_k, "cutoffs": [top_k], framework: fw_block}

# --------------------------------------------------------------- export/run
st.divider()
c1, c2 = st.columns(2)


def export():
    os.makedirs(STREAMLIT_DIR, exist_ok=True)
    files = {"data": data_cfg, "injection": injection_cfg}
    if run_eval:
        files["evaluation"] = evaluation_cfg
    elif os.path.exists(os.path.join(STREAMLIT_DIR, "evaluation.yaml")):
        os.remove(os.path.join(STREAMLIT_DIR, "evaluation.yaml"))
    for name, cfg in files.items():
        with open(os.path.join(STREAMLIT_DIR, f"{name}.yaml"), "w") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False, allow_unicode=True)
    return files


if c1.button("Export YAML files", disabled=not yaml_ok):
    files = export()
    st.success(f"Configuration written to {STREAMLIT_DIR}")
    for name, cfg in files.items():
        with st.expander(f"{name}.yaml"):
            shown = yaml.safe_load(yaml.safe_dump(cfg))
            llm = (shown.get(pert) or {}).get("llm_configuration") if name == "injection" else None
            if llm and llm.get("huggingface_token"):
                llm["huggingface_token"] = "*****"
            st.code(yaml.safe_dump(shown, sort_keys=False), language="yaml")

if c2.button("Export and run pipeline", type="primary", disabled=not yaml_ok):
    export()
    with st.spinner("Running RecProbe..."):
        proc = subprocess.run([sys.executable, "main.py", "--from-gui"], cwd=ROOT, capture_output=True, text=True)
    (st.success if proc.returncode == 0 else st.error)(f"Pipeline finished with exit code {proc.returncode}")
    st.code((proc.stdout + proc.stderr)[-8000:])
