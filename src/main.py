"""
RecProbe - data perturbation framework for robustness analysis of recommender systems.

Pipeline (Figure 1):  (1) setup and configuration -> (2) data processing ->
                      (3) defect injection -> (4) evaluation

Examples
--------
  # full pipeline with the three YAML files of Section 4.1
  python main.py --data config/files/data.yaml \
                 --injection config/files/injection/review_burst.yaml \
                 --evaluation config/files/evaluation.yaml

  # previous CLI: perturbation by name, using its template in config/files/injection/
  python main.py --profile rating --noise_injection rating_flip --baselines

  # single stages
  python main.py --data ... --stage process
  python main.py --data ... --injection ... --stage inject
  python main.py --data ... --injection ... --evaluation ... --stage evaluate

  # list the perturbations (Table 1) / regenerate the injection templates
  python main.py --list
  python main.py --write-templates
"""

import argparse
import logging
import os
import sys
from datetime import datetime

import pandas as pd

from baselines_orchestrator import BaselinesOrchestrator, create_unique_table
from config.config import (FILES_DIR, INJECTION_TEMPLATES_DIR, ConfigNode, dump_yaml,
                           load_config, load_streamlit_config)
from data_handler.loader import DatasetLoader, read_table
from injectors.registry import resolve_name, taxonomy_table
from logger import get_logger
from orchestrator import NoiseOrchestrator
from utils import set_seed

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    p = argparse.ArgumentParser(description="RecProbe: data perturbation and robustness evaluation")
    p.add_argument("--data", help="data processing YAML (phase 2)")
    p.add_argument("--injection", help="defect injection YAML (phase 3)")
    p.add_argument("--evaluation", help="evaluation YAML (phase 4)")
    p.add_argument("--stage", choices=["all", "process", "inject", "evaluate"], default="all")
    p.add_argument("--from-gui", action="store_true", help="use the YAML files exported by the web interface")
    # previous interface
    p.add_argument("--profile", choices=["interaction", "rating", "review", "general", ""], default="",
                   help="corrupted signal (row of Table 1)")
    p.add_argument("--noise_injection", default="", help="perturbation name (or legacy alias)")
    p.add_argument("--baselines", action="store_true", help="run the evaluation phase")
    # utilities
    p.add_argument("--list", action="store_true", help="print the implemented taxonomy and exit")
    p.add_argument("--write-templates", action="store_true", help="write one injection template per defect")
    return p.parse_args()


def write_templates(folder: str = INJECTION_TEMPLATES_DIR) -> None:
    """One injection YAML template per perturbation (Section 4.1: 'a template specific to each defect')."""
    from config.config import INJECTION_COMMON_DEFAULTS, deep_merge
    from injectors.registry import get_perturbation

    os.makedirs(folder, exist_ok=True)
    for row in taxonomy_table():
        cls = get_perturbation(row["perturbation"])
        params = deep_merge(INJECTION_COMMON_DEFAULTS, {"target": cls.default_target})
        params = deep_merge(params, cls.DEFAULTS)
        if cls.__name__.endswith("Shilling"):          # one group hits all targets: no per-node limits
            for k in ("min_per_node", "max_per_node", "target"):
                params.pop(k, None)
        path = os.path.join(folder, f"{cls.name}.yaml")
        dump_yaml({"budget": 0.1, "random_seed": 42, cls.name: params}, path)
        with open(path) as fh:
            body = fh.read()
        header = (f"# {cls.name}: {row['description']}\n"
                  f"# corrupted signal: {cls.signal} | observability: {cls.observability} | "
                  f"operation: {cls.operation}\n")
        with open(path, "w") as fh:
            fh.write(header + body)
    print(f"Templates written to {folder}")


def resolve_paths(args):
    if args.from_gui:
        cfg = load_streamlit_config()
        if cfg is None:
            sys.exit("No configuration exported by the web interface was found.")
        return cfg
    data = args.data or os.path.join(FILES_DIR, "data.yaml")
    injection = args.injection
    perturbation = None
    if args.noise_injection:
        perturbation = resolve_name(args.noise_injection, signal=args.profile or None)
        injection = injection or os.path.join(INJECTION_TEMPLATES_DIR, f"{perturbation}.yaml")
    evaluation = args.evaluation
    if evaluation is None and (args.baselines or args.stage == "evaluate"):
        evaluation = os.path.join(FILES_DIR, "evaluation.yaml")
    return load_config(data, injection, evaluation, perturbation)


def masked(injection_cfg: ConfigNode) -> dict:
    d = injection_cfg.to_dict()
    llm = d.get("params", {}).get("llm_configuration")
    if isinstance(llm, dict) and llm.get("huggingface_token"):
        llm["huggingface_token"] = "*****"
    return d


def load_noisy(loader: DatasetLoader, tag: str):
    folder = os.path.join(loader.out_dir, "noisy", tag)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Perturbed dataset '{tag}' not found in {folder}: run the inject stage first")
    fmt = loader.cfg.output.reviews.get("format", "csv")
    sep = loader.cfg.output.reviews.get("separator", "\t")
    parts = []
    for name in ("train", "validation", "test"):
        p = loader.split_path(name, tag)
        parts.append(read_table(p, fmt, sep) if os.path.exists(p) else None)
    fakes_path = os.path.join(folder, "fake_users.txt")
    fakes = set(open(fakes_path).read().split()) if os.path.exists(fakes_path) else set()
    return parts, fakes


def main():
    args = parse_args()
    if args.list:
        print(pd.DataFrame(taxonomy_table()).to_string(index=False))
        return
    if args.write_templates:
        write_templates()
        return

    config = resolve_paths(args)
    log_file = os.path.join(BASE_DIR, "logs", f"recprobe_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger = get_logger(log_file=log_file, level=logging.INFO if config.data.verbose else logging.WARNING)
    set_seed(config.random_seed, verbose=False)
    logger.info(f"Configuration files: {config.paths}")

    stage = args.stage
    do_process = stage in ("all", "process")
    do_inject = stage in ("all", "inject") and config.injection is not None
    do_eval = (stage == "evaluate") or (stage == "all" and config.evaluation is not None)

    # (2) data processing
    loader = DatasetLoader(logger=logger, config=config)
    if do_process:
        train, val, test = loader.load_data()
        loader.save_clean(train, val, test)
    else:
        train, val, test = loader.load_processed()
        if train is None:
            sys.exit("Processed dataset not found: run the 'process' stage first.")

    # (3) defect injection
    tag = config.injection_tag()
    noisy, fakes = None, set()
    if do_inject:
        result = NoiseOrchestrator(logger, config, loader.items, loader.rating_scale).apply(train, val, test)
        loader.save_noisy(tag, result, masked(config.injection))
        noisy = (result.splits["train"], result.splits.get("validation"), result.splits.get("test"))
        if not config.data.split.get("noise_in_validation"):
            noisy = (noisy[0], None, noisy[2])
        if not config.data.split.get("noise_in_test"):
            noisy = (noisy[0], noisy[1], None)
        fakes = result.fake_users

    # (4) evaluation
    if do_eval:
        if config.evaluation is None:
            sys.exit("The evaluation stage needs --evaluation (or --baselines for the default file).")
        evaluator = BaselinesOrchestrator(logger, config)
        if config.injection is not None and noisy is None:
            (tr_n, va_n, te_n), fakes = load_noisy(loader, tag)
            noisy = (tr_n, va_n, te_n)
        comp = evaluator.run((train, val, test), noisy, tag, fakes)
        if noisy is not None:
            create_unique_table(os.path.dirname(evaluator.results_dir))
            if comp is not None and not comp.empty:
                view = comp[comp["segment"] == "all"][["model", "metric", "clean_mean", "noisy_mean", "delta_pct"]]
                logger.info("Clean vs noisy (all users):\n" + view.to_string(index=False))

    logger.info("Done.")


if __name__ == "__main__":
    main()
