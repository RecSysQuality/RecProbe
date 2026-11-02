# RecProbe

A data perturbation framework for robustness analysis of recommender systems.
RecProbe injects the data defects of the taxonomy below into a training set,
stores the perturbed dataset as a reusable artifact, and evaluates recommendation
models from RecBole, Cornac and Reviews4Rec on clean and perturbed data, with
accuracy and beyond-accuracy metrics, multiple seeds and user segments.

## Taxonomy (Table 1) and perturbations

| Corrupted signal | Point anomaly | Contextual anomaly | Collective anomaly |
|---|---|---|---|
| **A – Interaction** | `invalid_timestamp` | `sequence_corruption` | `interaction_burst`, `interaction_shilling` |
| **B – Rating** | `invalid_rating` | `history_inconsistent_rating`, `rating_flip` | `rating_burst`, `rating_shilling` |
| **C – Review** | `uninformative_review`, `off_topic_review` | `context_swap`, `sentiment_flip` | `review_burst`, `review_shilling` |

General-purpose operations outside the taxonomy: `record_removal`, `metadata_modification`.

`python main.py --list` prints the table with a description of each perturbation.
Every perturbation has a template in `config/files/injection/<name>.yaml`
(regenerate with `python main.py --write-templates`).

Names used by the previous version of the code are still accepted:
`misclick` → `sequence_corruption`, `implicit_burst` → `interaction_burst`,
`shilling_attack` → `rating_shilling`, `profile_inconsistent` → `sequence_corruption`
(`--profile interaction`) or `history_inconsistent_rating` (`--profile rating`).

## Pipeline

```
(1) setup and configuration -> (2) data processing -> (3) defect injection -> (4) evaluation
```

Each phase has its own YAML file:

| Phase | File | Content |
|---|---|---|
| 2 | `config/files/data.yaml` | input/output, field mapping, k-core, rating range, duplicates, review length, split |
| 3 | `config/files/injection/*.yaml` | budget, target selection, per-target limits, ratings, time window, fake users, LLM |
| 4 | `config/files/evaluation.yaml` | library, models and hyperparameters, metrics, beyond-accuracy, segments, seeds |

```bash
pip install -r requirements.txt

python main.py --data config/files/data.yaml \
               --injection config/files/examples/review_burst_figure3.yaml \
               --evaluation config/files/evaluation.yaml

# previous command line
python main.py --profile rating --noise_injection rating_flip --baselines

# single stages
python main.py --data ... --stage process
python main.py --data ... --injection ... --stage inject
python main.py --data ... --injection ... --evaluation ... --stage evaluate

# web interface (writes the same three YAML files)
streamlit run streamlit_app/app.py
```

### (2) Data processing
* Amazon Reviews (2023 and older field names) and Yelp presets; custom CSV/JSON/JSONL.
  Only `user_id` and `item_id` are required; `timestamp`, `rating`, `review_text` are optional.
  Other column names are mapped with `fields:`. Optional item metadata (`input.items`) provides
  categories and titles used by some perturbations.
* Filters, applied before the split: duplicate removal, rating range, minimum review length, iterative k-core.
* Splits: `random_holdout`, `temporal_leave_one_out`, `global_temporal`, `random_leave_one_out`,
  or a custom function `package.module:function` with signature `f(df, split_cfg, rng) -> (train, val, test)`.

### (3) Defect injection
* `budget`: absolute number of records, or fraction of the split's edges (`0.3`, `"30%"`).
  It counts the records added or corrupted on the targets; the filler records of fake profiles
  are reported separately in `manifest.json`.
* `target` (`user`/`item`), `selection_strategy` (`uniform`, `top`, `bottom`), `top_k`, `target_ids`.
* `min_per_node` / `max_per_node` (Figure 3 names such as `max_reviews_per_node` are accepted).
* `rating_behavior`: `uniform` on `[min_rating, max_rating]` or `gaussian` on the target's ratings,
  clipped and mapped to the rating scale.
* `temporal_behavior`: interval `start_timestamp`–`end_timestamp` and burst `window`.
* `fake_users`: `use_fake_users`, `fake_behavior: random | mimic`.
* `llm_configuration`: `backend` (`vllm`, `transformers`, or `template` for offline tests only),
  `model`, `huggingface_token`, `temperature`, `prompt` (placeholders `{item}`, `{title}`,
  `{category}`, `{rating}`, `{polarity}`, `{review}`).
* Perturbations are applied to the training split only, unless `split.noise_in_validation` /
  `split.noise_in_test` are set.

Output in `<output>/noisy/<perturbation>_b<budget>_<hash>/`: perturbed splits,
`injection_log.tsv` (every added / modified / removed record with old and new values),
`fake_users.txt`, `injection.yaml` and `manifest.json`.

### (4) Evaluation
* Libraries: Cornac and RecBole (models, hyperparameters and metrics passed as they are;
  a native RecBole YAML can be given with `native_config`; the runner includes the fixes needed by
  RecBole 1.2 with NumPy 2 and PyTorch >= 2.6), Reviews4Rec through a command
  adapter (see `src/baselines/reviews4rec/reviews4rec_runner.py`), and custom models
  (`src/baselines/custom/custom_runner.py`).
* Accuracy comes from the library. Beyond-accuracy (coverage, novelty, entropy, Gini) is computed by
  RecProbe from the exported ranked lists, so it is comparable across libraries.
* Each run is repeated over `seeds`; mean and standard deviation are reported.
* Segments: percentile thresholds (`user_segments: 0 25 75`) or explicit id lists
  (`user_segment_ids`), computed once on the clean training set; fake users are excluded.
* Results: `results/<dataset>/<tag>/comparison.tsv` (clean vs noisy, delta and delta %),
  per-seed `raw.tsv`, and a unified table `results/comparison_<timestamp>.tsv`.
  Clean results are cached and reused across perturbations.

## Extending
* New perturbation: subclass `injectors.base.Perturbation` (or `BurstPerturbation` /
  `ShillingPerturbation`), set `name`, `signal`, `observability`, `operation`, `DEFAULTS`,
  implement `perturb(df)` and decorate with `@register`.
* New library: subclass `src.baselines.base_runner.BaseRunner` and add it to `RUNNERS`.
* New metric: `custom_metrics: ["package.module:function"]` in evaluation.yaml.

## Tests
`python tests/make_synthetic.py` creates a small Amazon-like dataset; the YAML files in
`tests/` run the pipeline on it (review generation with the offline `template` backend).
