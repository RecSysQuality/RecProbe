"""
Checks that every perturbation behaves as defined in Section 3 / Table 1.
Run from the repository root:   python tests/make_synthetic.py && python -m pytest tests -q
"""

import os
import sys

import pandas as pd
import pytest
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from config.config import load_config  # noqa: E402
from data_handler.loader import DatasetLoader  # noqa: E402
from injectors.registry import taxonomy_table  # noqa: E402
from logger import get_logger  # noqa: E402
from orchestrator import NoiseOrchestrator  # noqa: E402

DATA = "tests/test_data.yaml"
LOG = get_logger("recprobe_tests", level=40)
TEMPLATE = {"llm_configuration": {"backend": "template"}}


@pytest.fixture(scope="module")
def splits():
    if not os.path.exists("data/input/synthetic/Synthetic.csv"):
        from tests.make_synthetic import main as make
        make("data/input/synthetic")
    loader = DatasetLoader(LOG, load_config(DATA))
    tr, va, te = loader.load_data()
    return loader, tr, va, te


def run(splits, tmp_path, name, params, budget=50):
    loader, tr, va, te = splits
    p = tmp_path / f"{name}.yaml"
    p.write_text(yaml.safe_dump({"budget": budget, name: params}))
    cfg = load_config(DATA, str(p))
    return NoiseOrchestrator(LOG, cfg, loader.items, loader.rating_scale).apply(tr, va, te), tr


@pytest.mark.parametrize("name", [r["perturbation"] for r in taxonomy_table()])
def test_every_perturbation_runs(splits, tmp_path, name):
    params = dict(TEMPLATE) if name in {"off_topic_review", "sentiment_flip", "review_burst",
                                        "review_shilling"} else {}
    res, tr = run(splits, tmp_path, name, params)
    assert res.summary["splits"]["train"]["target_records"] > 0
    assert not res.log.empty


def test_rating_flip(splits, tmp_path):
    res, _ = run(splits, tmp_path, "rating_flip", {})
    assert (res.log.old_value + res.log.new_value == 6).all()      # 1-5 scale
    assert (res.log.old_value != 3).all()                          # neutral ratings skipped


def test_invalid_timestamp_outside_range(splits, tmp_path):
    res, tr = run(splits, tmp_path, "invalid_timestamp", {"modes": ["future", "epoch", "sentinel", "pre_platform"]})
    v = res.log.new_value
    assert ((v > tr.timestamp.max()) | (v < tr.timestamp.min())).all()


def test_history_inconsistent(splits, tmp_path):
    res, tr = run(splits, tmp_path, "history_inconsistent_rating", {"min_deviation": 2.0})
    mean = tr.groupby("user_id").rating.mean()
    assert ((res.log.new_value - res.log.user_id.map(mean)).abs() >= 2.0 - 1e-9).all()


def test_sequence_corruption_inside_history(splits, tmp_path):
    res, tr = run(splits, tmp_path, "sequence_corruption", {})
    out = res.splits["train"]
    new = out.loc[res.log.record_id]
    rng = tr.groupby("user_id").timestamp.agg(["min", "max"])
    assert all(rng.at[u, "min"] <= t <= rng.at[u, "max"] for u, t in zip(new.user_id, new.timestamp))
    assert not out.duplicated(["user_id", "item_id"]).any()


def test_burst_window(splits, tmp_path):
    tb = {"start_timestamp": "2020-01-01", "end_timestamp": "2020-12-31",
          "window": {"days": 0, "hours": 10, "minutes": 0}}
    res, _ = run(splits, tmp_path, "rating_burst",
                 {"temporal_behavior": tb, "rating_behavior": {"min_rating": 1, "max_rating": 2}})
    new = res.splits["train"].loc[res.log[res.log.role == "target"].record_id]
    assert (new.groupby("item_id").timestamp.agg(lambda s: s.max() - s.min()) <= 10 * 3600).all()
    assert set(new.rating.unique()) <= {1.0, 2.0}
    assert new.user_id.str.startswith("fake").all()


@pytest.mark.parametrize("attack", ["random", "average", "bandwagon", "segment", "love_hate"])
@pytest.mark.parametrize("direction", ["push", "nuke"])
def test_shilling_targets(splits, tmp_path, attack, direction):
    res, _ = run(splits, tmp_path, "rating_shilling",
                 {"attack_type": attack, "direction": direction, "num_targets": 2, "filler_size": 5}, budget=20)
    out = res.splits["train"]
    tg = out.loc[res.log[res.log.role == "target"].record_id]
    assert set(tg.rating) == {5.0 if direction == "push" else 1.0}
    assert tg.item_id.nunique() == 2 and len(res.fake_users) == 10   # every profile hits every target


def test_sentiment_flip_keeps_rating(splits, tmp_path):
    res, tr = run(splits, tmp_path, "sentiment_flip", {"strategy": "donor"})
    assert (res.log.field == "review_text").all()
    out = res.splits["train"]
    assert (out.loc[res.log.record_id, "rating"].values == tr.loc[res.log.record_id, "rating"].values).all()


def test_train_only_by_default(splits, tmp_path):
    loader, tr, va, te = splits
    res, _ = run(splits, tmp_path, "interaction_burst", {})
    assert res.splits["test"] is te and res.splits["validation"] is va
    assert set(res.log.split) == {"train"}


def test_reproducible(splits, tmp_path):
    a, _ = run(splits, tmp_path, "rating_shilling", {"attack_type": "average"})
    b, _ = run(splits, tmp_path, "rating_shilling", {"attack_type": "average"})
    pd.testing.assert_frame_equal(a.splits["train"], b.splits["train"])


def test_global_temporal_no_leakage(tmp_path):
    cfg = yaml.safe_load(open(DATA))
    cfg["split"]["strategy"] = "global_temporal"
    p = tmp_path / "gt.yaml"
    p.write_text(yaml.safe_dump(cfg))
    tr, va, te = DatasetLoader(LOG, load_config(str(p))).load_data()
    assert tr.timestamp.max() <= va.timestamp.min() <= te.timestamp.min()


def test_beyond_accuracy_known_values():
    from src.evaluation.metrics import beyond_accuracy
    ranked = pd.DataFrame({"user_id": ["a", "a", "b", "b"], "item_id": ["x", "y", "x", "y"], "rank": [1, 2, 1, 2]})
    pop = pd.Series({"x": 2, "y": 1, "z": 1})
    m = beyond_accuracy(ranked, {"a", "b"}, 2, ["x", "y", "z"], pop, 4)
    assert m["coverage@2"] == pytest.approx(2 / 3)
    assert m["entropy@2"] == pytest.approx(1.0)
    assert m["novelty@2"] == pytest.approx((1 + 2) / 2)          # -log2(2/4), -log2(1/4)
    assert m["gini@2"] == pytest.approx(0.5)                      # counts (0, 2, 2)
