"""Unit tests for dags/src/segmentation.py (no Airflow needed)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dags"))
from src import segmentation as seg  # noqa: E402


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    """Run the whole pipeline once on the real data; tests inspect the outputs."""
    run_dir = tmp_path_factory.mktemp("run")
    feats = seg.DEFAULT_FEATURES
    validation = seg.validate_data(seg.TRAIN_CSV, feats, run_dir)
    prep = seg.preprocess(seg.TRAIN_CSV, feats, run_dir)
    k_info = seg.sweep_k(prep["X_path"], run_dir, 2, 8)
    model = seg.train_final_model(prep["preprocessor_path"], prep["X_path"], k_info["chosen_k"], run_dir)
    metrics = seg.evaluate_model(prep["X_path"], model["labels_path"], run_dir)
    profile = seg.profile_segments(seg.TRAIN_CSV, feats, model["labels_path"], run_dir)
    preds = seg.predict_new_customers(model["model_path"], seg.NEW_CUSTOMERS_CSV, feats,
                                      profile["segment_names"], run_dir)
    report = seg.write_report(run_dir, "test_run", k_info, metrics, "promote_model", profile, preds)
    return dict(run_dir=run_dir, validation=validation, prep=prep, k_info=k_info,
                model=model, metrics=metrics, profile=profile, preds=preds, report=report)


def test_validation_passes_on_real_data(run):
    assert run["validation"]["passed"]
    assert run["validation"]["rows"] == 8950


def test_validation_rejects_missing_column(tmp_path):
    with pytest.raises(ValueError, match="missing columns"):
        seg.validate_data(seg.TRAIN_CSV, ["BALANCE", "NOT_A_COLUMN"], tmp_path)


def test_preprocess_keeps_every_row(run):
    # baseline dropped rows with any NaN; we impute instead
    assert run["prep"]["n_rows"] == 8950
    assert run["prep"]["n_features"] == len(seg.DEFAULT_FEATURES)


def test_k_choice_in_range_and_plot_saved(run):
    k = run["k_info"]
    assert 2 <= k["chosen_k"] <= 8
    assert Path(k["plot_path"]).stat().st_size > 0


def test_model_has_chosen_k_and_metrics_are_sane(run):
    assert run["model"]["k"] == run["k_info"]["chosen_k"]
    m = run["metrics"]
    assert -1 <= m["silhouette"] <= 1
    assert sum(m["cluster_sizes"].values()) == 8950


def test_saved_pipeline_predicts_on_raw_rows(run):
    """The upstream bug: predicting on unscaled data. Our Pipeline handles raw input."""
    model = seg._load_pickle(run["model"]["model_path"])
    raw = pd.read_csv(seg.TRAIN_CSV)[seg.DEFAULT_FEATURES].head(50)
    labels = seg.pd.Series(seg.np.load(run["model"]["labels_path"])[:50])
    assert (model.predict(raw) == labels.values).all()


def test_every_cluster_named_and_new_customers_scored(run):
    names = run["profile"]["segment_names"]
    assert len(names) == run["k_info"]["chosen_k"]
    assignments = run["preds"]["assignments"]
    assert len(assignments) == 5
    assert all(a["segment_name"] in names.values() for a in assignments)


def test_quality_gate():
    assert seg.passes_quality_gate({"silhouette": 0.3, "smallest_cluster_frac": 0.1}, 0.2)
    assert not seg.passes_quality_gate({"silhouette": 0.1, "smallest_cluster_frac": 0.1}, 0.2)
    assert not seg.passes_quality_gate({"silhouette": 0.5, "smallest_cluster_frac": 0.001}, 0.2)


def test_report_written(run):
    text = Path(run["report"]).read_text(encoding="utf-8")
    assert "Chosen k" in text and "New customer assignments" in text
