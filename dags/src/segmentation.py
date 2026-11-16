"""
Customer segmentation pipeline: my extension of MLOps Airflow Lab 1.

Every function here is plain Python (no Airflow imports), so the logic can be
unit-tested locally and the DAG file stays a thin orchestration layer.

Changes vs. the baseline `src/lab.py`:
  * Tasks exchange small JSON metadata (file paths and metrics) over XCom
    instead of base64-pickled DataFrames.
  * Preprocessing uses median imputation (not dropna), log1p on skewed dollar
    columns, and StandardScaler, all wrapped in one sklearn ColumnTransformer.
  * The saved model is a full sklearn Pipeline (preprocessor + KMeans fitted at
    the *chosen* k). The baseline saved the k=49 model and ran it on unscaled
    test data, so its predictions did not match how it was trained.
  * k is chosen by the elbow method or by silhouette score, and a diagnostic
    plot is saved for each run.
  * Clusters are profiled and given readable names, then used to score a
    small file of new customers.
"""
from __future__ import annotations

import json
import os
import pickle
import shutil
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
TRAIN_CSV = DATA_DIR / "file.csv"
NEW_CUSTOMERS_CSV = DATA_DIR / "new_customers.csv"

DEFAULT_FEATURES = [
    "BALANCE",
    "PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
    "PURCHASES_FREQUENCY",
    "PRC_FULL_PAYMENT",
]

# Dollar-valued columns are heavily right-skewed, so they get log1p before scaling.
MONEY_COLUMNS = {
    "BALANCE",
    "PURCHASES",
    "ONEOFF_PURCHASES",
    "INSTALLMENTS_PURCHASES",
    "CASH_ADVANCE",
    "CREDIT_LIMIT",
    "PAYMENTS",
    "MINIMUM_PAYMENTS",
}

FRIENDLY_NAMES = {
    "BALANCE": "Balance",
    "PURCHASES": "Spend",
    "ONEOFF_PURCHASES": "One-off Spend",
    "INSTALLMENTS_PURCHASES": "Installment Spend",
    "CASH_ADVANCE": "Cash Advance",
    "CREDIT_LIMIT": "Credit Limit",
    "PAYMENTS": "Payments",
    "MINIMUM_PAYMENTS": "Min Payments",
    "PURCHASES_FREQUENCY": "Purchase Frequency",
    "PRC_FULL_PAYMENT": "Full-Payment Rate",
    "TENURE": "Tenure",
}

RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def default_workdir() -> Path:
    """Where run artifacts go: the docker-mounted working_data dir if present."""
    env = os.environ.get("SEGMENTATION_WORKDIR")
    if env:
        return Path(env)
    container_dir = Path("/opt/airflow/working_data")
    if container_dir.exists():
        return container_dir / "segmentation"
    return Path(__file__).resolve().parents[2] / "working_data" / "segmentation"


def make_run_dir(run_id: str, base: str | Path | None = None) -> str:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in run_id)
    run_dir = Path(base or default_workdir()) / "runs" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    return str(run_dir)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _load_pickle(path: str | Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _dump_pickle(obj, path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


# --------------------------------------------------------------------------- #
# 1. validation
# --------------------------------------------------------------------------- #
def validate_data(
    csv_path: str | Path,
    features: list[str],
    run_dir: str | Path,
    min_rows: int = 500,
    max_null_frac: float = 0.10,
) -> dict:
    """Fail fast if the input is not fit to train on. Writes validation.json."""
    df = pd.read_csv(csv_path)
    problems: list[str] = []

    missing = [c for c in features if c not in df.columns]
    if missing:
        problems.append(f"missing columns: {missing}")
    present = [c for c in features if c in df.columns]

    if len(df) < min_rows:
        problems.append(f"only {len(df)} rows (< {min_rows})")

    null_frac = df[present].isna().mean().round(4).to_dict()
    too_null = {c: f for c, f in null_frac.items() if f > max_null_frac}
    if too_null:
        problems.append(f"null fraction above {max_null_frac}: {too_null}")

    non_numeric = [c for c in present if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        problems.append(f"non-numeric columns: {non_numeric}")

    negative = [
        c for c in present
        if c in MONEY_COLUMNS and pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any()
    ]
    if negative:
        problems.append(f"negative dollar values in: {negative}")

    report = {
        "csv": str(csv_path),
        "rows": int(len(df)),
        "features": features,
        "null_fraction": null_frac,
        "passed": not problems,
        "problems": problems,
    }
    _write_json(Path(run_dir) / "validation.json", report)
    if problems:
        raise ValueError("Data validation failed: " + "; ".join(problems))
    return report


# --------------------------------------------------------------------------- #
# 2. preprocessing
# --------------------------------------------------------------------------- #
def build_preprocessor(features: list[str]) -> ColumnTransformer:
    money = [c for c in features if c in MONEY_COLUMNS]
    other = [c for c in features if c not in MONEY_COLUMNS]
    transformers = []
    if money:
        transformers.append((
            "money",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
                ("scale", StandardScaler()),
            ]),
            money,
        ))
    if other:
        transformers.append((
            "other",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            other,
        ))
    return ColumnTransformer(transformers, remainder="drop")


def preprocess(csv_path: str | Path, features: list[str], run_dir: str | Path) -> dict:
    """Fit the preprocessor and save it with the transformed matrix."""
    run_dir = Path(run_dir)
    df = pd.read_csv(csv_path)
    preprocessor = build_preprocessor(features)
    X = preprocessor.fit_transform(df[features])

    prep_path = run_dir / "preprocessor.pkl"
    X_path = run_dir / "X.npy"
    _dump_pickle(preprocessor, prep_path)
    np.save(X_path, X)
    return {
        "preprocessor_path": str(prep_path),
        "X_path": str(X_path),
        "n_rows": int(X.shape[0]),
        "n_features": int(X.shape[1]),
        "features": features,
    }


# --------------------------------------------------------------------------- #
# 3. choose k
# --------------------------------------------------------------------------- #
def sweep_k(
    X_path: str | Path,
    run_dir: str | Path,
    k_min: int = 2,
    k_max: int = 12,
    method: str = "elbow",
    silhouette_sample: int = 3000,
) -> dict:
    """Fit KMeans for k in [k_min, k_max]; pick k by elbow or silhouette."""
    if k_min < 2 or k_max <= k_min:
        raise ValueError(f"need 2 <= k_min < k_max, got {k_min}, {k_max}")
    if method not in {"elbow", "silhouette"}:
        raise ValueError(f"method must be 'elbow' or 'silhouette', got {method!r}")

    X = np.load(X_path)
    ks = list(range(k_min, k_max + 1))
    sse, sil = [], []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
        sse.append(float(km.inertia_))
        sil.append(float(silhouette_score(
            X, km.labels_,
            sample_size=min(silhouette_sample, len(X)),
            random_state=RANDOM_STATE,
        )))

    elbow = KneeLocator(ks, sse, curve="convex", direction="decreasing").elbow
    best_sil_k = ks[int(np.argmax(sil))]
    if method == "silhouette" or elbow is None:
        chosen = best_sil_k
    else:
        chosen = int(elbow)

    plot_path = Path(run_dir) / "k_selection.png"
    _plot_k_selection(ks, sse, sil, elbow, best_sil_k, chosen, plot_path)

    return {
        "k_values": ks,
        "sse": sse,
        "silhouette": sil,
        "elbow_k": int(elbow) if elbow is not None else None,
        "best_silhouette_k": int(best_sil_k),
        "method": method,
        "chosen_k": int(chosen),
        "plot_path": str(plot_path),
    }


def _plot_k_selection(ks, sse, sil, elbow, best_sil_k, chosen, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(ks, sse, "o-", color="#1f6feb", label="SSE (inertia)")
    ax1.set_xlabel("k (number of clusters)")
    ax1.set_ylabel("SSE", color="#1f6feb")
    ax2 = ax1.twinx()
    ax2.plot(ks, sil, "s--", color="#d97706", label="Silhouette")
    ax2.set_ylabel("Silhouette score", color="#d97706")
    if elbow is not None:
        ax1.axvline(elbow, color="#1f6feb", alpha=0.3, lw=6, label=f"elbow k={elbow}")
    ax1.axvline(best_sil_k, color="#d97706", alpha=0.3, lw=6, label=f"best silhouette k={best_sil_k}")
    ax1.set_title(f"Choosing k: chosen k = {chosen}")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# 4. train + 5. evaluate
# --------------------------------------------------------------------------- #
def train_final_model(
    preprocessor_path: str | Path,
    X_path: str | Path,
    k: int,
    run_dir: str | Path,
) -> dict:
    """Fit KMeans at the chosen k and save preprocessor + KMeans as one Pipeline."""
    X = np.load(X_path)
    km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE).fit(X)
    pipeline = Pipeline([
        ("preprocess", _load_pickle(preprocessor_path)),
        ("kmeans", km),
    ])
    model_path = Path(run_dir) / "segmentation_model.pkl"
    labels_path = Path(run_dir) / "labels.npy"
    _dump_pickle(pipeline, model_path)
    np.save(labels_path, km.labels_)
    return {"model_path": str(model_path), "labels_path": str(labels_path), "k": int(k)}


def evaluate_model(X_path: str | Path, labels_path: str | Path, run_dir: str | Path) -> dict:
    X = np.load(X_path)
    labels = np.load(labels_path)
    sizes = pd.Series(labels).value_counts().sort_index()
    metrics = {
        "silhouette": round(float(silhouette_score(X, labels)), 4),
        "davies_bouldin": round(float(davies_bouldin_score(X, labels)), 4),
        "calinski_harabasz": round(float(calinski_harabasz_score(X, labels)), 1),
        "cluster_sizes": {int(k): int(v) for k, v in sizes.items()},
        "smallest_cluster_frac": round(float(sizes.min() / sizes.sum()), 4),
    }
    _write_json(Path(run_dir) / "metrics.json", metrics)
    return metrics


def passes_quality_gate(metrics: dict, min_silhouette: float, min_cluster_frac: float = 0.02) -> bool:
    """A model ships only if clusters are reasonably separated and none is tiny."""
    return (
        metrics["silhouette"] >= min_silhouette
        and metrics["smallest_cluster_frac"] >= min_cluster_frac
    )


def promote_model(model_path: str | Path, metrics: dict, k_info: dict, run_id: str,
                  base: str | Path | None = None) -> dict:
    """Copy the model into production/ with a metadata card."""
    prod = Path(base or default_workdir()) / "production"
    prod.mkdir(parents=True, exist_ok=True)
    dest = prod / "segmentation_model.pkl"
    shutil.copy2(model_path, dest)
    card = {
        "promoted_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_run_id": run_id,
        "k": k_info["chosen_k"],
        "k_method": k_info["method"],
        "metrics": metrics,
    }
    _write_json(prod / "model_card.json", card)
    return {"production_model": str(dest), **card}


# --------------------------------------------------------------------------- #
# 6. profile segments + 7. score new customers
# --------------------------------------------------------------------------- #
def profile_segments(
    csv_path: str | Path,
    features: list[str],
    labels_path: str | Path,
    run_dir: str | Path,
    z_threshold: float = 0.5,
) -> dict:
    """Per-cluster feature means, with each cluster named by its standout traits."""
    df = pd.read_csv(csv_path)[features]
    df = df.fillna(df.median(numeric_only=True))
    labels = np.load(labels_path)

    overall_mean, overall_std = df.mean(), df.std(ddof=0).replace(0, 1)
    profile = df.groupby(labels).mean()
    z = (profile - overall_mean) / overall_std

    names: dict[int, str] = {}
    for cid, row in z.iterrows():
        top = row[row.abs() >= z_threshold].abs().sort_values(ascending=False).index[:2]
        if len(top) == 0:
            label = "Average Customers"
        else:
            label = " + ".join(
                f"{'High' if row[c] > 0 else 'Low'} {FRIENDLY_NAMES.get(c, c)}" for c in top
            )
        names[int(cid)] = label

    out = profile.round(2)
    out.insert(0, "segment_name", [names[int(c)] for c in out.index])
    out.insert(1, "n_customers", pd.Series(labels).value_counts().sort_index().values)
    out.index.name = "cluster"
    profile_path = Path(run_dir) / "segment_profile.csv"
    out.to_csv(profile_path)
    return {"profile_path": str(profile_path), "segment_names": {str(k): v for k, v in names.items()}}


def predict_new_customers(
    model_path: str | Path,
    new_csv: str | Path,
    features: list[str],
    segment_names: dict,
    run_dir: str | Path,
) -> dict:
    """Score unseen customers with the saved Pipeline (preprocessing included)."""
    df = pd.read_csv(new_csv)
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"{new_csv} is missing feature columns {missing}")
    model = _load_pickle(model_path)
    clusters = model.predict(df[features])

    out = df[[c for c in ("CUST_ID", "PERSONA") if c in df.columns]].copy()
    out["cluster"] = clusters
    out["segment_name"] = [segment_names[str(int(c))] for c in clusters]
    pred_path = Path(run_dir) / "new_customer_predictions.csv"
    out.to_csv(pred_path, index=False)
    return {"predictions_path": str(pred_path), "assignments": out.to_dict(orient="records")}


# --------------------------------------------------------------------------- #
# 8. report
# --------------------------------------------------------------------------- #
def write_report(run_dir: str | Path, run_id: str, k_info: dict, metrics: dict,
                 gate_decision: str, profile: dict, predictions: dict) -> str:
    run_dir = Path(run_dir)
    prof = pd.read_csv(profile["profile_path"])
    lines = [
        f"# Customer Segmentation Run: `{run_id}`",
        "",
        f"- **Chosen k:** {k_info['chosen_k']} (method: {k_info['method']}; "
        f"elbow={k_info['elbow_k']}, best silhouette={k_info['best_silhouette_k']})",
        f"- **Silhouette:** {metrics['silhouette']}  |  **Davies-Bouldin:** {metrics['davies_bouldin']}"
        f"  |  **Calinski-Harabasz:** {metrics['calinski_harabasz']}",
        f"- **Quality gate:** {'PROMOTED to production' if gate_decision == 'promote_model' else 'FLAGGED for review (not promoted)'}",
        "",
        "![k selection](k_selection.png)",
        "",
        "## Segments",
        "",
        _md_table(prof),
        "",
        "## New customer assignments",
        "",
        _md_table(pd.DataFrame(predictions["assignments"])),
        "",
    ]
    path = run_dir / "report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _md_table(df: pd.DataFrame) -> str:
    cols = [str(c) for c in df.columns]
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        rows.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(rows)
