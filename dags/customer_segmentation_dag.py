"""
Customer_Segmentation_Pipeline: my extended version of Airflow Lab 1.

Baseline (airflow_lab1.py)            ->  This DAG
-----------------------------------------------------------------------------
classic PythonOperator + op_args      ->  TaskFlow API (@task / @task.branch)
manual trigger only                   ->  weekly schedule + manual runs with Params
pickled DataFrames passed over XCom   ->  file paths + metrics passed over XCom
no input checks                       ->  validate_data fails the run early
3 features, dropna, MinMax            ->  8 features, median impute, log1p, standard scale
model saved at k=49                   ->  model saved at the chosen k (elbow or silhouette)
predicts on unscaled test data        ->  one saved Pipeline (preprocessing + KMeans)
no evaluation                         ->  silhouette / DB / CH metrics + quality-gate branch
single int printed                    ->  named segments, scored new customers, markdown report
"""
from __future__ import annotations

from datetime import datetime, timedelta

from airflow.providers.standard.operators.bash import BashOperator
from airflow.sdk import Param, dag, get_current_context, task

from src import segmentation as seg

default_args = {
    "owner": "Keith Fernandes",
    "retries": 1,
    "retry_delay": timedelta(minutes=2),
}


@dag(
    dag_id="Customer_Segmentation_Pipeline",
    description="Credit-card customer segmentation with validation, k selection, quality gate and reporting",
    start_date=datetime(2026, 11, 1),
    schedule="0 6 * * 1",  # every Monday 06:00 UTC
    catchup=False,
    default_args=default_args,
    tags=["ie7374", "lab1", "clustering"],
    params={
        "features": Param(seg.DEFAULT_FEATURES, type="array",
                          description="Columns of file.csv to cluster on"),
        "k_min": Param(2, type="integer", minimum=2),
        "k_max": Param(10, type="integer", minimum=3, maximum=30),
        "k_method": Param("elbow", enum=["elbow", "silhouette"],
                          description="How to pick k from the sweep"),
        "min_silhouette": Param(0.20, type="number", minimum=0, maximum=1,
                                description="Quality gate: minimum silhouette to promote"),
    },
    doc_md=__doc__,
)
def customer_segmentation_pipeline():

    @task
    def make_run_dir() -> str:
        ctx = get_current_context()
        return seg.make_run_dir(ctx["run_id"])

    @task
    def validate_data(run_dir: str) -> dict:
        p = get_current_context()["params"]
        return seg.validate_data(seg.TRAIN_CSV, p["features"], run_dir)

    @task
    def preprocess(run_dir: str, validation: dict) -> dict:
        return seg.preprocess(seg.TRAIN_CSV, validation["features"], run_dir)

    @task
    def choose_k(run_dir: str, prep: dict) -> dict:
        p = get_current_context()["params"]
        k_info = seg.sweep_k(prep["X_path"], run_dir, p["k_min"], p["k_max"], p["k_method"])
        print(f"elbow k={k_info['elbow_k']}, best silhouette k={k_info['best_silhouette_k']}, "
              f"chosen k={k_info['chosen_k']}")
        return k_info

    @task
    def train_model(run_dir: str, prep: dict, k_info: dict) -> dict:
        return seg.train_final_model(prep["preprocessor_path"], prep["X_path"],
                                     k_info["chosen_k"], run_dir)

    @task
    def evaluate(run_dir: str, prep: dict, model: dict) -> dict:
        metrics = seg.evaluate_model(prep["X_path"], model["labels_path"], run_dir)
        print(f"metrics: {metrics}")
        return metrics

    @task.branch
    def quality_gate(metrics: dict) -> str:
        threshold = get_current_context()["params"]["min_silhouette"]
        ok = seg.passes_quality_gate(metrics, threshold)
        print(f"silhouette={metrics['silhouette']} vs threshold={threshold} -> "
              f"{'promote' if ok else 'flag'}")
        return "promote_model" if ok else "flag_for_review"

    @task
    def promote_model(model: dict, metrics: dict, k_info: dict) -> dict:
        return seg.promote_model(model["model_path"], metrics, k_info,
                                 get_current_context()["run_id"])

    @task
    def flag_for_review(metrics: dict) -> None:
        print(f"Model NOT promoted; production model left unchanged. metrics={metrics}")

    @task
    def profile_segments(run_dir: str, prep: dict, model: dict) -> dict:
        return seg.profile_segments(seg.TRAIN_CSV, prep["features"], model["labels_path"], run_dir)

    @task
    def score_new_customers(run_dir: str, prep: dict, model: dict, profile: dict) -> dict:
        preds = seg.predict_new_customers(model["model_path"], seg.NEW_CUSTOMERS_CSV,
                                          prep["features"], profile["segment_names"], run_dir)
        for row in preds["assignments"]:
            print(row)
        return preds

    # Runs after whichever branch was taken (the other one is skipped).
    @task(trigger_rule="none_failed_min_one_success")
    def write_report(run_dir: str, k_info: dict, metrics: dict,
                     profile: dict, preds: dict) -> str:
        ctx = get_current_context()
        # A @task.branch doesn't push its return value to XCom in Airflow 3, so
        # check whether promote_model ran: it only pushes a model card if it did.
        promoted = ctx["ti"].xcom_pull(task_ids="promote_model")
        decision = "promote_model" if promoted else "flag_for_review"
        return seg.write_report(run_dir, ctx["run_id"],
                                k_info, metrics, decision, profile, preds)

    # ---- wiring ----------------------------------------------------------- #
    run_dir = make_run_dir()
    validation = validate_data(run_dir)
    prep = preprocess(run_dir, validation)
    k_info = choose_k(run_dir, prep)
    model = train_model(run_dir, prep, k_info)
    metrics = evaluate(run_dir, prep, model)

    decision = quality_gate(metrics)
    promoted = promote_model(model, metrics, k_info)
    flagged = flag_for_review(metrics)
    decision >> [promoted, flagged]

    profile = profile_segments(run_dir, prep, model)
    preds = score_new_customers(run_dir, prep, model, profile)

    report = write_report(run_dir, k_info, metrics, profile, preds)
    [promoted, flagged] >> report

    list_artifacts = BashOperator(
        task_id="list_artifacts",
        bash_command="echo 'Artifacts for run {{ run_id }}:' && ls -la {{ ti.xcom_pull(task_ids='make_run_dir') }}",
    )
    report >> list_artifacts


customer_segmentation_pipeline()
