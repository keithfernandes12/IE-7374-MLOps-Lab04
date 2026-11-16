# IE 7374 MLOps: Airflow Lab 1 submission

**Repo:** https://github.com/keithfernandes12/IE-7374-MLOps-Lab04

**Commits:**
- [`0297d8c`](https://github.com/keithfernandes12/IE-7374-MLOps-Lab04/commit/0297d8cd8f727b1abfab60a123318879d40105e5): baseline redo + customer segmentation pipeline
- [`8888e49`](https://github.com/keithfernandes12/IE-7374-MLOps-Lab04/commit/8888e49eff7bcfe64616eb1d1669731853f49671): Airflow UI screenshots in the README

## What I did

1. **Redid the original lab** (`Airflow_Lab1`) on Apache Airflow 3.4 in Docker: four tasks that load the credit-card data, preprocess it, run a K-Means elbow sweep, and load the model to predict. While redoing it, I found two bugs. It saves the k=49 model instead of the elbow k (8), and it predicts on unscaled test data even though the model was trained on scaled data.
2. **Built my own extended DAG** (`Customer_Segmentation_Pipeline`) that fixes these bugs and adds:
   - TaskFlow API, a weekly schedule, and runtime `Params` (features, k range, how k is chosen, quality threshold)
   - A data validation task that stops the run if the input data is bad
   - 8 features with median imputation, log transform and standard scaling, saved together with KMeans as one sklearn Pipeline
   - k chosen by the elbow method or silhouette score, with a diagnostic plot
   - Evaluation metrics and a **quality-gate branch** that promotes the model to `production/` (with a model card) or flags it for review
   - Named customer segments, scoring of 5 hand-made customer profiles, and a Markdown report for each run
   - 9 pytest unit tests

## Results

k = 4 (the elbow and silhouette methods agree), silhouette = 0.268, so the model is promoted. I tested both outcomes of the gate: a run with `min_silhouette=0.5` correctly skips promotion.

## Lesson learned

In Airflow 3, `@task.branch` doesn't push its return value to XCom. I found this while debugging a report that always said "flagged."

The [README](README.md) has setup steps, a side-by-side comparison with the original, and UI screenshots.
