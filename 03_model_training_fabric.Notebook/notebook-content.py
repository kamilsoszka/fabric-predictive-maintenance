# Fabric notebook source

# METADATA ********************

# META {
# META   "kernel_info": {
# META     "name": "synapse_pyspark"
# META   },
# META   "dependencies": {
# META     "lakehouse": {
# META       "default_lakehouse": "6a34805e-db4c-43fa-853e-c8cf68b8b23a",
# META       "default_lakehouse_name": "lh_predictive_maintenance",
# META       "default_lakehouse_workspace_id": "62823219-97ec-413a-b9df-33901ded29df",
# META       "known_lakehouses": [
# META         {
# META           "id": "6a34805e-db4c-43fa-853e-c8cf68b8b23a"
# META         }
# META       ]
# META     }
# META   }
# META }

# MARKDOWN ********************

# # 03_model_training
# 
# Chronological model training, validation-based threshold selection, final test evaluation, and MLflow registration.

# CELL ********************

from pyspark.sql import functions as F
import json
import tempfile
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42
REGISTERED_MODEL_NAME = "predictive_maintenance_lightgbm"
DEPLOYMENT_STATUS = "champion"
MINIMUM_PRECISION = 0.25
FEATURE_TABLE = "gold_machine_features_daily"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_df = spark.table(FEATURE_TABLE)
feature_dates = [r["feature_date"] for r in feature_df.select("feature_date").distinct().orderBy("feature_date").collect()]
assert len(feature_dates) >= 30, "Insufficient dates for temporal model validation"

train_count = int(len(feature_dates) * 0.70)
validation_count = int(len(feature_dates) * 0.15)
train_end = feature_dates[train_count - 1]
validation_end = feature_dates[train_count + validation_count - 1]

split_df = (
    feature_df.withColumn(
        "dataset_split",
        F.when(F.col("feature_date") <= F.lit(train_end), "train")
         .when(F.col("feature_date") <= F.lit(validation_end), "validation")
         .otherwise("test")
    )
)

split_summary = split_df.groupBy("dataset_split").agg(
    F.count("*").alias("row_count"),
    F.min("feature_date").alias("minimum_date"),
    F.max("feature_date").alias("maximum_date"),
    F.sum("failure_within_7d").alias("positive_rows")
)
display(split_summary.orderBy("minimum_date"))

(split_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable("gold_machine_features_split"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

excluded = {"machine_id", "feature_date", "failure_within_7d", "dataset_split", "is_active"}
categorical_columns = ["machine_type", "manufacturer", "location"]
numeric_columns = [
    name for name, dtype in split_df.dtypes
    if name not in excluded and name not in categorical_columns
    and (dtype in {"tinyint", "smallint", "int", "bigint", "float", "double"} or dtype.startswith("decimal"))
]
feature_columns = categorical_columns + numeric_columns

def to_pandas(split_name):
    return (split_df.filter(F.col("dataset_split") == split_name)
            .select(*feature_columns, "failure_within_7d").toPandas())

train_pdf = to_pandas("train")
validation_pdf = to_pandas("validation")
test_pdf = to_pandas("test")

X_train, y_train = train_pdf[feature_columns], train_pdf["failure_within_7d"].astype(int)
X_validation, y_validation = validation_pdf[feature_columns], validation_pdf["failure_within_7d"].astype(int)
X_test, y_test = test_pdf[feature_columns], test_pdf["failure_within_7d"].astype(int)

numeric_preprocessor = Pipeline([("imputer", SimpleImputer(strategy="median"))])
categorical_preprocessor = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
preprocessor = ColumnTransformer([
    ("numeric", numeric_preprocessor, numeric_columns),
    ("categorical", categorical_preprocessor, categorical_columns)
])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

def calculate_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, predictions, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
    }

def select_threshold(y_true, probabilities, minimum_precision):
    candidates = sorted(set(float(v) for v in probabilities))
    rows = [calculate_metrics(y_true, probabilities, t) for t in candidates]
    eligible = [r for r in rows if r["precision"] >= minimum_precision]
    assert eligible, "No threshold satisfies the minimum precision policy"
    return max(eligible, key=lambda r: (r["f1"], r["recall"], r["threshold"])), pd.DataFrame(rows)

baseline = Pipeline([
    ("preprocessor", ColumnTransformer([
        ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), numeric_columns),
        ("categorical", categorical_preprocessor, categorical_columns)
    ])),
    ("classifier", LogisticRegression(class_weight="balanced", max_iter=1000, random_state=RANDOM_STATE))
])
baseline.fit(X_train, y_train)
baseline_probabilities = baseline.predict_proba(X_validation)[:, 1]
baseline_metrics = calculate_metrics(y_validation, baseline_probabilities, 0.50)

champion_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LGBMClassifier(
        objective="binary", n_estimators=150, learning_rate=0.03,
        num_leaves=15, max_depth=6, min_child_samples=80,
        colsample_bytree=0.80, reg_alpha=0.2, reg_lambda=1.0,
        class_weight="balanced", random_state=RANDOM_STATE,
        n_jobs=-1, verbosity=-1
    ))
])
champion_pipeline.fit(X_train, y_train)
validation_probabilities = champion_pipeline.predict_proba(X_validation)[:, 1]
selected_policy, threshold_grid = select_threshold(y_validation, validation_probabilities, MINIMUM_PRECISION)
SELECTED_THRESHOLD = selected_policy["threshold"]

display(pd.DataFrame([{"model": "Logistic Regression", **baseline_metrics}, {"model": "LightGBM", **selected_policy}]))
print(f"Selected threshold: {SELECTED_THRESHOLD:.6f}")
print(f"Policy: maximize validation F1 with precision >= {MINIMUM_PRECISION:.2f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Use the untouched test set exactly once.
test_probabilities = champion_pipeline.predict_proba(X_test)[:, 1]
test_metrics = calculate_metrics(y_test, test_probabilities, SELECTED_THRESHOLD)
display(pd.DataFrame([{"dataset": "test", **test_metrics}]))

# Quantify overfitting explicitly.
train_probabilities = champion_pipeline.predict_proba(X_train)[:, 1]
train_roc_auc = float(roc_auc_score(y_train, train_probabilities))
validation_roc_auc = float(roc_auc_score(y_validation, validation_probabilities))
roc_auc_gap = train_roc_auc - validation_roc_auc
print(f"Train ROC-AUC: {train_roc_auc:.4f}")
print(f"Validation ROC-AUC: {validation_roc_auc:.4f}")
print(f"ROC-AUC gap: {roc_auc_gap:.4f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

signature = infer_signature(X_train, champion_pipeline.predict_proba(X_train)[:, 1])
configuration = {
    "registered_model_name": REGISTERED_MODEL_NAME,
    "deployment_status": DEPLOYMENT_STATUS,
    "decision_threshold": SELECTED_THRESHOLD,
    "threshold_policy": f"maximize_validation_f1_with_precision_at_least_{MINIMUM_PRECISION}",
    "feature_columns": feature_columns,
    "train_end_date": str(train_end),
    "validation_end_date": str(validation_end)
}

with mlflow.start_run(run_name="lightgbm_champion") as run:
    for key, value in {
        "decision_threshold": SELECTED_THRESHOLD,
        "minimum_precision": MINIMUM_PRECISION,
        "model_type": "LightGBM",
        "feature_table": FEATURE_TABLE
    }.items():
        mlflow.log_param(key, value)
    for prefix, metrics in [("validation", selected_policy), ("test", test_metrics)]:
        for key, value in metrics.items():
            if key != "threshold":
                mlflow.log_metric(f"{prefix}_{key}", value)
    mlflow.log_metric("train_roc_auc", train_roc_auc)
    mlflow.log_metric("validation_roc_auc", validation_roc_auc)
    mlflow.log_metric("roc_auc_gap", roc_auc_gap)
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = Path(temp_dir) / "model_configuration.json"
        config_path.write_text(json.dumps(configuration, indent=2), encoding="utf-8")
        mlflow.log_artifact(str(config_path), artifact_path="configuration")
    model_info = mlflow.sklearn.log_model(
        sk_model=champion_pipeline,
        artifact_path="model",
        signature=signature,
        input_example=X_train.head(5)
    )
    run_id = run.info.run_id

model_version = mlflow.register_model(model_info.model_uri, REGISTERED_MODEL_NAME)
client = MlflowClient()
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "decision_threshold", str(SELECTED_THRESHOLD))
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "threshold_policy", configuration["threshold_policy"])
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "run_id", run_id)
client.set_model_version_tag(REGISTERED_MODEL_NAME, model_version.version, "deployment_status", DEPLOYMENT_STATUS)

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Version: {model_version.version}")
print(f"Deployment status: {DEPLOYMENT_STATUS}")
print(f"Decision threshold: {SELECTED_THRESHOLD:.6f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
