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

# CELL ********************

# STEP 53: Load and validate the latest registered model version

import mlflow
import mlflow.sklearn

from mlflow.tracking import MlflowClient

REGISTERED_MODEL_NAME = "predictive_maintenance_lightgbm"
DECISION_THRESHOLD = 0.50

mlflow_client = MlflowClient()

model_versions = list(
    mlflow_client.search_model_versions(
        f"name = '{REGISTERED_MODEL_NAME}'"
    )
)

assert len(model_versions) > 0, (
    f"No registered versions found for model: "
    f"{REGISTERED_MODEL_NAME}"
)

latest_model_version = max(
    model_versions,
    key=lambda model_version: int(model_version.version)
)

MODEL_VERSION = str(latest_model_version.version)
MODEL_URI = (
    f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
)

registered_model_pipeline = mlflow.sklearn.load_model(
    MODEL_URI
)

assert hasattr(registered_model_pipeline, "predict_proba"), (
    "The loaded model does not support probability predictions"
)

assert hasattr(registered_model_pipeline, "named_steps"), (
    "The loaded model is not a scikit-learn pipeline"
)

assert "preprocessor" in registered_model_pipeline.named_steps, (
    "The preprocessing step is missing from the registered pipeline"
)

assert "classifier" in registered_model_pipeline.named_steps, (
    "The classifier step is missing from the registered pipeline"
)

registered_classifier = (
    registered_model_pipeline.named_steps["classifier"]
)

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"Loaded model version: {MODEL_VERSION}")
print(f"Model URI: {MODEL_URI}")
print(f"Model class: {type(registered_model_pipeline).__name__}")
print(f"Classifier class: {type(registered_classifier).__name__}")
print(f"Decision threshold: {DECISION_THRESHOLD:.2f}")
print("Probability prediction supported: True")
print("Registered model loaded successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 54: Prepare the latest feature record for each machine

from pyspark.sql import functions as F
from pyspark.sql import Window

FEATURE_TABLE_NAME = "gold_machine_features_daily"

feature_dataset_df = spark.table(FEATURE_TABLE_NAME)

latest_record_window = (
    Window
    .partitionBy("machine_id")
    .orderBy(F.col("feature_date").desc())
)

latest_machine_features_df = (
    feature_dataset_df
    .withColumn(
        "latest_record_rank",
        F.row_number().over(latest_record_window)
    )
    .filter(F.col("latest_record_rank") == 1)
    .drop("latest_record_rank")
    .cache()
)

latest_feature_validation_df = (
    latest_machine_features_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.countDistinct("feature_date").alias("feature_date_count"),
        F.min("feature_date").alias("minimum_feature_date"),
        F.max("feature_date").alias("maximum_feature_date"),
        F.sum(
            F.when(
                F.col("machine_id").isNull(),
                1
            ).otherwise(0)
        ).alias("null_machine_ids")
    )
)

validation_row = latest_feature_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["feature_date_count"] == 1, (
    "The latest records do not share one feature date"
)

assert validation_row["null_machine_ids"] == 0, (
    f"Null machine IDs found: {validation_row['null_machine_ids']}"
)

display(latest_feature_validation_df)

print(f"Source feature table: {FEATURE_TABLE_NAME}")
print(f"Latest machine records: {validation_row['row_count']}")
print(
    f"Scoring feature date: "
    f"{validation_row['maximum_feature_date']}"
)
print("Prediction generated: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 55: Generate and validate failure predictions for all machines

import pandas as pd
import numpy as np

from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql import types as T

MODEL_NAME = REGISTERED_MODEL_NAME
MODEL_VERSION_VALUE = MODEL_VERSION
PREDICTION_THRESHOLD = DECISION_THRESHOLD

assert hasattr(
    registered_model_pipeline,
    "feature_names_in_"
), "The registered pipeline does not expose its input feature names"

model_feature_columns = list(
    registered_model_pipeline.feature_names_in_
)

missing_feature_columns = [
    column_name
    for column_name in model_feature_columns
    if column_name not in latest_machine_features_df.columns
]

assert len(missing_feature_columns) == 0, (
    f"Missing model input features: {missing_feature_columns}"
)

scoring_source_pdf = (
    latest_machine_features_df
    .select(
        "machine_id",
        "feature_date",
        *model_feature_columns
    )
    .orderBy("machine_id")
    .toPandas()
)

X_scoring = scoring_source_pdf[model_feature_columns]

failure_probabilities = (
    registered_model_pipeline
    .predict_proba(X_scoring)[:, 1]
)

predicted_failures = (
    failure_probabilities >= PREDICTION_THRESHOLD
).astype(int)

prediction_timestamp = datetime.now(
    timezone.utc
).replace(tzinfo=None)

predictions_pdf = pd.DataFrame(
    {
        "machine_id": scoring_source_pdf["machine_id"],
        "feature_date": scoring_source_pdf["feature_date"],
        "failure_probability": failure_probabilities,
        "predicted_failure_7d": predicted_failures
    }
)

predictions_pdf["risk_level"] = pd.cut(
    predictions_pdf["failure_probability"],
    bins=[
        -np.inf,
        0.20,
        0.40,
        0.70,
        np.inf
    ],
    labels=[
        "LOW",
        "MEDIUM",
        "HIGH",
        "CRITICAL"
    ],
    right=False
).astype(str)

predictions_pdf["model_name"] = MODEL_NAME
predictions_pdf["model_version"] = MODEL_VERSION_VALUE
predictions_pdf["decision_threshold"] = PREDICTION_THRESHOLD
predictions_pdf["prediction_timestamp"] = prediction_timestamp

prediction_schema = T.StructType(
    [
        T.StructField(
            "machine_id",
            T.StringType(),
            False
        ),
        T.StructField(
            "feature_date",
            T.DateType(),
            False
        ),
        T.StructField(
            "failure_probability",
            T.DoubleType(),
            False
        ),
        T.StructField(
            "predicted_failure_7d",
            T.IntegerType(),
            False
        ),
        T.StructField(
            "risk_level",
            T.StringType(),
            False
        ),
        T.StructField(
            "model_name",
            T.StringType(),
            False
        ),
        T.StructField(
            "model_version",
            T.StringType(),
            False
        ),
        T.StructField(
            "decision_threshold",
            T.DoubleType(),
            False
        ),
        T.StructField(
            "prediction_timestamp",
            T.TimestampType(),
            False
        )
    ]
)

failure_predictions_df = spark.createDataFrame(
    predictions_pdf,
    schema=prediction_schema
).cache()

prediction_validation_df = (
    failure_predictions_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("predicted_failure_7d").alias(
            "predicted_failure_count"
        ),
        F.min("failure_probability").alias(
            "minimum_failure_probability"
        ),
        F.max("failure_probability").alias(
            "maximum_failure_probability"
        ),
        F.sum(
            F.when(
                F.col("failure_probability").isNull()
                | F.col("risk_level").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_predictions")
    )
)

validation_row = prediction_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected prediction row count: "
    f"{validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: "
    f"{validation_row['machine_count']}"
)

assert validation_row["rows_with_null_predictions"] == 0, (
    "Null prediction values found: "
    f"{validation_row['rows_with_null_predictions']}"
)

assert (
    0.0
    <= validation_row["minimum_failure_probability"]
    <= 1.0
), "Invalid minimum failure probability"

assert (
    0.0
    <= validation_row["maximum_failure_probability"]
    <= 1.0
), "Invalid maximum failure probability"

display(prediction_validation_df)

display(
    failure_predictions_df
    .orderBy(
        F.col("failure_probability").desc()
    )
    .limit(10)
)

print(f"Scored machines: {validation_row['row_count']}")
print(
    f"Predicted failures: "
    f"{validation_row['predicted_failure_count']}"
)
print(f"Model name: {MODEL_NAME}")
print(f"Model version: {MODEL_VERSION_VALUE}")
print(f"Decision threshold: {PREDICTION_THRESHOLD:.2f}")
print("Predictions generated successfully.")
print("Predictions saved to Delta table: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 56: Save and validate the Gold failure predictions table

PREDICTION_TABLE_NAME = "gold_failure_predictions"

(
    failure_predictions_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTION_TABLE_NAME)
)

saved_predictions_df = spark.table(
    PREDICTION_TABLE_NAME
)

saved_prediction_validation_df = (
    saved_predictions_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("predicted_failure_7d").alias(
            "predicted_failure_count"
        ),
        F.countDistinct("model_version").alias(
            "model_version_count"
        ),
        F.countDistinct("feature_date").alias(
            "feature_date_count"
        ),
        F.sum(
            F.when(
                F.col("failure_probability").isNull()
                | F.col("predicted_failure_7d").isNull()
                | F.col("risk_level").isNull()
                | F.col("prediction_timestamp").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_predictions")
    )
)

validation_row = saved_prediction_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected saved row count: "
    f"{validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected saved machine count: "
    f"{validation_row['machine_count']}"
)

assert validation_row["predicted_failure_count"] == 2, (
    f"Unexpected predicted failure count: "
    f"{validation_row['predicted_failure_count']}"
)

assert validation_row["model_version_count"] == 1, (
    f"Unexpected model version count: "
    f"{validation_row['model_version_count']}"
)

assert validation_row["feature_date_count"] == 1, (
    f"Unexpected feature date count: "
    f"{validation_row['feature_date_count']}"
)

assert validation_row["rows_with_null_predictions"] == 0, (
    "Null saved predictions found: "
    f"{validation_row['rows_with_null_predictions']}"
)

display(saved_prediction_validation_df)

display(
    saved_predictions_df
    .filter(F.col("predicted_failure_7d") == 1)
    .orderBy(F.col("failure_probability").desc())
)

print(f"Saved Delta table: {PREDICTION_TABLE_NAME}")
print(f"Saved rows: {validation_row['row_count']}")
print(
    f"Saved predicted failures: "
    f"{validation_row['predicted_failure_count']}"
)
print(f"Saved model version: {MODEL_VERSION_VALUE}")
print("Batch scoring completed successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
