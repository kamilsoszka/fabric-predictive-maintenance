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

# # 04_batch_scoring
# 
# Loads the approved Champion model and its version-bound threshold, validates feature freshness, and writes idempotent historical predictions.

# CELL ********************

from datetime import datetime, timezone
import mlflow.sklearn
import numpy as np
import pandas as pd
from delta.tables import DeltaTable
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.sql import types as T

REGISTERED_MODEL_NAME = "predictive_maintenance_lightgbm"
DEPLOYMENT_STATUS = "champion"
FEATURE_TABLE = "gold_machine_features_daily"
PREDICTION_HISTORY_TABLE = "gold_failure_predictions_history"
MAX_FEATURE_AGE_DAYS = 7

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

client = MlflowClient()
model_versions = list(client.search_model_versions(f"name = '{REGISTERED_MODEL_NAME}'"))
approved_versions = [
    version for version in model_versions
    if (version.tags or {}).get("deployment_status") == DEPLOYMENT_STATUS
]
assert approved_versions, "No model version is tagged deployment_status=champion"
champion_version = max(approved_versions, key=lambda version: int(version.version))
MODEL_VERSION = str(champion_version.version)
model_tags = champion_version.tags or {}
assert "decision_threshold" in model_tags, "Champion model version has no decision_threshold tag"
DECISION_THRESHOLD = float(model_tags["decision_threshold"])
MODEL_URI = f"models:/{REGISTERED_MODEL_NAME}/{MODEL_VERSION}"
model_pipeline = mlflow.sklearn.load_model(MODEL_URI)
assert hasattr(model_pipeline, "predict_proba")
assert hasattr(model_pipeline, "feature_names_in_")
print(f"Loaded approved {REGISTERED_MODEL_NAME} version {MODEL_VERSION}")
print(f"Deployment status: {DEPLOYMENT_STATUS}")
print(f"Version-bound threshold: {DECISION_THRESHOLD:.6f}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

feature_df = spark.table(FEATURE_TABLE)
latest_date = feature_df.agg(F.max("feature_date").alias("latest_date")).first()["latest_date"]
assert latest_date is not None
age_days = spark.range(1).select(F.datediff(F.current_date(), F.lit(latest_date)).alias("age_days")).first()["age_days"]
assert 0 <= age_days <= MAX_FEATURE_AGE_DAYS, f"Stale feature data: {age_days} days old"

latest_df = feature_df.filter(F.col("feature_date") == F.lit(latest_date))
assert latest_df.count() == latest_df.select("machine_id").distinct().count(), "Duplicate machine rows on latest feature date"
feature_columns = list(model_pipeline.feature_names_in_)
missing = [c for c in feature_columns if c not in latest_df.columns]
assert not missing, f"Missing features: {missing}"

source_pdf = latest_df.select("machine_id", "feature_date", *feature_columns).orderBy("machine_id").toPandas()
probabilities = model_pipeline.predict_proba(source_pdf[feature_columns])[:, 1]
alerts = (probabilities >= DECISION_THRESHOLD).astype(int)
scored_at = datetime.now(timezone.utc).replace(tzinfo=None)

pdf = pd.DataFrame({
    "machine_id": source_pdf["machine_id"],
    "feature_date": source_pdf["feature_date"],
    "failure_probability": probabilities,
    "model_alert": alerts
})
pdf["risk_level"] = pd.cut(pdf["failure_probability"], bins=[-np.inf, .20, .40, .70, np.inf], labels=["LOW", "MEDIUM", "HIGH", "CRITICAL"], right=False).astype(str)
pdf["model_name"] = REGISTERED_MODEL_NAME
pdf["model_version"] = MODEL_VERSION
pdf["deployment_status"] = DEPLOYMENT_STATUS
pdf["decision_threshold"] = DECISION_THRESHOLD
pdf["prediction_timestamp_utc"] = scored_at

schema = T.StructType([
    T.StructField("machine_id", T.StringType(), False),
    T.StructField("feature_date", T.DateType(), False),
    T.StructField("failure_probability", T.DoubleType(), False),
    T.StructField("model_alert", T.IntegerType(), False),
    T.StructField("risk_level", T.StringType(), False),
    T.StructField("model_name", T.StringType(), False),
    T.StructField("model_version", T.StringType(), False),
    T.StructField("deployment_status", T.StringType(), False),
    T.StructField("decision_threshold", T.DoubleType(), False),
    T.StructField("prediction_timestamp_utc", T.TimestampType(), False)
])
predictions_df = spark.createDataFrame(pdf, schema=schema)
assert predictions_df.filter(F.col("failure_probability").isNull()).count() == 0
display(predictions_df.orderBy(F.desc("failure_probability")).limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# Idempotent history key: machine, feature date, model version, threshold.
if not spark.catalog.tableExists(PREDICTION_HISTORY_TABLE):
    predictions_df.write.format("delta").mode("overwrite").saveAsTable(PREDICTION_HISTORY_TABLE)
else:
    target = DeltaTable.forName(spark, PREDICTION_HISTORY_TABLE)
    condition = """
        target.machine_id = source.machine_id AND
        target.feature_date = source.feature_date AND
        target.model_name = source.model_name AND
        target.model_version = source.model_version AND
        target.decision_threshold = source.decision_threshold
    """
    (target.alias("target").merge(predictions_df.alias("source"), condition)
     .whenMatchedUpdateAll().whenNotMatchedInsertAll().execute())

saved = spark.table(PREDICTION_HISTORY_TABLE)
current = saved.filter(
    (F.col("feature_date") == F.lit(latest_date)) &
    (F.col("model_name") == REGISTERED_MODEL_NAME) &
    (F.col("model_version") == MODEL_VERSION) &
    (F.col("decision_threshold") == DECISION_THRESHOLD)
)
assert current.count() == predictions_df.count()
print(f"History table: {PREDICTION_HISTORY_TABLE}")
print(f"Current scored machines: {current.count()}")
print(f"Current model alerts: {current.agg(F.sum('model_alert')).first()[0]}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
