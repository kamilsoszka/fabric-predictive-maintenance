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

# # 05_reporting_layer
# 
# Builds an explicit current reporting snapshot from prediction history and keeps risk bands separate from binary model alerts.

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql import Window

PREDICTION_HISTORY_TABLE = "gold_failure_predictions_history"
FEATURE_TABLE = "gold_machine_features_daily"
MACHINE_TABLE = "silver_machine_master"
CURRENT_TABLE = "gold_maintenance_risk_current"

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

history_df = spark.table(PREDICTION_HISTORY_TABLE)
features_df = spark.table(FEATURE_TABLE)
machines_df = spark.table(MACHINE_TABLE)

latest_prediction_window = Window.partitionBy("machine_id").orderBy(F.col("prediction_timestamp_utc").desc())
current_predictions_df = (
    history_df.withColumn("prediction_rank", F.row_number().over(latest_prediction_window))
    .filter(F.col("prediction_rank") == 1).drop("prediction_rank")
)

current_features_df = features_df.join(
    current_predictions_df.select("machine_id", "feature_date"),
    on=["machine_id", "feature_date"], how="inner"
)

report_df = (
    current_predictions_df.alias("p")
    .join(machines_df.alias("m"), "machine_id", "inner")
    .join(current_features_df.alias("f"), ["machine_id", "feature_date"], "inner")
    .withColumn(
        "inspection_priority",
        F.when(F.col("model_alert") == 1, 1)
         .when(F.col("risk_level") == "CRITICAL", 1)
         .when(F.col("risk_level") == "HIGH", 2)
         .when(F.col("risk_level") == "MEDIUM", 3)
         .otherwise(4)
    )
    .withColumn(
        "recommended_action",
        F.when(F.col("model_alert") == 1, "MODEL_ALERT_INSPECT_WITHIN_24_HOURS")
         .when(F.col("risk_level") == "CRITICAL", "CRITICAL_RISK_INSPECT_WITHIN_24_HOURS")
         .when(F.col("risk_level") == "HIGH", "HIGH_RISK_INSPECT_WITHIN_72_HOURS")
         .when(F.col("risk_level") == "MEDIUM", "MEDIUM_RISK_REVIEW_WITHIN_7_DAYS")
         .otherwise("CONTINUE_STANDARD_MONITORING")
    )
    .select(
        "machine_id", "feature_date",
        F.col("m.machine_type").alias("machine_type"),
        F.col("m.manufacturer").alias("manufacturer"),
        F.col("m.location").alias("location"),
        "failure_probability", "model_alert", "risk_level",
        "inspection_priority", "recommended_action",
        "temperature_avg", "temperature_max", "vibration_avg", "vibration_max",
        "pressure_avg", "pressure_min", "power_consumption_avg", "power_consumption_max",
        "load_avg", "load_max", "days_since_last_maintenance", "maintenance_count_30d",
        "maintenance_duration_hours_30d", "maintenance_cost_30d",
        "model_name", "model_version", "deployment_status", "decision_threshold", "prediction_timestamp_utc"
    )
)

summary = report_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("machine_id").alias("machine_count"),
    F.sum("model_alert").alias("model_alert_count"),
    *[F.sum(F.when(F.col("risk_level") == level, 1).otherwise(0)).alias(f"{level.lower()}_risk_count") for level in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]]
)
row = summary.first()
assert row["row_count"] == row["machine_count"]
assert row["model_alert_count"] <= row["machine_count"]
assert sum(row[f"{level}_risk_count"] for level in ["low", "medium", "high", "critical"]) == row["row_count"]
display(summary)
display(report_df.orderBy("inspection_priority", F.desc("failure_probability")).limit(20))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

(report_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(CURRENT_TABLE))

saved = spark.table(CURRENT_TABLE)
assert saved.count() == report_df.count()
print(f"Saved current snapshot: {CURRENT_TABLE}")
print("Prediction history remains available in gold_failure_predictions_history")
print("Risk level and binary model alert are intentionally separate fields")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
