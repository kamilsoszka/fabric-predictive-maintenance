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

# STEP 58: Create and validate the maintenance risk dashboard table

from pyspark.sql import functions as F
from pyspark.sql import Window

PREDICTION_TABLE_NAME = "gold_failure_predictions"
FEATURE_TABLE_NAME = "gold_machine_features_daily"
MACHINE_TABLE_NAME = "silver_machine_master"

predictions_df = spark.table(PREDICTION_TABLE_NAME)
features_df = spark.table(FEATURE_TABLE_NAME)
machines_df = spark.table(MACHINE_TABLE_NAME)

latest_feature_window = (
    Window
    .partitionBy("machine_id")
    .orderBy(F.col("feature_date").desc())
)

latest_features_df = (
    features_df
    .withColumn(
        "feature_rank",
        F.row_number().over(latest_feature_window)
    )
    .filter(F.col("feature_rank") == 1)
    .drop("feature_rank")
)

maintenance_risk_dashboard_df = (
    predictions_df.alias("predictions")
    .join(
        machines_df.alias("machines"),
        on="machine_id",
        how="inner"
    )
    .join(
        latest_features_df.alias("features"),
        on=["machine_id", "feature_date"],
        how="inner"
    )
    .withColumn(
        "inspection_priority",
        F.when(
            F.col("predicted_failure_7d") == 1,
            F.lit(1)
        )
        .when(
            F.col("risk_level") == "HIGH",
            F.lit(2)
        )
        .when(
            F.col("risk_level") == "MEDIUM",
            F.lit(3)
        )
        .otherwise(F.lit(4))
    )
    .withColumn(
        "recommended_action",
        F.when(
            F.col("predicted_failure_7d") == 1,
            F.lit("INSPECT_WITHIN_24_HOURS")
        )
        .when(
            F.col("risk_level") == "HIGH",
            F.lit("INSPECT_WITHIN_72_HOURS")
        )
        .when(
            F.col("risk_level") == "MEDIUM",
            F.lit("REVIEW_WITHIN_7_DAYS")
        )
        .otherwise(
            F.lit("CONTINUE_STANDARD_MONITORING")
        )
    )
    .select(
        "machine_id",
        "feature_date",
        F.col("machines.machine_type").alias("machine_type"),
        F.col("machines.manufacturer").alias("manufacturer"),
        F.col("machines.location").alias("location"),
        "failure_probability",
        "predicted_failure_7d",
        "risk_level",
        "inspection_priority",
        "recommended_action",
        "temperature_avg",
        "temperature_max",
        "vibration_avg",
        "vibration_max",
        "pressure_avg",
        "pressure_min",
        "power_consumption_avg",
        "power_consumption_max",
        "load_avg",
        "load_max",
        "days_since_last_maintenance",
        "maintenance_count_30d",
        "maintenance_duration_hours_30d",
        "maintenance_cost_30d",
        F.col("predictions.model_name").alias("model_name"),
        F.col("predictions.model_version").alias("model_version"),
        F.col("predictions.decision_threshold").alias(
            "decision_threshold"
        ),
        F.col("predictions.prediction_timestamp").alias(
            "prediction_timestamp"
        )
    )
    .cache()
)

dashboard_validation_df = (
    maintenance_risk_dashboard_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("predicted_failure_7d").alias(
            "predicted_failure_count"
        ),
        F.sum(
            F.when(
                F.col("risk_level") == "HIGH",
                1
            ).otherwise(0)
        ).alias("high_risk_count"),
        F.sum(
            F.when(
                F.col("risk_level") == "MEDIUM",
                1
            ).otherwise(0)
        ).alias("medium_risk_count"),
        F.sum(
            F.when(
                F.col("risk_level") == "LOW",
                1
            ).otherwise(0)
        ).alias("low_risk_count"),
        F.sum(
            F.when(
                F.col("machine_type").isNull()
                | F.col("failure_probability").isNull()
                | F.col("recommended_action").isNull()
                | F.col("temperature_avg").isNull()
                | F.col("days_since_last_maintenance").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_reporting_values")
    )
)

validation_row = dashboard_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected dashboard row count: "
    f"{validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected dashboard machine count: "
    f"{validation_row['machine_count']}"
)

assert validation_row["predicted_failure_count"] == 2, (
    f"Unexpected predicted failure count: "
    f"{validation_row['predicted_failure_count']}"
)

assert validation_row["rows_with_null_reporting_values"] == 0, (
    "Null reporting values found: "
    f"{validation_row['rows_with_null_reporting_values']}"
)

display(dashboard_validation_df)

display(
    maintenance_risk_dashboard_df
    .orderBy(
        F.col("inspection_priority").asc(),
        F.col("failure_probability").desc()
    )
    .limit(10)
)

print("Maintenance risk dashboard dataset created successfully.")
print(f"Dashboard rows: {validation_row['row_count']}")
print(
    f"Predicted failures: "
    f"{validation_row['predicted_failure_count']}"
)
print("Dashboard table saved: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 59: Save and validate the maintenance risk dashboard table

DASHBOARD_TABLE_NAME = "gold_maintenance_risk_dashboard"

(
    maintenance_risk_dashboard_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(DASHBOARD_TABLE_NAME)
)

saved_dashboard_df = spark.table(
    DASHBOARD_TABLE_NAME
)

saved_dashboard_validation_df = (
    saved_dashboard_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("predicted_failure_7d").alias(
            "predicted_failure_count"
        ),
        F.countDistinct("feature_date").alias(
            "feature_date_count"
        ),
        F.countDistinct("model_version").alias(
            "model_version_count"
        ),
        F.sum(
            F.when(
                F.col("machine_id").isNull()
                | F.col("failure_probability").isNull()
                | F.col("risk_level").isNull()
                | F.col("inspection_priority").isNull()
                | F.col("recommended_action").isNull()
                | F.col("prediction_timestamp").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_reporting_values")
    )
)

validation_row = saved_dashboard_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected dashboard row count: "
    f"{validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: "
    f"{validation_row['machine_count']}"
)

assert validation_row["predicted_failure_count"] == 2, (
    f"Unexpected predicted failure count: "
    f"{validation_row['predicted_failure_count']}"
)

assert validation_row["feature_date_count"] == 1, (
    f"Unexpected feature date count: "
    f"{validation_row['feature_date_count']}"
)

assert validation_row["model_version_count"] == 1, (
    f"Unexpected model version count: "
    f"{validation_row['model_version_count']}"
)

assert validation_row["rows_with_null_reporting_values"] == 0, (
    "Null reporting values found: "
    f"{validation_row['rows_with_null_reporting_values']}"
)

display(saved_dashboard_validation_df)

display(
    saved_dashboard_df
    .orderBy(
        F.col("inspection_priority").asc(),
        F.col("failure_probability").desc()
    )
    .limit(10)
)

print(f"Saved Delta table: {DASHBOARD_TABLE_NAME}")
print(f"Saved rows: {validation_row['row_count']}")
print(
    f"Saved predicted failures: "
    f"{validation_row['predicted_failure_count']}"
)
print(f"Saved columns: {len(saved_dashboard_df.columns)}")
print("Reporting layer completed successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
