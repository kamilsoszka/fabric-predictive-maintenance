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

# STEP 32: Load and inspect the Silver tables

from pyspark.sql import functions as F
from pyspark.sql import Window

silver_machine_master_df = spark.table("silver_machine_master")
silver_failure_events_df = spark.table("silver_failure_events")
silver_maintenance_events_df = spark.table("silver_maintenance_events")
silver_machine_telemetry_df = spark.table("silver_machine_telemetry")

silver_tables = {
    "silver_machine_master": silver_machine_master_df,
    "silver_failure_events": silver_failure_events_df,
    "silver_maintenance_events": silver_maintenance_events_df,
    "silver_machine_telemetry": silver_machine_telemetry_df
}

inspection_results = []

for table_name, table_df in silver_tables.items():
    inspection_results.append(
        (
            table_name,
            table_df.count(),
            len(table_df.columns)
        )
    )

silver_inspection_df = spark.createDataFrame(
    inspection_results,
    [
        "table_name",
        "row_count",
        "column_count"
    ]
)

display(silver_inspection_df.orderBy("table_name"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 33: Create and validate complete daily telemetry aggregates

daily_telemetry_features_df = (
    silver_machine_telemetry_df
    .withColumn(
        "feature_date",
        F.to_date(F.col("telemetry_timestamp"))
    )
    .groupBy(
        "machine_id",
        "feature_date"
    )
    .agg(
        F.count("*").alias("telemetry_sample_count"),

        F.avg("temperature_c").alias("temperature_avg"),
        F.min("temperature_c").alias("temperature_min"),
        F.max("temperature_c").alias("temperature_max"),
        F.stddev("temperature_c").alias("temperature_std"),

        F.avg("vibration_mm_s").alias("vibration_avg"),
        F.max("vibration_mm_s").alias("vibration_max"),
        F.stddev("vibration_mm_s").alias("vibration_std"),

        F.avg("pressure_bar").alias("pressure_avg"),
        F.min("pressure_bar").alias("pressure_min"),
        F.stddev("pressure_bar").alias("pressure_std"),

        F.avg("rotation_speed_rpm").alias("rotation_speed_avg"),
        F.stddev("rotation_speed_rpm").alias("rotation_speed_std"),

        F.avg("power_consumption_kw").alias("power_consumption_avg"),
        F.max("power_consumption_kw").alias("power_consumption_max"),

        F.avg("load_percent").alias("load_avg"),
        F.max("load_percent").alias("load_max")
    )
    .filter(
        F.col("telemetry_sample_count") == 96
    )
    .select(
        "machine_id",
        "feature_date",
        "telemetry_sample_count",
        *[
            F.round(F.col(column_name), 4).alias(column_name)
            for column_name in [
                "temperature_avg",
                "temperature_min",
                "temperature_max",
                "temperature_std",
                "vibration_avg",
                "vibration_max",
                "vibration_std",
                "pressure_avg",
                "pressure_min",
                "pressure_std",
                "rotation_speed_avg",
                "rotation_speed_std",
                "power_consumption_avg",
                "power_consumption_max",
                "load_avg",
                "load_max"
            ]
        ]
    )
    .cache()
)

daily_feature_validation_df = daily_telemetry_features_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("machine_id").alias("machine_count"),
    F.countDistinct("feature_date").alias("complete_date_count"),
    F.min("feature_date").alias("minimum_feature_date"),
    F.max("feature_date").alias("maximum_feature_date"),
    F.min("telemetry_sample_count").alias("minimum_sample_count"),
    F.max("telemetry_sample_count").alias("maximum_sample_count")
)

validation_row = daily_feature_validation_df.first()

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["minimum_sample_count"] == 96, (
    f"Unexpected minimum sample count: "
    f"{validation_row['minimum_sample_count']}"
)

assert validation_row["maximum_sample_count"] == 96, (
    f"Unexpected maximum sample count: "
    f"{validation_row['maximum_sample_count']}"
)

assert validation_row["row_count"] == (
    validation_row["machine_count"]
    * validation_row["complete_date_count"]
), (
    f"Incomplete machine-date combinations found: "
    f"{validation_row['row_count']}"
)

display(daily_feature_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 34: Add historical maintenance features

maintenance_history_df = (
    daily_telemetry_features_df.alias("daily")
    .join(
        silver_maintenance_events_df.alias("maintenance"),
        (
            F.col("daily.machine_id")
            == F.col("maintenance.machine_id")
        )
        & (
            F.col("maintenance.maintenance_date")
            <= F.col("daily.feature_date")
        ),
        "left"
    )
    .groupBy(
        F.col("daily.machine_id").alias("machine_id"),
        F.col("daily.feature_date").alias("feature_date")
    )
    .agg(
        F.max(
            F.col("maintenance.maintenance_date")
        ).alias("last_maintenance_date"),

        F.countDistinct(
            F.when(
                F.col("maintenance.maintenance_date")
                >= F.date_sub(F.col("daily.feature_date"), 29),
                F.col("maintenance.maintenance_id")
            )
        ).alias("maintenance_count_30d"),

        F.sum(
            F.when(
                F.col("maintenance.maintenance_date")
                >= F.date_sub(F.col("daily.feature_date"), 29),
                F.col("maintenance.duration_hours")
            ).otherwise(F.lit(0.0))
        ).alias("maintenance_duration_hours_30d"),

        F.sum(
            F.when(
                F.col("maintenance.maintenance_date")
                >= F.date_sub(F.col("daily.feature_date"), 29),
                F.col("maintenance.maintenance_cost")
            ).otherwise(F.lit(0.0))
        ).alias("maintenance_cost_30d")
    )
    .withColumn(
        "days_since_last_maintenance",
        F.when(
            F.col("last_maintenance_date").isNotNull(),
            F.datediff(
                F.col("feature_date"),
                F.col("last_maintenance_date")
            )
        ).otherwise(F.lit(9999))
    )
)

daily_features_with_maintenance_df = (
    daily_telemetry_features_df.alias("daily")
    .join(
        maintenance_history_df.alias("maintenance"),
        on=["machine_id", "feature_date"],
        how="left"
    )
    .fillna(
        {
            "maintenance_count_30d": 0,
            "maintenance_duration_hours_30d": 0.0,
            "maintenance_cost_30d": 0.0,
            "days_since_last_maintenance": 9999
        }
    )
    .drop("last_maintenance_date")
    .cache()
)

maintenance_feature_validation_df = (
    daily_features_with_maintenance_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum(
            F.when(
                F.col("maintenance_count_30d").isNull()
                | F.col("maintenance_duration_hours_30d").isNull()
                | F.col("maintenance_cost_30d").isNull()
                | F.col("days_since_last_maintenance").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_maintenance_features"),
        F.sum(
            F.when(
                F.col("days_since_last_maintenance") < 0,
                1
            ).otherwise(0)
        ).alias("rows_with_future_maintenance"),
        F.max("maintenance_count_30d").alias(
            "maximum_maintenance_count_30d"
        ),
        F.min("days_since_last_maintenance").alias(
            "minimum_days_since_maintenance"
        )
    )
)

validation_row = maintenance_feature_validation_df.first()

assert validation_row["row_count"] == 17900, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["rows_with_null_maintenance_features"] == 0, (
    "Null maintenance features found: "
    f"{validation_row['rows_with_null_maintenance_features']}"
)

assert validation_row["rows_with_future_maintenance"] == 0, (
    "Future maintenance leakage found: "
    f"{validation_row['rows_with_future_maintenance']}"
)

display(maintenance_feature_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 35: Create and validate the seven-day failure target

future_failure_targets_df = (
    daily_features_with_maintenance_df.alias("features")
    .join(
        silver_failure_events_df.alias("failures"),
        (
            F.col("features.machine_id")
            == F.col("failures.machine_id")
        )
        & (
            F.col("failures.failure_date")
            > F.col("features.feature_date")
        )
        & (
            F.col("failures.failure_date")
            <= F.date_add(
                F.col("features.feature_date"),
                7
            )
        ),
        "left"
    )
    .groupBy(
        F.col("features.machine_id").alias("machine_id"),
        F.col("features.feature_date").alias("feature_date")
    )
    .agg(
        F.when(
            F.count(F.col("failures.failure_id")) > 0,
            F.lit(1)
        ).otherwise(F.lit(0)).alias("failure_within_7d"),
        F.min(
            F.col("failures.failure_date")
        ).alias("next_failure_date")
    )
)

ml_feature_dataset_df = (
    daily_features_with_maintenance_df.alias("features")
    .join(
        future_failure_targets_df.alias("targets"),
        on=["machine_id", "feature_date"],
        how="inner"
    )
    .cache()
)

target_validation_df = (
    ml_feature_dataset_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("failure_within_7d").alias("positive_rows"),
        F.sum(
            F.when(
                F.col("failure_within_7d") == 0,
                1
            ).otherwise(0)
        ).alias("negative_rows"),
        F.round(
            F.avg(
                F.col("failure_within_7d").cast("double")
            ) * 100,
            2
        ).alias("positive_rate_percent"),
        F.sum(
            F.when(
                F.col("next_failure_date")
                <= F.col("feature_date"),
                1
            ).otherwise(0)
        ).alias("target_leakage_rows")
    )
)

validation_row = target_validation_df.first()

assert validation_row["row_count"] == 17900, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["positive_rows"] > 0, (
    "No positive target rows were generated"
)

assert validation_row["negative_rows"] > 0, (
    "No negative target rows were generated"
)

assert validation_row["target_leakage_rows"] == 0, (
    "Target leakage rows found: "
    f"{validation_row['target_leakage_rows']}"
)

display(target_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 36: Add machine attributes and create the final ML feature dataset

final_ml_feature_dataset_df = (
    ml_feature_dataset_df.alias("features")
    .join(
        silver_machine_master_df.alias("machines"),
        on="machine_id",
        how="inner"
    )
    .withColumn(
        "machine_age_days",
        F.datediff(
            F.col("feature_date"),
            F.col("installation_date")
        )
    )
    .select(
        "machine_id",
        "feature_date",
        "machine_type",
        "manufacturer",
        "location",
        "is_active",
        "machine_age_days",
        *[
            column_name
            for column_name in ml_feature_dataset_df.columns
            if column_name not in {
                "machine_id",
                "feature_date",
                "next_failure_date"
            }
        ]
    )
    .cache()
)

final_feature_validation_df = (
    final_ml_feature_dataset_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("machine_id").alias("machine_count"),
        F.sum("failure_within_7d").alias("positive_rows"),
        F.sum(
            F.when(
                F.col("machine_type").isNull()
                | F.col("manufacturer").isNull()
                | F.col("location").isNull()
                | F.col("machine_age_days").isNull(),
                1
            ).otherwise(0)
        ).alias("rows_with_null_machine_features"),
        F.sum(
            F.when(
                F.col("machine_age_days") < 0,
                1
            ).otherwise(0)
        ).alias("rows_with_negative_machine_age"),
        F.min("machine_age_days").alias("minimum_machine_age_days"),
        F.max("machine_age_days").alias("maximum_machine_age_days")
    )
)

for forbidden_column in [
    "next_failure_date",
    "failure_date",
    "failure_id",
    "pre_failure_signal",
    "degradation_strength"
]:
    assert forbidden_column not in final_ml_feature_dataset_df.columns, (
        f"Potential leakage column found: {forbidden_column}"
    )

validation_row = final_feature_validation_df.first()

assert validation_row["row_count"] == 17900, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["positive_rows"] == 1044, (
    f"Unexpected positive row count: {validation_row['positive_rows']}"
)

assert validation_row["rows_with_null_machine_features"] == 0, (
    "Null machine features found: "
    f"{validation_row['rows_with_null_machine_features']}"
)

assert validation_row["rows_with_negative_machine_age"] == 0, (
    "Negative machine age rows found: "
    f"{validation_row['rows_with_negative_machine_age']}"
)

display(final_feature_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 37: Save the final ML feature dataset as a Gold Delta table

TABLE_NAME = "gold_machine_features_daily"

(
    final_ml_feature_dataset_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_feature_dataset_df = spark.table(TABLE_NAME)

saved_row_count = saved_feature_dataset_df.count()
saved_column_count = len(saved_feature_dataset_df.columns)
saved_positive_count = (
    saved_feature_dataset_df
    .filter(F.col("failure_within_7d") == 1)
    .count()
)

for forbidden_column in [
    "next_failure_date",
    "failure_date",
    "failure_id",
    "pre_failure_signal",
    "degradation_strength"
]:
    assert forbidden_column not in saved_feature_dataset_df.columns, (
        f"Potential leakage column found after save: {forbidden_column}"
    )

assert saved_row_count == 17900, (
    f"Unexpected row count: {saved_row_count}"
)

assert saved_positive_count == 1044, (
    f"Unexpected positive row count: {saved_positive_count}"
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_row_count:,}")
print(f"Saved columns: {saved_column_count}")
print(f"Saved positive rows: {saved_positive_count:,}")
print("Known leakage columns excluded: True")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
