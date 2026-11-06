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

# STEP 21: Load and inspect Bronze tables

from pyspark.sql import functions as F
from pyspark.sql import types as T

bronze_machine_master_df = spark.table("bronze_machine_master")
bronze_failure_events_df = spark.table("bronze_failure_events")
bronze_maintenance_events_df = spark.table("bronze_maintenance_events")
bronze_machine_telemetry_df = spark.table("bronze_machine_telemetry")

bronze_tables = {
    "bronze_machine_master": bronze_machine_master_df,
    "bronze_failure_events": bronze_failure_events_df,
    "bronze_maintenance_events": bronze_maintenance_events_df,
    "bronze_machine_telemetry": bronze_machine_telemetry_df
}

inspection_results = []

for table_name, table_df in bronze_tables.items():
    inspection_results.append(
        (
            table_name,
            table_df.count(),
            len(table_df.columns),
            ", ".join(table_df.columns)
        )
    )

bronze_inspection_df = spark.createDataFrame(
    inspection_results,
    [
        "table_name",
        "row_count",
        "column_count",
        "column_names"
    ]
)

display(bronze_inspection_df.orderBy("table_name"))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 22: Create and validate the Silver machine master table

silver_machine_master_df = (
    bronze_machine_master_df
    .select(
        F.upper(F.trim(F.col("machine_id"))).alias("machine_id"),
        F.upper(F.trim(F.col("machine_type"))).alias("machine_type"),
        F.trim(F.col("manufacturer")).alias("manufacturer"),
        F.to_date(F.col("installation_date")).alias("installation_date"),
        F.trim(F.col("location")).alias("location"),
        F.col("is_active").cast("boolean").alias("is_active")
    )
    .dropDuplicates(["machine_id"])
)

machine_master_validation_df = silver_machine_master_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("machine_id").alias("unique_machine_ids"),
    F.sum(
        F.when(F.col("machine_id").isNull(), 1).otherwise(0)
    ).alias("null_machine_ids"),
    F.sum(
        F.when(F.col("installation_date").isNull(), 1).otherwise(0)
    ).alias("null_installation_dates"),
    F.sum(
        F.when(
            F.col("installation_date") > F.current_date(),
            1
        ).otherwise(0)
    ).alias("future_installation_dates")
)

validation_row = machine_master_validation_df.first()

assert validation_row["row_count"] == 100, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["unique_machine_ids"] == 100, (
    f"Unexpected unique machine count: "
    f"{validation_row['unique_machine_ids']}"
)

assert validation_row["null_machine_ids"] == 0, (
    f"Null machine IDs found: {validation_row['null_machine_ids']}"
)

assert validation_row["null_installation_dates"] == 0, (
    "Null installation dates found: "
    f"{validation_row['null_installation_dates']}"
)

assert validation_row["future_installation_dates"] == 0, (
    "Future installation dates found: "
    f"{validation_row['future_installation_dates']}"
)

display(machine_master_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 23: Save the Silver machine master Delta table

TABLE_NAME = "silver_machine_master"

(
    silver_machine_master_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_machine_master_df = spark.table(TABLE_NAME)

saved_row_count = saved_machine_master_df.count()
saved_column_count = len(saved_machine_master_df.columns)

assert saved_row_count == 100, (
    f"Unexpected row count: {saved_row_count}"
)

assert saved_column_count == 6, (
    f"Unexpected column count: {saved_column_count}"
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_row_count}")
print(f"Saved columns: {saved_column_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 24: Create and validate the Silver failure events table

silver_failure_events_df = (
    bronze_failure_events_df
    .select(
        F.upper(F.trim(F.col("failure_id"))).alias("failure_id"),
        F.upper(F.trim(F.col("machine_id"))).alias("machine_id"),
        F.to_date(F.col("failure_date")).alias("failure_date"),
        F.upper(F.trim(F.col("failure_type"))).alias("failure_type"),
        F.upper(F.trim(F.col("severity"))).alias("severity"),
        F.col("downtime_hours").cast("double").alias("downtime_hours")
    )
    .dropDuplicates(["failure_id"])
)

known_machine_ids_df = (
    spark.table("silver_machine_master")
    .select("machine_id")
)

orphan_failure_count = (
    silver_failure_events_df.alias("failures")
    .join(
        known_machine_ids_df.alias("machines"),
        on="machine_id",
        how="left_anti"
    )
    .count()
)

failure_validation_df = silver_failure_events_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("failure_id").alias("unique_failure_ids"),
    F.sum(
        F.when(F.col("failure_id").isNull(), 1).otherwise(0)
    ).alias("null_failure_ids"),
    F.sum(
        F.when(F.col("failure_date").isNull(), 1).otherwise(0)
    ).alias("null_failure_dates"),
    F.sum(
        F.when(F.col("failure_date") > F.current_date(), 1).otherwise(0)
    ).alias("future_failure_dates"),
    F.sum(
        F.when(F.col("downtime_hours") <= 0, 1).otherwise(0)
    ).alias("invalid_downtime_rows")
)

failure_validation_df = (
    failure_validation_df
    .withColumn(
        "orphan_machine_ids",
        F.lit(orphan_failure_count)
    )
)

validation_row = failure_validation_df.first()

assert validation_row["row_count"] == 150, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["unique_failure_ids"] == 150, (
    f"Unexpected unique failure count: "
    f"{validation_row['unique_failure_ids']}"
)

assert validation_row["null_failure_ids"] == 0, (
    f"Null failure IDs found: {validation_row['null_failure_ids']}"
)

assert validation_row["null_failure_dates"] == 0, (
    f"Null failure dates found: {validation_row['null_failure_dates']}"
)

assert validation_row["future_failure_dates"] == 0, (
    f"Future failure dates found: "
    f"{validation_row['future_failure_dates']}"
)

assert validation_row["invalid_downtime_rows"] == 0, (
    f"Invalid downtime rows found: "
    f"{validation_row['invalid_downtime_rows']}"
)

assert validation_row["orphan_machine_ids"] == 0, (
    f"Orphan machine IDs found: "
    f"{validation_row['orphan_machine_ids']}"
)

display(failure_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 25: Save the Silver failure events Delta table

TABLE_NAME = "silver_failure_events"

(
    silver_failure_events_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_failure_events_df = spark.table(TABLE_NAME)

saved_row_count = saved_failure_events_df.count()
saved_column_count = len(saved_failure_events_df.columns)

assert saved_row_count == 150, (
    f"Unexpected row count: {saved_row_count}"
)

assert saved_column_count == 6, (
    f"Unexpected column count: {saved_column_count}"
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_row_count}")
print(f"Saved columns: {saved_column_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 26: Create and validate the Silver maintenance events table

silver_maintenance_events_df = (
    bronze_maintenance_events_df
    .select(
        F.upper(F.trim(F.col("maintenance_id"))).alias("maintenance_id"),
        F.upper(F.trim(F.col("machine_id"))).alias("machine_id"),
        F.to_date(F.col("maintenance_date")).alias("maintenance_date"),
        F.upper(F.trim(F.col("maintenance_type"))).alias("maintenance_type"),
        F.upper(F.trim(F.col("component"))).alias("component"),
        F.col("duration_hours").cast("double").alias("duration_hours"),
        F.col("maintenance_cost").cast("double").alias("maintenance_cost")
    )
    .dropDuplicates(["maintenance_id"])
)

known_machine_ids_df = (
    spark.table("silver_machine_master")
    .select("machine_id")
)

orphan_maintenance_count = (
    silver_maintenance_events_df.alias("maintenance")
    .join(
        known_machine_ids_df.alias("machines"),
        on="machine_id",
        how="left_anti"
    )
    .count()
)

maintenance_validation_df = (
    silver_maintenance_events_df
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("maintenance_id").alias(
            "unique_maintenance_ids"
        ),
        F.sum(
            F.when(
                F.col("maintenance_id").isNull(),
                1
            ).otherwise(0)
        ).alias("null_maintenance_ids"),
        F.sum(
            F.when(
                F.col("maintenance_date").isNull(),
                1
            ).otherwise(0)
        ).alias("null_maintenance_dates"),
        F.sum(
            F.when(
                F.col("maintenance_date") > F.current_date(),
                1
            ).otherwise(0)
        ).alias("future_maintenance_dates"),
        F.sum(
            F.when(
                F.col("duration_hours") <= 0,
                1
            ).otherwise(0)
        ).alias("invalid_duration_rows"),
        F.sum(
            F.when(
                F.col("maintenance_cost") < 0,
                1
            ).otherwise(0)
        ).alias("invalid_cost_rows")
    )
    .withColumn(
        "orphan_machine_ids",
        F.lit(orphan_maintenance_count)
    )
)

validation_row = maintenance_validation_df.first()

assert validation_row["row_count"] == 300, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["unique_maintenance_ids"] == 300, (
    "Unexpected unique maintenance count: "
    f"{validation_row['unique_maintenance_ids']}"
)

assert validation_row["null_maintenance_ids"] == 0, (
    "Null maintenance IDs found: "
    f"{validation_row['null_maintenance_ids']}"
)

assert validation_row["null_maintenance_dates"] == 0, (
    "Null maintenance dates found: "
    f"{validation_row['null_maintenance_dates']}"
)

assert validation_row["future_maintenance_dates"] == 0, (
    "Future maintenance dates found: "
    f"{validation_row['future_maintenance_dates']}"
)

assert validation_row["invalid_duration_rows"] == 0, (
    "Invalid duration rows found: "
    f"{validation_row['invalid_duration_rows']}"
)

assert validation_row["invalid_cost_rows"] == 0, (
    "Invalid cost rows found: "
    f"{validation_row['invalid_cost_rows']}"
)

assert validation_row["orphan_machine_ids"] == 0, (
    "Orphan machine IDs found: "
    f"{validation_row['orphan_machine_ids']}"
)

display(maintenance_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 27: Save the Silver maintenance events Delta table

TABLE_NAME = "silver_maintenance_events"

(
    silver_maintenance_events_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_maintenance_events_df = spark.table(TABLE_NAME)

saved_row_count = saved_maintenance_events_df.count()
saved_column_count = len(saved_maintenance_events_df.columns)

assert saved_row_count == 300, (
    f"Unexpected row count: {saved_row_count}"
)

assert saved_column_count == 7, (
    f"Unexpected column count: {saved_column_count}"
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_row_count}")
print(f"Saved columns: {saved_column_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 28: Create and validate the Silver machine telemetry table

silver_machine_telemetry_df = (
    bronze_machine_telemetry_df
    .select(
        F.upper(F.trim(F.col("telemetry_id"))).alias("telemetry_id"),
        F.upper(F.trim(F.col("machine_id"))).alias("machine_id"),
        F.col("telemetry_timestamp").cast("timestamp").alias(
            "telemetry_timestamp"
        ),
        F.col("temperature_c").cast("double").alias("temperature_c"),
        F.col("vibration_mm_s").cast("double").alias("vibration_mm_s"),
        F.col("pressure_bar").cast("double").alias("pressure_bar"),
        F.col("rotation_speed_rpm").cast("double").alias(
            "rotation_speed_rpm"
        ),
        F.col("power_consumption_kw").cast("double").alias(
            "power_consumption_kw"
        ),
        F.col("load_percent").cast("double").alias("load_percent")
    )
    .dropDuplicates(["telemetry_id"])
)

telemetry_validation_df = silver_machine_telemetry_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("telemetry_id").alias("unique_telemetry_ids"),
    F.countDistinct("machine_id").alias("machine_count"),
    F.sum(
        F.when(F.col("telemetry_id").isNull(), 1).otherwise(0)
    ).alias("null_telemetry_ids"),
    F.sum(
        F.when(F.col("telemetry_timestamp").isNull(), 1).otherwise(0)
    ).alias("null_timestamps"),
    F.sum(
        F.when(
            F.col("temperature_c").isNull()
            | F.col("vibration_mm_s").isNull()
            | F.col("pressure_bar").isNull()
            | F.col("rotation_speed_rpm").isNull()
            | F.col("power_consumption_kw").isNull()
            | F.col("load_percent").isNull(),
            1
        ).otherwise(0)
    ).alias("rows_with_null_sensor_values"),
    F.sum(
        F.when(
            (F.col("temperature_c") < -20)
            | (F.col("temperature_c") > 150)
            | (F.col("vibration_mm_s") < 0)
            | (F.col("vibration_mm_s") > 20)
            | (F.col("pressure_bar") <= 0)
            | (F.col("pressure_bar") > 20)
            | (F.col("rotation_speed_rpm") <= 0)
            | (F.col("rotation_speed_rpm") > 5000)
            | (F.col("power_consumption_kw") < 0)
            | (F.col("power_consumption_kw") > 200)
            | (F.col("load_percent") < 0)
            | (F.col("load_percent") > 100),
            1
        ).otherwise(0)
    ).alias("invalid_sensor_rows")
)

validation_row = telemetry_validation_df.first()

assert validation_row["row_count"] == 1728000, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["unique_telemetry_ids"] == 1728000, (
    "Unexpected unique telemetry count: "
    f"{validation_row['unique_telemetry_ids']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

assert validation_row["null_telemetry_ids"] == 0, (
    f"Null telemetry IDs found: {validation_row['null_telemetry_ids']}"
)

assert validation_row["null_timestamps"] == 0, (
    f"Null timestamps found: {validation_row['null_timestamps']}"
)

assert validation_row["rows_with_null_sensor_values"] == 0, (
    "Rows with null sensor values found: "
    f"{validation_row['rows_with_null_sensor_values']}"
)

assert validation_row["invalid_sensor_rows"] == 0, (
    f"Invalid sensor rows found: "
    f"{validation_row['invalid_sensor_rows']}"
)

display(telemetry_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 29: Save the Silver machine telemetry Delta table

TABLE_NAME = "silver_machine_telemetry"

(
    silver_machine_telemetry_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_telemetry_df = spark.table(TABLE_NAME)

saved_row_count = saved_telemetry_df.count()
saved_column_count = len(saved_telemetry_df.columns)

assert saved_row_count == 1728000, (
    f"Unexpected row count: {saved_row_count}"
)

assert saved_column_count == 9, (
    f"Unexpected column count: {saved_column_count}"
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_row_count:,}")
print(f"Saved columns: {saved_column_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 30: Validate all Silver Delta tables

expected_tables = {
    "silver_machine_master": (100, 6),
    "silver_failure_events": (150, 6),
    "silver_maintenance_events": (300, 7),
    "silver_machine_telemetry": (1728000, 9)
}

validation_results = []

for table_name, expected_values in expected_tables.items():
    expected_row_count, expected_column_count = expected_values

    table_df = spark.table(table_name)
    actual_row_count = table_df.count()
    actual_column_count = len(table_df.columns)

    row_count_status = (
        "PASS"
        if actual_row_count == expected_row_count
        else "FAIL"
    )

    column_count_status = (
        "PASS"
        if actual_column_count == expected_column_count
        else "FAIL"
    )

    validation_status = (
        "PASS"
        if row_count_status == "PASS"
        and column_count_status == "PASS"
        else "FAIL"
    )

    validation_results.append(
        (
            table_name,
            expected_row_count,
            actual_row_count,
            expected_column_count,
            actual_column_count,
            validation_status
        )
    )

silver_validation_df = spark.createDataFrame(
    validation_results,
    [
        "table_name",
        "expected_row_count",
        "actual_row_count",
        "expected_column_count",
        "actual_column_count",
        "validation_status"
    ]
)

failed_table_count = (
    silver_validation_df
    .filter(F.col("validation_status") == "FAIL")
    .count()
)

assert failed_table_count == 0, (
    f"Silver validation failed for {failed_table_count} table(s)"
)

display(
    silver_validation_df.orderBy("table_name")
)

print("Silver layer validation completed successfully.")
print(f"Validated tables: {len(expected_tables)}")
print(f"Failed tables: {failed_table_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
