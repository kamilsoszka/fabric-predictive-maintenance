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

from datetime import datetime

PROJECT_NAME = "fabric_predictive_maintenance_ml"
RANDOM_SEED = 42
MACHINE_COUNT = 100
HISTORY_DAYS = 180
TELEMETRY_INTERVAL_MINUTES = 15

print(f"Project: {PROJECT_NAME}")
print(f"Machines: {MACHINE_COUNT}")
print(f"History: {HISTORY_DAYS} days")
print(f"Telemetry interval: {TELEMETRY_INTERVAL_MINUTES} minutes")
print(f"Expected telemetry rows: {
    MACHINE_COUNT * HISTORY_DAYS * 24 * (60 // TELEMETRY_INTERVAL_MINUTES)
:,}")
print(f"Execution time: {datetime.now()}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from pyspark.sql import functions as F

machine_master_df = (
    spark.range(1, MACHINE_COUNT + 1)
    .withColumn(
        "machine_id",
        F.format_string("M%04d", F.col("id"))
    )
    .withColumn(
        "machine_type",
        F.element_at(
            F.array(
                F.lit("CNC"),
                F.lit("PRESS"),
                F.lit("PUMP"),
                F.lit("COMPRESSOR")
            ),
            ((F.col("id") - 1) % 4 + 1).cast("int")
        )
    )
    .withColumn(
        "manufacturer",
        F.element_at(
            F.array(
                F.lit("NordTech"),
                F.lit("MechaWorks"),
                F.lit("InduMotion")
            ),
            ((F.col("id") - 1) % 3 + 1).cast("int")
        )
    )
    .withColumn(
        "installation_date",
        F.date_sub(
            F.current_date(),
            (365 + F.pmod(F.col("id") * 137, F.lit(3650))).cast("int")
        )
    )
    .withColumn(
        "location",
        F.element_at(
            F.array(
                F.lit("Warsaw"),
                F.lit("Gdansk"),
                F.lit("Wroclaw"),
                F.lit("Poznan")
            ),
            ((F.col("id") - 1) % 4 + 1).cast("int")
        )
    )
    .withColumn("is_active", F.lit(True))
    .drop("id")
)

machine_master_df.printSchema()
display(machine_master_df.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

validation_result = machine_master_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("machine_id").alias("unique_machine_ids"),
    F.sum(
        F.when(F.col("machine_id").isNull(), 1).otherwise(0)
    ).alias("null_machine_ids")
)

display(validation_result)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

TABLE_NAME = "bronze_machine_master"

(
    machine_master_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

print(f"Saved Delta table: {TABLE_NAME}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

failure_events_df = (
    spark.range(1, 151)
    .withColumn(
        "failure_id",
        F.format_string("F%05d", F.col("id"))
    )
    .withColumn(
        "machine_number",
        (F.pmod(F.col("id") * 37, F.lit(MACHINE_COUNT)) + 1).cast("long")
    )
    .withColumn(
        "machine_id",
        F.format_string("M%04d", F.col("machine_number"))
    )
    .withColumn(
        "failure_date",
        F.date_sub(
            F.current_date(),
            (1 + F.pmod(F.col("id") * 29, F.lit(HISTORY_DAYS - 5))).cast("int")
        )
    )
    .withColumn(
        "failure_type",
        F.element_at(
            F.array(
                F.lit("OVERHEATING"),
                F.lit("BEARING_FAILURE"),
                F.lit("PRESSURE_LOSS"),
                F.lit("ELECTRICAL_FAILURE")
            ),
            (F.pmod(F.col("id") - 1, F.lit(4)) + 1).cast("int")
        )
    )
    .withColumn(
        "severity",
        F.element_at(
            F.array(
                F.lit("LOW"),
                F.lit("MEDIUM"),
                F.lit("HIGH"),
                F.lit("CRITICAL")
            ),
            (F.pmod(F.col("id") * 3, F.lit(4)) + 1).cast("int")
        )
    )
    .withColumn(
        "downtime_hours",
        F.round(
            F.lit(1.0) + F.pmod(F.col("id") * 17, F.lit(95)) / F.lit(4.0),
            2
        )
    )
    .select(
        "failure_id",
        "machine_id",
        "failure_date",
        "failure_type",
        "severity",
        "downtime_hours"
    )
)

failure_events_df.printSchema()
display(failure_events_df.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

failure_validation_df = failure_events_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("failure_id").alias("unique_failure_ids"),
    F.countDistinct("machine_id").alias("machines_with_failures"),
    F.sum(
        F.when(F.col("failure_id").isNull(), 1).otherwise(0)
    ).alias("null_failure_ids"),
    F.sum(
        F.when(F.col("machine_id").isNull(), 1).otherwise(0)
    ).alias("null_machine_ids")
)

display(failure_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

machine_master_saved_df = spark.table("bronze_machine_master")

orphan_failures_df = (
    failure_events_df.alias("failures")
    .join(
        machine_master_saved_df.alias("machines"),
        on="machine_id",
        how="left_anti"
    )
)

orphan_failure_count = orphan_failures_df.count()

print(f"Failure records with unknown machine_id: {orphan_failure_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

TABLE_NAME = "bronze_failure_events"

(
    failure_events_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_failure_count = spark.table(TABLE_NAME).count()

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_failure_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

maintenance_events_df = (
    spark.range(1, 301)
    .withColumn(
        "maintenance_id",
        F.format_string("MN%05d", F.col("id"))
    )
    .withColumn(
        "machine_number",
        (F.pmod(F.col("id") * 43, F.lit(MACHINE_COUNT)) + 1).cast("long")
    )
    .withColumn(
        "machine_id",
        F.format_string("M%04d", F.col("machine_number"))
    )
    .withColumn(
        "maintenance_date",
        F.date_sub(
            F.current_date(),
            (1 + F.pmod(F.col("id") * 17, F.lit(HISTORY_DAYS))).cast("int")
        )
    )
    .withColumn(
        "maintenance_type",
        F.element_at(
            F.array(
                F.lit("INSPECTION"),
                F.lit("PREVENTIVE"),
                F.lit("CORRECTIVE"),
                F.lit("COMPONENT_REPLACEMENT")
            ),
            (F.pmod(F.col("id") - 1, F.lit(4)) + 1).cast("int")
        )
    )
    .withColumn(
        "component",
        F.element_at(
            F.array(
                F.lit("BEARING"),
                F.lit("MOTOR"),
                F.lit("PUMP"),
                F.lit("ELECTRICAL_SYSTEM"),
                F.lit("COOLING_SYSTEM")
            ),
            (F.pmod(F.col("id") * 7, F.lit(5)) + 1).cast("int")
        )
    )
    .withColumn(
        "duration_hours",
        F.round(
            F.lit(0.5)
            + F.pmod(F.col("id") * 11, F.lit(31)) / F.lit(4.0),
            2
        )
    )
    .withColumn(
        "maintenance_cost",
        F.round(
            F.lit(100.0)
            + F.pmod(F.col("id") * 157, F.lit(4900)),
            2
        )
    )
    .select(
        "maintenance_id",
        "machine_id",
        "maintenance_date",
        "maintenance_type",
        "component",
        "duration_hours",
        "maintenance_cost"
    )
)

maintenance_events_df.printSchema()
display(maintenance_events_df.limit(10))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

maintenance_validation_df = maintenance_events_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("maintenance_id").alias("unique_maintenance_ids"),
    F.countDistinct("machine_id").alias("machines_with_maintenance"),
    F.sum(
        F.when(F.col("maintenance_id").isNull(), 1).otherwise(0)
    ).alias("null_maintenance_ids"),
    F.sum(
        F.when(F.col("machine_id").isNull(), 1).otherwise(0)
    ).alias("null_machine_ids")
)

display(maintenance_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

orphan_maintenance_count = (
    maintenance_events_df.alias("maintenance")
    .join(
        spark.table("bronze_machine_master").alias("machines"),
        on="machine_id",
        how="left_anti"
    )
    .count()
)

assert orphan_maintenance_count == 0, (
    f"Found {orphan_maintenance_count} records with unknown machine_id"
)

TABLE_NAME = "bronze_maintenance_events"

(
    maintenance_events_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_maintenance_count = spark.table(TABLE_NAME).count()

print(f"Maintenance records with unknown machine_id: {orphan_maintenance_count}")
print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_maintenance_count}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

from datetime import datetime, timedelta

END_TIMESTAMP = datetime.now().replace(
    minute=0,
    second=0,
    microsecond=0
)

START_TIMESTAMP = END_TIMESTAMP - timedelta(days=HISTORY_DAYS)

INTERVALS_PER_DAY = 24 * (60 // TELEMETRY_INTERVAL_MINUTES)
INTERVAL_COUNT = HISTORY_DAYS * INTERVALS_PER_DAY

telemetry_timestamps_df = (
    spark.range(0, INTERVAL_COUNT)
    .withColumn(
        "telemetry_timestamp",
        F.expr(
            f"timestamp'{START_TIMESTAMP:%Y-%m-%d %H:%M:%S}' "
            f"+ INTERVAL {TELEMETRY_INTERVAL_MINUTES} MINUTES * id"
        )
    )
    .select("telemetry_timestamp")
)

print(f"Start: {START_TIMESTAMP}")
print(f"End boundary: {END_TIMESTAMP}")
print(f"Expected timestamps: {INTERVAL_COUNT}")
print(f"Generated timestamps: {telemetry_timestamps_df.count()}")

display(telemetry_timestamps_df.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

machine_ids_df = (
    spark.table("bronze_machine_master")
    .select("machine_id")
)

telemetry_base_df = (
    machine_ids_df
    .crossJoin(F.broadcast(telemetry_timestamps_df))
    .withColumn(
        "telemetry_id",
        F.concat(
            F.col("machine_id"),
            F.lit("_"),
            F.date_format(
                F.col("telemetry_timestamp"),
                "yyyyMMddHHmm"
            )
        )
    )
    .select(
        "telemetry_id",
        "machine_id",
        "telemetry_timestamp"
    )
)

telemetry_base_df = telemetry_base_df.cache()

generated_telemetry_count = telemetry_base_df.count()

print(f"Expected telemetry rows: {MACHINE_COUNT * INTERVAL_COUNT:,}")
print(f"Generated telemetry rows: {generated_telemetry_count:,}")

display(telemetry_base_df.limit(5))

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

telemetry_sensor_df = (
    telemetry_base_df
    .withColumn(
        "machine_number",
        F.regexp_extract("machine_id", r"(\d+)", 1).cast("int")
    )
    .withColumn(
        "time_index",
        (
            F.unix_timestamp("telemetry_timestamp")
            / (TELEMETRY_INTERVAL_MINUTES * 60)
        ).cast("long")
    )

    # Deterministyczny szum zależny od maszyny i czasu
    .withColumn(
        "noise_1",
        (
            F.pmod(
                F.hash("machine_id", "telemetry_timestamp"),
                F.lit(1000)
            ) / 1000.0
        ) - 0.5
    )
    .withColumn(
        "noise_2",
        (
            F.pmod(
                F.hash("telemetry_timestamp", "machine_id", F.lit("sensor2")),
                F.lit(1000)
            ) / 1000.0
        ) - 0.5
    )

    # Cykl dobowy
    .withColumn(
        "daily_cycle",
        F.sin(
            2 * 3.141592653589793
            * F.hour("telemetry_timestamp")
            / 24.0
        )
    )

    # Pomiary czujników
    .withColumn(
        "temperature_c",
        F.round(
            62.0
            + (F.col("machine_number") % 8) * 0.7
            + 4.0 * F.col("daily_cycle")
            + 3.0 * F.col("noise_1"),
            2
        )
    )
    .withColumn(
        "vibration_mm_s",
        F.round(
            2.0
            + (F.col("machine_number") % 5) * 0.15
            + 0.8 * F.abs(F.col("noise_2")),
            3
        )
    )
    .withColumn(
        "pressure_bar",
        F.round(
            5.8
            + (F.col("machine_number") % 4) * 0.2
            + 0.4 * F.col("noise_1"),
            2
        )
    )
    .withColumn(
        "rotation_speed_rpm",
        F.round(
            1450.0
            + (F.col("machine_number") % 6) * 35.0
            + 80.0 * F.col("noise_2"),
            1
        )
    )
    .withColumn(
        "power_consumption_kw",
        F.round(
            21.0
            + (F.col("machine_number") % 7) * 1.2
            + 2.0 * F.col("daily_cycle")
            + 2.5 * F.col("noise_1"),
            2
        )
    )
    .withColumn(
        "load_percent",
        F.round(
            65.0
            + (F.col("machine_number") % 10) * 2.0
            + 8.0 * F.col("daily_cycle")
            + 5.0 * F.col("noise_2"),
            2
        )
    )
    .select(
        "telemetry_id",
        "machine_id",
        "telemetry_timestamp",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "rotation_speed_rpm",
        "power_consumption_kw",
        "load_percent"
    )
)

display(
    telemetry_sensor_df
    .orderBy("machine_id", "telemetry_timestamp")
    .limit(10)
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

telemetry_validation_df = telemetry_sensor_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("telemetry_id").alias("unique_telemetry_ids"),
    F.countDistinct("machine_id").alias("machine_count"),

    F.sum(
        F.when(F.col("telemetry_id").isNull(), 1).otherwise(0)
    ).alias("null_telemetry_ids"),

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

    F.min("temperature_c").alias("min_temperature_c"),
    F.max("temperature_c").alias("max_temperature_c"),

    F.min("vibration_mm_s").alias("min_vibration_mm_s"),
    F.max("vibration_mm_s").alias("max_vibration_mm_s"),

    F.min("pressure_bar").alias("min_pressure_bar"),
    F.max("pressure_bar").alias("max_pressure_bar")
)

display(telemetry_validation_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 17: Add pre-failure degradation signals and validate the result

failure_windows_df = (
    spark.table("bronze_failure_events")
    .select(
        "failure_id",
        "machine_id",
        F.col("failure_date").cast("timestamp").alias("failure_timestamp")
    )
    .withColumn(
        "window_start",
        F.expr("failure_timestamp - INTERVAL 7 DAYS")
    )
)

# Identify telemetry records within seven days before a failure
failure_signal_matches_df = (
    telemetry_sensor_df.alias("t")
    .join(
        F.broadcast(failure_windows_df).alias("f"),
        (
            (F.col("t.machine_id") == F.col("f.machine_id"))
            & (F.col("t.telemetry_timestamp") >= F.col("f.window_start"))
            & (F.col("t.telemetry_timestamp") < F.col("f.failure_timestamp"))
        ),
        "inner"
    )
    .withColumn(
        "hours_to_failure",
        (
            F.unix_timestamp("failure_timestamp")
            - F.unix_timestamp("telemetry_timestamp")
        ) / F.lit(3600.0)
    )
    .withColumn(
        "degradation_strength",
        F.greatest(
            F.lit(0.0),
            F.least(
                F.lit(1.0),
                F.lit(1.0)
                - F.col("hours_to_failure") / F.lit(7.0 * 24.0)
            )
        )
    )
    # Use the strongest signal when failure windows overlap
    .groupBy("telemetry_id")
    .agg(
        F.max("degradation_strength").alias("degradation_strength")
    )
)

# Join the degradation signal without duplicating telemetry records
telemetry_with_failure_signal_df = (
    telemetry_sensor_df.alias("t")
    .join(
        failure_signal_matches_df.alias("s"),
        on="telemetry_id",
        how="left"
    )
    .withColumn(
        "degradation_strength",
        F.coalesce(
            F.col("degradation_strength"),
            F.lit(0.0)
        )
    )
    .withColumn(
        "temperature_c",
        F.round(
            F.col("temperature_c")
            + 12.0 * F.col("degradation_strength"),
            2
        )
    )
    .withColumn(
        "vibration_mm_s",
        F.round(
            F.col("vibration_mm_s")
            + 5.0 * F.col("degradation_strength"),
            3
        )
    )
    .withColumn(
        "pressure_bar",
        F.round(
            F.col("pressure_bar")
            - 1.3 * F.col("degradation_strength"),
            2
        )
    )
    .withColumn(
        "power_consumption_kw",
        F.round(
            F.col("power_consumption_kw")
            + 8.0 * F.col("degradation_strength"),
            2
        )
    )
    .withColumn(
        "load_percent",
        F.round(
            F.least(
                F.lit(100.0),
                F.col("load_percent")
                + 10.0 * F.col("degradation_strength")
            ),
            2
        )
    )
    .withColumn(
        "pre_failure_signal",
        F.when(
            F.col("degradation_strength") > 0,
            F.lit(1)
        ).otherwise(F.lit(0))
    )
    .select(
        "telemetry_id",
        "machine_id",
        "telemetry_timestamp",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "rotation_speed_rpm",
        "power_consumption_kw",
        "load_percent",
        "pre_failure_signal"
    )
    .cache()
)

# Validate row uniqueness and degradation ranges
signal_summary_df = telemetry_with_failure_signal_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("telemetry_id").alias("unique_telemetry_ids"),
    F.sum("pre_failure_signal").alias("pre_failure_rows"),
    F.max("temperature_c").alias("max_temperature_c"),
    F.max("vibration_mm_s").alias("max_vibration_mm_s"),
    F.min("pressure_bar").alias("min_pressure_bar")
)

display(signal_summary_df)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 18: Save the Bronze telemetry Delta table

TABLE_NAME = "bronze_machine_telemetry"

telemetry_to_save_df = (
    telemetry_with_failure_signal_df
    .drop("pre_failure_signal")
)

(
    telemetry_to_save_df.write
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
print("Pre-failure helper column excluded: True")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 19: Validate all Bronze Delta tables

expected_tables = {
    "bronze_machine_master": 100,
    "bronze_failure_events": 150,
    "bronze_maintenance_events": 300,
    "bronze_machine_telemetry": 1728000
}

validation_results = []

for table_name, expected_row_count in expected_tables.items():
    table_df = spark.table(table_name)
    actual_row_count = table_df.count()
    column_count = len(table_df.columns)
    status = "PASS" if actual_row_count == expected_row_count else "FAIL"

    validation_results.append(
        (
            table_name,
            expected_row_count,
            actual_row_count,
            column_count,
            status
        )
    )

bronze_validation_df = spark.createDataFrame(
    validation_results,
    [
        "table_name",
        "expected_row_count",
        "actual_row_count",
        "column_count",
        "validation_status"
    ]
)

failed_table_count = (
    bronze_validation_df
    .filter(F.col("validation_status") == "FAIL")
    .count()
)

assert failed_table_count == 0, (
    f"Bronze validation failed for {failed_table_count} table(s)"
)

display(
    bronze_validation_df.orderBy("table_name")
)

print("Bronze layer validation completed successfully.")
print(f"Validated tables: {len(expected_tables)}")
print("Failed tables: 0")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 43: Rebuild telemetry with realistic and non-deterministic degradation signals

from pyspark.sql import functions as F

TELEMETRY_INTERVAL_MINUTES = 15
EXPECTED_ROW_COUNT = 1728000
TABLE_NAME = "bronze_machine_telemetry"

existing_telemetry_df = (
    spark.table(TABLE_NAME)
    .select(
        "telemetry_id",
        "machine_id",
        "telemetry_timestamp"
    )
)

failure_events_df = spark.table("bronze_failure_events")

# Rebuild clean baseline sensor measurements
baseline_telemetry_df = (
    existing_telemetry_df
    .withColumn(
        "machine_number",
        F.regexp_extract("machine_id", r"(\d+)", 1).cast("int")
    )
    .withColumn(
        "noise_1",
        (
            F.pmod(
                F.hash(
                    "machine_id",
                    "telemetry_timestamp",
                    F.lit("noise_1")
                ),
                F.lit(10000)
            ) / F.lit(10000.0)
        ) - F.lit(0.5)
    )
    .withColumn(
        "noise_2",
        (
            F.pmod(
                F.hash(
                    "machine_id",
                    "telemetry_timestamp",
                    F.lit("noise_2")
                ),
                F.lit(10000)
            ) / F.lit(10000.0)
        ) - F.lit(0.5)
    )
    .withColumn(
        "daily_cycle",
        F.sin(
            F.lit(2.0 * 3.141592653589793)
            * (
                F.hour("telemetry_timestamp")
                + F.minute("telemetry_timestamp") / F.lit(60.0)
            )
            / F.lit(24.0)
        )
    )
    .withColumn(
        "temperature_c",
        F.lit(62.0)
        + (F.col("machine_number") % 8) * F.lit(0.7)
        + F.lit(4.0) * F.col("daily_cycle")
        + F.lit(5.0) * F.col("noise_1")
    )
    .withColumn(
        "vibration_mm_s",
        F.lit(2.0)
        + (F.col("machine_number") % 5) * F.lit(0.15)
        + F.lit(1.3) * F.abs(F.col("noise_2"))
    )
    .withColumn(
        "pressure_bar",
        F.lit(5.8)
        + (F.col("machine_number") % 4) * F.lit(0.2)
        + F.lit(0.7) * F.col("noise_1")
    )
    .withColumn(
        "rotation_speed_rpm",
        F.lit(1450.0)
        + (F.col("machine_number") % 6) * F.lit(35.0)
        + F.lit(130.0) * F.col("noise_2")
    )
    .withColumn(
        "power_consumption_kw",
        F.lit(21.0)
        + (F.col("machine_number") % 7) * F.lit(1.2)
        + F.lit(2.0) * F.col("daily_cycle")
        + F.lit(4.0) * F.col("noise_1")
    )
    .withColumn(
        "load_percent",
        F.lit(65.0)
        + (F.col("machine_number") % 10) * F.lit(2.0)
        + F.lit(8.0) * F.col("daily_cycle")
        + F.lit(8.0) * F.col("noise_2")
    )
)

# Create deterministic but variable warning profiles
failure_profiles_df = (
    failure_events_df
    .select(
        "failure_id",
        "machine_id",
        "failure_type",
        F.col("failure_date").cast("timestamp").alias("failure_timestamp")
    )
    .withColumn(
        "warning_window_days",
        (
            F.lit(2)
            + F.pmod(
                F.hash("failure_id", F.lit("window")),
                F.lit(13)
            )
        ).cast("int")
    )
    .withColumn(
        "signal_probability",
        F.pmod(
            F.hash("failure_id", F.lit("signal")),
            F.lit(1000)
        ) / F.lit(1000.0)
    )
    .withColumn(
        "signal_enabled",
        F.col("signal_probability") >= F.lit(0.20)
    )
    .withColumn(
        "signal_intensity",
        F.lit(0.35)
        + (
            F.pmod(
                F.hash("failure_id", F.lit("intensity")),
                F.lit(651)
            ) / F.lit(1000.0)
        )
    )
    .withColumn(
        "warning_start",
        F.expr(
            "failure_timestamp "
            "- make_interval(0, 0, 0, warning_window_days)"
        )
    )
)

# Match telemetry to enabled warning windows
failure_matches_df = (
    baseline_telemetry_df.alias("telemetry")
    .join(
        F.broadcast(failure_profiles_df).alias("failure"),
        (
            (
                F.col("telemetry.machine_id")
                == F.col("failure.machine_id")
            )
            & F.col("failure.signal_enabled")
            & (
                F.col("telemetry.telemetry_timestamp")
                >= F.col("failure.warning_start")
            )
            & (
                F.col("telemetry.telemetry_timestamp")
                < F.col("failure.failure_timestamp")
            )
        ),
        "inner"
    )
    .withColumn(
        "hours_to_failure",
        (
            F.unix_timestamp("failure_timestamp")
            - F.unix_timestamp("telemetry_timestamp")
        ) / F.lit(3600.0)
    )
    .withColumn(
        "window_hours",
        F.col("warning_window_days") * F.lit(24.0)
    )
    .withColumn(
        "progress",
        F.greatest(
            F.lit(0.0),
            F.least(
                F.lit(1.0),
                F.lit(1.0)
                - F.col("hours_to_failure") / F.col("window_hours")
            )
        )
    )
    .withColumn(
        "local_variation",
        F.lit(0.65)
        + (
            F.pmod(
                F.hash(
                    F.col("telemetry.telemetry_id"),
                    F.col("failure.failure_id")
                ),
                F.lit(701)
            ) / F.lit(1000.0)
        )
    )
    .withColumn(
        "signal_strength",
        F.col("signal_intensity")
        * F.pow(F.col("progress"), F.lit(1.4))
        * F.col("local_variation")
    )
    .withColumn(
        "temperature_effect",
        F.when(
            F.col("failure_type").isin(
                "OVERHEATING",
                "ELECTRICAL_FAILURE"
            ),
            F.lit(5.0) * F.col("signal_strength")
        ).otherwise(F.lit(0.8) * F.col("signal_strength"))
    )
    .withColumn(
        "vibration_effect",
        F.when(
            F.col("failure_type") == "BEARING_FAILURE",
            F.lit(2.2) * F.col("signal_strength")
        ).otherwise(F.lit(0.35) * F.col("signal_strength"))
    )
    .withColumn(
        "pressure_effect",
        F.when(
            F.col("failure_type") == "PRESSURE_LOSS",
            F.lit(-0.65) * F.col("signal_strength")
        ).otherwise(F.lit(-0.08) * F.col("signal_strength"))
    )
    .withColumn(
        "rotation_effect",
        F.when(
            F.col("failure_type").isin(
                "BEARING_FAILURE",
                "ELECTRICAL_FAILURE"
            ),
            F.lit(-75.0) * F.col("signal_strength")
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "power_effect",
        F.when(
            F.col("failure_type").isin(
                "OVERHEATING",
                "ELECTRICAL_FAILURE"
            ),
            F.lit(4.0) * F.col("signal_strength")
        ).otherwise(F.lit(0.6) * F.col("signal_strength"))
    )
    .withColumn(
        "load_effect",
        F.lit(4.0) * F.col("signal_strength")
    )
    .groupBy(
        F.col("telemetry.telemetry_id").alias("telemetry_id")
    )
    .agg(
        F.max("temperature_effect").alias("temperature_effect"),
        F.max("vibration_effect").alias("vibration_effect"),
        F.min("pressure_effect").alias("pressure_effect"),
        F.min("rotation_effect").alias("rotation_effect"),
        F.max("power_effect").alias("power_effect"),
        F.max("load_effect").alias("load_effect")
    )
)

# Add unrelated background anomalies to create class overlap
final_telemetry_df = (
    baseline_telemetry_df.alias("baseline")
    .join(
        failure_matches_df.alias("effects"),
        on="telemetry_id",
        how="left"
    )
    .fillna(
        {
            "temperature_effect": 0.0,
            "vibration_effect": 0.0,
            "pressure_effect": 0.0,
            "rotation_effect": 0.0,
            "power_effect": 0.0,
            "load_effect": 0.0
        }
    )
    .withColumn(
        "background_anomaly_score",
        F.pmod(
            F.hash(
                "telemetry_id",
                F.lit("background_anomaly")
            ),
            F.lit(10000)
        ) / F.lit(10000.0)
    )
    .withColumn(
        "background_anomaly_strength",
        F.when(
            F.col("background_anomaly_score") < F.lit(0.015),
            F.lit(0.25)
            + F.col("background_anomaly_score")
            / F.lit(0.015)
        ).otherwise(F.lit(0.0))
    )
    .withColumn(
        "temperature_c",
        F.round(
            F.col("temperature_c")
            + F.col("temperature_effect")
            + F.lit(3.0) * F.col("background_anomaly_strength"),
            2
        )
    )
    .withColumn(
        "vibration_mm_s",
        F.round(
            F.col("vibration_mm_s")
            + F.col("vibration_effect")
            + F.lit(1.2) * F.col("background_anomaly_strength"),
            3
        )
    )
    .withColumn(
        "pressure_bar",
        F.round(
            F.greatest(
                F.lit(0.5),
                F.col("pressure_bar")
                + F.col("pressure_effect")
                - F.lit(0.25) * F.col("background_anomaly_strength")
            ),
            2
        )
    )
    .withColumn(
        "rotation_speed_rpm",
        F.round(
            F.greatest(
                F.lit(100.0),
                F.col("rotation_speed_rpm")
                + F.col("rotation_effect")
            ),
            1
        )
    )
    .withColumn(
        "power_consumption_kw",
        F.round(
            F.greatest(
                F.lit(0.0),
                F.col("power_consumption_kw")
                + F.col("power_effect")
                + F.lit(2.0) * F.col("background_anomaly_strength")
            ),
            2
        )
    )
    .withColumn(
        "load_percent",
        F.round(
            F.greatest(
                F.lit(0.0),
                F.least(
                    F.lit(100.0),
                    F.col("load_percent")
                    + F.col("load_effect")
                    + F.lit(3.0) * F.col("background_anomaly_strength")
                )
            ),
            2
        )
    )
    .select(
        "telemetry_id",
        "machine_id",
        "telemetry_timestamp",
        "temperature_c",
        "vibration_mm_s",
        "pressure_bar",
        "rotation_speed_rpm",
        "power_consumption_kw",
        "load_percent"
    )
)

# Save the rebuilt telemetry table
(
    final_telemetry_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_telemetry_df = spark.table(TABLE_NAME)

validation_df = saved_telemetry_df.agg(
    F.count("*").alias("row_count"),
    F.countDistinct("telemetry_id").alias("unique_telemetry_ids"),
    F.countDistinct("machine_id").alias("machine_count"),
    F.min("temperature_c").alias("minimum_temperature_c"),
    F.max("temperature_c").alias("maximum_temperature_c"),
    F.min("vibration_mm_s").alias("minimum_vibration_mm_s"),
    F.max("vibration_mm_s").alias("maximum_vibration_mm_s"),
    F.min("pressure_bar").alias("minimum_pressure_bar"),
    F.max("pressure_bar").alias("maximum_pressure_bar")
)

validation_row = validation_df.first()

assert validation_row["row_count"] == EXPECTED_ROW_COUNT, (
    f"Unexpected row count: {validation_row['row_count']}"
)

assert validation_row["unique_telemetry_ids"] == EXPECTED_ROW_COUNT, (
    "Unexpected unique telemetry ID count: "
    f"{validation_row['unique_telemetry_ids']}"
)

assert validation_row["machine_count"] == 100, (
    f"Unexpected machine count: {validation_row['machine_count']}"
)

display(validation_df)

print(f"Rebuilt Delta table: {TABLE_NAME}")
print(f"Saved rows: {validation_row['row_count']:,}")
print(
    "Synthetic warning profiles enabled for approximately "
    "80 percent of failures."
)
print(
    "Variable warning windows, failure-specific effects, and "
    "background anomalies applied."
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
