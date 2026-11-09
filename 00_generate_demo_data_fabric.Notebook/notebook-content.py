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

# # 00_generate_demo_data_fabric
# 
# Deterministic synthetic Bronze-layer generator for the predictive-maintenance portfolio project. All dates are anchored to a fixed UTC reference timestamp so repeated runs produce the same logical dataset.

# CELL ********************

from datetime import datetime, timedelta
from pyspark.sql import functions as F

PROJECT_NAME = "fabric_predictive_maintenance_ml"
REFERENCE_TIMESTAMP = datetime(2026, 11, 6, 14, 0, 0)
REFERENCE_DATE = REFERENCE_TIMESTAMP.date()
MACHINE_COUNT = 100
HISTORY_DAYS = 180
TELEMETRY_INTERVAL_MINUTES = 15
INTERVALS_PER_DAY = 24 * (60 // TELEMETRY_INTERVAL_MINUTES)
INTERVAL_COUNT = HISTORY_DAYS * INTERVALS_PER_DAY
EXPECTED_TELEMETRY_ROWS = MACHINE_COUNT * INTERVAL_COUNT

print(f"Project: {PROJECT_NAME}")
print(f"Reference timestamp: {REFERENCE_TIMESTAMP} UTC")
print(f"Expected telemetry rows: {EXPECTED_TELEMETRY_ROWS:,}")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

machine_master_df = (
    spark.range(1, MACHINE_COUNT + 1)
    .withColumn("machine_id", F.format_string("M%04d", F.col("id")))
    .withColumn("machine_type", F.element_at(F.array(*[F.lit(v) for v in ["CNC", "PRESS", "PUMP", "COMPRESSOR"]]), ((F.col("id") - 1) % 4 + 1).cast("int")))
    .withColumn("manufacturer", F.element_at(F.array(*[F.lit(v) for v in ["NordTech", "MechaWorks", "InduMotion"]]), ((F.col("id") - 1) % 3 + 1).cast("int")))
    .withColumn("installation_date", F.date_sub(F.lit(REFERENCE_DATE), (365 + F.pmod(F.col("id") * 137, F.lit(3650))).cast("int")))
    .withColumn("location", F.element_at(F.array(*[F.lit(v) for v in ["Warsaw", "Gdansk", "Wroclaw", "Poznan"]]), ((F.col("id") - 1) % 4 + 1).cast("int")))
    .withColumn("is_active", F.lit(True))
    .drop("id")
)
assert machine_master_df.count() == MACHINE_COUNT
assert machine_master_df.select("machine_id").distinct().count() == MACHINE_COUNT

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

failure_events_df = (
    spark.range(1, 151)
    .withColumn("failure_id", F.format_string("F%05d", F.col("id")))
    .withColumn("machine_number", (F.pmod(F.col("id") * 37, F.lit(MACHINE_COUNT)) + 1).cast("long"))
    .withColumn("machine_id", F.format_string("M%04d", F.col("machine_number")))
    .withColumn("failure_date", F.date_sub(F.lit(REFERENCE_DATE), (1 + F.pmod(F.col("id") * 29, F.lit(HISTORY_DAYS - 5))).cast("int")))
    .withColumn("failure_type", F.element_at(F.array(*[F.lit(v) for v in ["OVERHEATING", "BEARING_FAILURE", "PRESSURE_LOSS", "ELECTRICAL_FAILURE"]]), (F.pmod(F.col("id") - 1, F.lit(4)) + 1).cast("int")))
    .withColumn("severity", F.element_at(F.array(*[F.lit(v) for v in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]]), (F.pmod(F.col("id") * 3, F.lit(4)) + 1).cast("int")))
    .withColumn("downtime_hours", F.round(F.lit(1.0) + F.pmod(F.col("id") * 17, F.lit(95)) / F.lit(4.0), 2))
    .select("failure_id", "machine_id", "failure_date", "failure_type", "severity", "downtime_hours")
)
maintenance_events_df = (
    spark.range(1, 301)
    .withColumn("maintenance_id", F.format_string("MN%05d", F.col("id")))
    .withColumn("machine_number", (F.pmod(F.col("id") * 43, F.lit(MACHINE_COUNT)) + 1).cast("long"))
    .withColumn("machine_id", F.format_string("M%04d", F.col("machine_number")))
    .withColumn("maintenance_date", F.date_sub(F.lit(REFERENCE_DATE), (1 + F.pmod(F.col("id") * 17, F.lit(HISTORY_DAYS))).cast("int")))
    .withColumn("maintenance_type", F.element_at(F.array(*[F.lit(v) for v in ["INSPECTION", "PREVENTIVE", "CORRECTIVE", "COMPONENT_REPLACEMENT"]]), (F.pmod(F.col("id") - 1, F.lit(4)) + 1).cast("int")))
    .withColumn("component", F.element_at(F.array(*[F.lit(v) for v in ["BEARING", "MOTOR", "PUMP", "ELECTRICAL_SYSTEM", "COOLING_SYSTEM"]]), (F.pmod(F.col("id") * 7, F.lit(5)) + 1).cast("int")))
    .withColumn("duration_hours", F.round(F.lit(0.5) + F.pmod(F.col("id") * 11, F.lit(31)) / F.lit(4.0), 2))
    .withColumn("maintenance_cost", F.round(F.lit(100.0) + F.pmod(F.col("id") * 157, F.lit(4900)), 2))
    .select("maintenance_id", "machine_id", "maintenance_date", "maintenance_type", "component", "duration_hours", "maintenance_cost")
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

start_timestamp = REFERENCE_TIMESTAMP - timedelta(days=HISTORY_DAYS)
timestamps_df = (
    spark.range(0, INTERVAL_COUNT)
    .withColumn("telemetry_timestamp", F.expr(f"timestamp'{start_timestamp:%Y-%m-%d %H:%M:%S}' + INTERVAL {TELEMETRY_INTERVAL_MINUTES} MINUTES * id"))
    .select("telemetry_timestamp")
)
base_df = (
    machine_master_df.select("machine_id").crossJoin(F.broadcast(timestamps_df))
    .withColumn("telemetry_id", F.concat_ws("_", "machine_id", F.date_format("telemetry_timestamp", "yyyyMMddHHmm")))
    .withColumn("machine_number", F.regexp_extract("machine_id", r"(\d+)", 1).cast("int"))
    .withColumn("noise_1", F.pmod(F.hash("machine_id", "telemetry_timestamp", F.lit("noise_1")), F.lit(10000)) / 10000.0 - 0.5)
    .withColumn("noise_2", F.pmod(F.hash("machine_id", "telemetry_timestamp", F.lit("noise_2")), F.lit(10000)) / 10000.0 - 0.5)
    .withColumn("daily_cycle", F.sin(F.lit(2.0 * 3.141592653589793) * (F.hour("telemetry_timestamp") + F.minute("telemetry_timestamp") / 60.0) / 24.0))
    .withColumn("temperature_c", F.lit(62.0) + (F.col("machine_number") % 8) * 0.7 + 4.0 * F.col("daily_cycle") + 5.0 * F.col("noise_1"))
    .withColumn("vibration_mm_s", F.lit(2.0) + (F.col("machine_number") % 5) * 0.15 + 1.3 * F.abs(F.col("noise_2")))
    .withColumn("pressure_bar", F.lit(5.8) + (F.col("machine_number") % 4) * 0.2 + 0.7 * F.col("noise_1"))
    .withColumn("rotation_speed_rpm", F.lit(1450.0) + (F.col("machine_number") % 6) * 35.0 + 130.0 * F.col("noise_2"))
    .withColumn("power_consumption_kw", F.lit(21.0) + (F.col("machine_number") % 7) * 1.2 + 2.0 * F.col("daily_cycle") + 4.0 * F.col("noise_1"))
    .withColumn("load_percent", F.lit(65.0) + (F.col("machine_number") % 10) * 2.0 + 8.0 * F.col("daily_cycle") + 8.0 * F.col("noise_2"))
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

profiles_df = (
    failure_events_df
    .select("failure_id", "machine_id", "failure_type", F.col("failure_date").cast("timestamp").alias("failure_timestamp"))
    .withColumn("warning_window_days", (F.lit(2) + F.pmod(F.hash("failure_id", F.lit("window")), F.lit(13))).cast("int"))
    .withColumn("signal_enabled", (F.pmod(F.hash("failure_id", F.lit("signal")), F.lit(1000)) / 1000.0) >= 0.20)
    .withColumn("signal_intensity", F.lit(0.35) + F.pmod(F.hash("failure_id", F.lit("intensity")), F.lit(651)) / 1000.0)
    .withColumn("warning_start", F.expr("failure_timestamp - make_interval(0, 0, 0, warning_window_days)"))
)
matches_df = (
    base_df.alias("t").join(F.broadcast(profiles_df).alias("f"),
        (F.col("t.machine_id") == F.col("f.machine_id")) & F.col("f.signal_enabled") &
        (F.col("t.telemetry_timestamp") >= F.col("f.warning_start")) & (F.col("t.telemetry_timestamp") < F.col("f.failure_timestamp")), "inner")
    .withColumn("hours_to_failure", (F.unix_timestamp("failure_timestamp") - F.unix_timestamp("telemetry_timestamp")) / 3600.0)
    .withColumn("progress", F.greatest(F.lit(0.0), F.least(F.lit(1.0), F.lit(1.0) - F.col("hours_to_failure") / (F.col("warning_window_days") * 24.0))))
    .withColumn("variation", F.lit(0.65) + F.pmod(F.hash(F.col("t.telemetry_id"), F.col("f.failure_id")), F.lit(701)) / 1000.0)
    .withColumn("strength", F.col("signal_intensity") * F.pow("progress", 1.4) * F.col("variation"))
    .groupBy(F.col("t.telemetry_id").alias("telemetry_id"))
    .agg(
        F.max(F.when(F.col("failure_type").isin("OVERHEATING", "ELECTRICAL_FAILURE"), 5.0 * F.col("strength")).otherwise(0.8 * F.col("strength"))).alias("temp_effect"),
        F.max(F.when(F.col("failure_type") == "BEARING_FAILURE", 2.2 * F.col("strength")).otherwise(0.35 * F.col("strength"))).alias("vib_effect"),
        F.min(F.when(F.col("failure_type") == "PRESSURE_LOSS", -0.65 * F.col("strength")).otherwise(-0.08 * F.col("strength"))).alias("pressure_effect"),
        F.max(4.0 * F.col("strength")).alias("power_effect"),
        F.max(4.0 * F.col("strength")).alias("load_effect")
    )
)
telemetry_df = (
    base_df.join(matches_df, "telemetry_id", "left").fillna(0.0, subset=["temp_effect", "vib_effect", "pressure_effect", "power_effect", "load_effect"])
    .withColumn("background", F.when(F.pmod(F.hash("telemetry_id", F.lit("background")), F.lit(10000)) / 10000.0 < 0.015, 0.75).otherwise(0.0))
    .select(
        "telemetry_id", "machine_id", "telemetry_timestamp",
        F.round(F.col("temperature_c") + F.col("temp_effect") + 3.0 * F.col("background"), 2).alias("temperature_c"),
        F.round(F.col("vibration_mm_s") + F.col("vib_effect") + 1.2 * F.col("background"), 3).alias("vibration_mm_s"),
        F.round(F.greatest(F.lit(0.5), F.col("pressure_bar") + F.col("pressure_effect") - 0.25 * F.col("background")), 2).alias("pressure_bar"),
        F.round("rotation_speed_rpm", 1).alias("rotation_speed_rpm"),
        F.round(F.greatest(F.lit(0.0), F.col("power_consumption_kw") + F.col("power_effect") + 2.0 * F.col("background")), 2).alias("power_consumption_kw"),
        F.round(F.least(F.lit(100.0), F.greatest(F.lit(0.0), F.col("load_percent") + F.col("load_effect") + 3.0 * F.col("background"))), 2).alias("load_percent")
    )
)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

master_ids = machine_master_df.select("machine_id")
assert failure_events_df.join(master_ids, "machine_id", "left_anti").count() == 0
assert maintenance_events_df.join(master_ids, "machine_id", "left_anti").count() == 0
assert telemetry_df.join(master_ids, "machine_id", "left_anti").count() == 0
assert telemetry_df.count() == EXPECTED_TELEMETRY_ROWS
assert telemetry_df.select("telemetry_id").distinct().count() == EXPECTED_TELEMETRY_ROWS
assert telemetry_df.select("machine_id", "telemetry_timestamp").distinct().count() == EXPECTED_TELEMETRY_ROWS
assert telemetry_df.filter(
    F.col("temperature_c").isNull() | F.col("vibration_mm_s").isNull() | F.col("pressure_bar").isNull() |
    F.col("rotation_speed_rpm").isNull() | F.col("power_consumption_kw").isNull() | F.col("load_percent").isNull()
).count() == 0

for table_name, df in {
    "bronze_machine_master": machine_master_df,
    "bronze_failure_events": failure_events_df,
    "bronze_maintenance_events": maintenance_events_df,
    "bronze_machine_telemetry": telemetry_df
}.items():
    (df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(table_name))
    print(f"Saved {table_name}: {spark.table(table_name).count():,} rows")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
