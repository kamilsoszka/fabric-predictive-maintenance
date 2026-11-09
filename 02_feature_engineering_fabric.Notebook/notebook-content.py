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

# # 02_feature_engineering_fabric
# 
# Leakage-aware daily feature engineering with completeness diagnostics, maintenance-history indicators, seven-day targets, and trend features.

# CELL ********************

from pyspark.sql import functions as F
from pyspark.sql import Window

telemetry = spark.table("silver_machine_telemetry")
maintenance = spark.table("silver_maintenance_events")
failures = spark.table("silver_failure_events")
machines = spark.table("silver_machine_master")
EXPECTED_SAMPLES_PER_DAY = 96

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

daily_all = (
    telemetry.withColumn("feature_date", F.to_date("telemetry_timestamp"))
    .groupBy("machine_id", "feature_date")
    .agg(
        F.count("*").alias("telemetry_sample_count"),
        F.avg("temperature_c").alias("temperature_avg"), F.min("temperature_c").alias("temperature_min"), F.max("temperature_c").alias("temperature_max"), F.stddev("temperature_c").alias("temperature_std"),
        F.avg("vibration_mm_s").alias("vibration_avg"), F.max("vibration_mm_s").alias("vibration_max"), F.stddev("vibration_mm_s").alias("vibration_std"),
        F.avg("pressure_bar").alias("pressure_avg"), F.min("pressure_bar").alias("pressure_min"), F.stddev("pressure_bar").alias("pressure_std"),
        F.avg("rotation_speed_rpm").alias("rotation_speed_avg"), F.stddev("rotation_speed_rpm").alias("rotation_speed_std"),
        F.avg("power_consumption_kw").alias("power_consumption_avg"), F.max("power_consumption_kw").alias("power_consumption_max"),
        F.avg("load_percent").alias("load_avg"), F.max("load_percent").alias("load_max")
    )
)
incomplete_daily = daily_all.filter(F.col("telemetry_sample_count") != EXPECTED_SAMPLES_PER_DAY)
print(f"Incomplete machine-days excluded: {incomplete_daily.count():,}")
daily = daily_all.filter(F.col("telemetry_sample_count") == EXPECTED_SAMPLES_PER_DAY)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

history = (
    daily.alias("d").join(maintenance.alias("m"),
        (F.col("d.machine_id") == F.col("m.machine_id")) & (F.col("m.maintenance_date") <= F.col("d.feature_date")), "left")
    .groupBy(F.col("d.machine_id").alias("machine_id"), F.col("d.feature_date").alias("feature_date"))
    .agg(
        F.max("m.maintenance_date").alias("last_maintenance_date"),
        F.countDistinct(F.when(F.col("m.maintenance_date") >= F.date_sub(F.col("d.feature_date"), 29), F.col("m.maintenance_id"))).alias("maintenance_count_30d"),
        F.sum(F.when(F.col("m.maintenance_date") >= F.date_sub(F.col("d.feature_date"), 29), F.col("m.duration_hours")).otherwise(0.0)).alias("maintenance_duration_hours_30d"),
        F.sum(F.when(F.col("m.maintenance_date") >= F.date_sub(F.col("d.feature_date"), 29), F.col("m.maintenance_cost")).otherwise(0.0)).alias("maintenance_cost_30d")
    )
    .withColumn("has_maintenance_history", F.when(F.col("last_maintenance_date").isNotNull(), 1).otherwise(0))
    .withColumn("days_since_last_maintenance", F.when(F.col("last_maintenance_date").isNotNull(), F.datediff("feature_date", "last_maintenance_date")).otherwise(0))
    .drop("last_maintenance_date")
)
features = daily.join(history, ["machine_id", "feature_date"], "left").fillna(0)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

future_targets = (
    features.alias("x").join(failures.alias("f"),
        (F.col("x.machine_id") == F.col("f.machine_id")) &
        (F.col("f.failure_date") > F.col("x.feature_date")) &
        (F.col("f.failure_date") <= F.date_add(F.col("x.feature_date"), 7)), "left")
    .groupBy(F.col("x.machine_id").alias("machine_id"), F.col("x.feature_date").alias("feature_date"))
    .agg(F.when(F.count("f.failure_id") > 0, 1).otherwise(0).alias("failure_within_7d"))
)
features = features.join(future_targets, ["machine_id", "feature_date"], "inner")

w = Window.partitionBy("machine_id").orderBy("feature_date")
for column in ["temperature_avg", "vibration_avg", "pressure_avg", "power_consumption_avg", "load_avg"]:
    features = features.withColumn(f"{column}_change_1d", F.col(column) - F.lag(column, 1).over(w))
    features = features.withColumn(f"{column}_rolling_7d", F.avg(column).over(w.rowsBetween(-6, 0)))
features = features.fillna(0.0, subset=[c for c in features.columns if c.endswith("_change_1d")])

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

final_df = (
    features.join(machines, "machine_id", "inner")
    .withColumn("machine_age_days", F.datediff("feature_date", "installation_date"))
    .drop("installation_date")
)
for forbidden in ["failure_date", "failure_id", "next_failure_date", "pre_failure_signal", "degradation_strength"]:
    assert forbidden not in final_df.columns
assert final_df.filter(F.col("machine_age_days") < 0).count() == 0
assert final_df.select("machine_id", "feature_date").distinct().count() == final_df.count()
assert final_df.filter(F.col("failure_within_7d").isNull()).count() == 0

(final_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable("gold_machine_features_daily"))
summary = spark.table("gold_machine_features_daily").agg(
    F.count("*").alias("row_count"),
    F.countDistinct("machine_id").alias("machine_count"),
    F.sum("failure_within_7d").alias("positive_rows"),
    F.avg(F.col("failure_within_7d").cast("double")).alias("positive_rate")
)
display(summary)

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
