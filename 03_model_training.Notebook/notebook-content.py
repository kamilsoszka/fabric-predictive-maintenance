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

# STEP 39: Create and validate the temporal dataset split

from pyspark.sql import functions as F
from pyspark.sql import Window

feature_dataset_df = spark.table("gold_machine_features_daily")

feature_dates = [
    row["feature_date"]
    for row in (
        feature_dataset_df
        .select("feature_date")
        .distinct()
        .orderBy("feature_date")
        .collect()
    )
]

date_count = len(feature_dates)

train_date_count = int(date_count * 0.70)
validation_date_count = int(date_count * 0.15)

train_end_date = feature_dates[train_date_count - 1]
validation_start_date = feature_dates[train_date_count]
validation_end_index = (
    train_date_count
    + validation_date_count
    - 1
)
validation_end_date = feature_dates[validation_end_index]
test_start_date = feature_dates[validation_end_index + 1]

split_dataset_df = (
    feature_dataset_df
    .withColumn(
        "dataset_split",
        F.when(
            F.col("feature_date") <= F.lit(train_end_date),
            F.lit("train")
        )
        .when(
            F.col("feature_date") <= F.lit(validation_end_date),
            F.lit("validation")
        )
        .otherwise(F.lit("test"))
    )
    .cache()
)

split_validation_df = (
    split_dataset_df
    .groupBy("dataset_split")
    .agg(
        F.count("*").alias("row_count"),
        F.countDistinct("feature_date").alias("date_count"),
        F.min("feature_date").alias("minimum_date"),
        F.max("feature_date").alias("maximum_date"),
        F.sum("failure_within_7d").alias("positive_rows"),
        F.round(
            F.avg(
                F.col("failure_within_7d").cast("double")
            ) * 100,
            2
        ).alias("positive_rate_percent")
    )
)

split_names = {
    row["dataset_split"]
    for row in split_validation_df.collect()
}

assert split_names == {
    "train",
    "validation",
    "test"
}, f"Unexpected dataset splits: {split_names}"

split_row_count = split_dataset_df.count()

assert split_row_count == 17900, (
    f"Unexpected total row count: {split_row_count}"
)

assert train_end_date < validation_start_date, (
    "Training and validation dates overlap"
)

assert validation_end_date < test_start_date, (
    "Validation and test dates overlap"
)

display(
    split_validation_df.orderBy(
        F.when(
            F.col("dataset_split") == "train",
            1
        )
        .when(
            F.col("dataset_split") == "validation",
            2
        )
        .otherwise(3)
    )
)

print(f"Total unique dates: {date_count}")
print(f"Training end date: {train_end_date}")
print(f"Validation start date: {validation_start_date}")
print(f"Validation end date: {validation_end_date}")
print(f"Test start date: {test_start_date}")
print("Temporal overlap detected: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 40: Save and validate the temporal dataset split

TABLE_NAME = "gold_machine_features_split"

(
    split_dataset_df.write
    .format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(TABLE_NAME)
)

saved_split_df = spark.table(TABLE_NAME)

saved_split_summary_df = (
    saved_split_df
    .groupBy("dataset_split")
    .agg(
        F.count("*").alias("row_count"),
        F.sum("failure_within_7d").alias("positive_rows")
    )
)

saved_summary = {
    row["dataset_split"]: {
        "row_count": row["row_count"],
        "positive_rows": row["positive_rows"]
    }
    for row in saved_split_summary_df.collect()
}

assert saved_split_df.count() == 17900, (
    f"Unexpected total row count: {saved_split_df.count()}"
)

assert saved_summary["train"]["row_count"] == 12500, (
    f"Unexpected training row count: "
    f"{saved_summary['train']['row_count']}"
)

assert saved_summary["validation"]["row_count"] == 2600, (
    f"Unexpected validation row count: "
    f"{saved_summary['validation']['row_count']}"
)

assert saved_summary["test"]["row_count"] == 2800, (
    f"Unexpected test row count: "
    f"{saved_summary['test']['row_count']}"
)

assert saved_summary["train"]["positive_rows"] == 757, (
    f"Unexpected training positive count: "
    f"{saved_summary['train']['positive_rows']}"
)

assert saved_summary["validation"]["positive_rows"] == 154, (
    f"Unexpected validation positive count: "
    f"{saved_summary['validation']['positive_rows']}"
)

assert saved_summary["test"]["positive_rows"] == 133, (
    f"Unexpected test positive count: "
    f"{saved_summary['test']['positive_rows']}"
)

display(
    saved_split_summary_df.orderBy(
        F.when(F.col("dataset_split") == "train", 1)
        .when(F.col("dataset_split") == "validation", 2)
        .otherwise(3)
    )
)

print(f"Saved Delta table: {TABLE_NAME}")
print(f"Saved rows: {saved_split_df.count():,}")
print(f"Saved columns: {len(saved_split_df.columns)}")
print("Temporal split validation completed successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 41: Train and validate the baseline Logistic Regression model

import pandas as pd
import numpy as np

from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load the temporal dataset split
split_feature_df = spark.table("gold_machine_features_split")

training_spark_df = (
    split_feature_df
    .filter(F.col("dataset_split") == "train")
)

validation_spark_df = (
    split_feature_df
    .filter(F.col("dataset_split") == "validation")
)

# Define columns excluded from model inputs
excluded_columns = {
    "machine_id",
    "feature_date",
    "failure_within_7d",
    "dataset_split",
    "is_active"
}

categorical_columns = [
    "machine_type",
    "manufacturer",
    "location"
]

# Select numeric model features
numeric_columns = [
    column_name
    for column_name, data_type in training_spark_df.dtypes
    if column_name not in excluded_columns
    and column_name not in categorical_columns
    and (
        data_type in {
            "tinyint",
            "smallint",
            "int",
            "bigint",
            "float",
            "double"
        }
        or data_type.startswith("decimal")
    )
]

model_feature_columns = categorical_columns + numeric_columns

assert len(numeric_columns) > 0, (
    "No numeric model features were detected"
)

# Convert the training and validation datasets to pandas
training_pdf = (
    training_spark_df
    .select(
        *model_feature_columns,
        "failure_within_7d"
    )
    .toPandas()
)

validation_pdf = (
    validation_spark_df
    .select(
        *model_feature_columns,
        "failure_within_7d"
    )
    .toPandas()
)

X_train = training_pdf[model_feature_columns]
y_train = training_pdf["failure_within_7d"].astype(int)

X_validation = validation_pdf[model_feature_columns]
y_validation = validation_pdf["failure_within_7d"].astype(int)

# Define numeric preprocessing
numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ]
)

# Define categorical preprocessing
categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "one_hot_encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ]
)

# Combine numeric and categorical preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_transformer,
            numeric_columns
        ),
        (
            "categorical",
            categorical_transformer,
            categorical_columns
        )
    ]
)

# Create the baseline model pipeline
logistic_regression_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                class_weight="balanced",
                max_iter=1000,
                random_state=42
            )
        )
    ]
)

# Train the model
logistic_regression_pipeline.fit(
    X_train,
    y_train
)

# Generate validation probabilities and predictions
validation_probabilities = (
    logistic_regression_pipeline
    .predict_proba(X_validation)[:, 1]
)

decision_threshold = 0.50

validation_predictions = (
    validation_probabilities >= decision_threshold
).astype(int)

# Calculate validation metrics
validation_confusion_matrix = confusion_matrix(
    y_validation,
    validation_predictions,
    labels=[0, 1]
)

true_negative, false_positive, false_negative, true_positive = (
    validation_confusion_matrix.ravel()
)

validation_metrics = {
    "model_name": "Logistic Regression",
    "decision_threshold": decision_threshold,
    "pr_auc": float(
        average_precision_score(
            y_validation,
            validation_probabilities
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_validation,
            validation_probabilities
        )
    ),
    "precision": float(
        precision_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_validation,
            validation_predictions,
            zero_division=0
        )
    ),
    "true_negative": int(true_negative),
    "false_positive": int(false_positive),
    "false_negative": int(false_negative),
    "true_positive": int(true_positive)
}

validation_metrics_df = pd.DataFrame(
    [validation_metrics]
)

# Validate the execution
assert len(training_pdf) == 12500, (
    f"Unexpected training row count: {len(training_pdf)}"
)

assert len(validation_pdf) == 2600, (
    f"Unexpected validation row count: {len(validation_pdf)}"
)

assert y_train.sum() == 757, (
    f"Unexpected training positive count: {int(y_train.sum())}"
)

assert y_validation.sum() == 154, (
    f"Unexpected validation positive count: "
    f"{int(y_validation.sum())}"
)

assert 0.0 <= validation_metrics["pr_auc"] <= 1.0, (
    f"Invalid validation PR-AUC: "
    f"{validation_metrics['pr_auc']}"
)

assert 0.0 <= validation_metrics["roc_auc"] <= 1.0, (
    f"Invalid validation ROC-AUC: "
    f"{validation_metrics['roc_auc']}"
)

display(validation_metrics_df)

print(f"Training rows: {len(training_pdf):,}")
print(f"Training positive rows: {int(y_train.sum()):,}")
print(f"Validation rows: {len(validation_pdf):,}")
print(f"Validation positive rows: {int(y_validation.sum()):,}")
print(f"Numeric features: {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")
print(f"Decision threshold: {decision_threshold:.2f}")
print("Test dataset used: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 42: Diagnose potential synthetic target leakage

from sklearn.metrics import roc_auc_score
import pandas as pd
import numpy as np

univariate_results = []

for feature_name in numeric_columns:
    feature_values = (
        validation_pdf[feature_name]
        .replace([np.inf, -np.inf], np.nan)
    )

    if feature_values.isna().all():
        continue

    median_value = feature_values.median()
    feature_values = feature_values.fillna(median_value)

    if feature_values.nunique() < 2:
        continue

    direct_auc = roc_auc_score(
        y_validation,
        feature_values
    )

    inverse_auc = roc_auc_score(
        y_validation,
        -feature_values
    )

    best_auc = max(
        direct_auc,
        inverse_auc
    )

    relationship_direction = (
        "higher_increases_risk"
        if direct_auc >= inverse_auc
        else "lower_increases_risk"
    )

    positive_mean = feature_values[
        y_validation.values == 1
    ].mean()

    negative_mean = feature_values[
        y_validation.values == 0
    ].mean()

    univariate_results.append(
        {
            "feature_name": feature_name,
            "univariate_roc_auc": float(best_auc),
            "relationship_direction": relationship_direction,
            "positive_mean": float(positive_mean),
            "negative_mean": float(negative_mean),
            "mean_difference": float(
                positive_mean - negative_mean
            ),
            "potential_leakage_flag": (
                "REVIEW"
                if best_auc >= 0.95
                else "OK"
            )
        }
    )

univariate_diagnostics_df = (
    pd.DataFrame(univariate_results)
    .sort_values(
        by="univariate_roc_auc",
        ascending=False
    )
    .reset_index(drop=True)
)

high_risk_feature_count = int(
    (
        univariate_diagnostics_df[
            "univariate_roc_auc"
        ] >= 0.95
    ).sum()
)

display(
    univariate_diagnostics_df.head(15)
)

print(
    f"Numeric features evaluated: "
    f"{len(univariate_diagnostics_df)}"
)
print(
    f"Features with univariate ROC-AUC >= 0.95: "
    f"{high_risk_feature_count}"
)
print("Test dataset used: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 47: Select the validation decision threshold

import pandas as pd
import numpy as np

from sklearn.metrics import (
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

precision_values, recall_values, threshold_values = (
    precision_recall_curve(
        y_validation,
        validation_probabilities
    )
)

threshold_results = []

for threshold in threshold_values:
    predictions = (
        validation_probabilities >= threshold
    ).astype(int)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    threshold_results.append(
        {
            "decision_threshold": float(threshold),
            "precision": float(
                precision_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive)
        }
    )

threshold_results_df = pd.DataFrame(threshold_results)

eligible_thresholds_df = (
    threshold_results_df
    .loc[
        threshold_results_df["recall"] >= 0.75
    ]
    .sort_values(
        by=[
            "f1_score",
            "precision",
            "decision_threshold"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
)

assert not eligible_thresholds_df.empty, (
    "No threshold satisfies the minimum recall requirement"
)

selected_threshold_metrics = (
    eligible_thresholds_df
    .iloc[0]
    .to_dict()
)

selected_threshold = float(
    selected_threshold_metrics["decision_threshold"]
)

top_thresholds_df = (
    eligible_thresholds_df
    .head(10)
    .reset_index(drop=True)
)

display(top_thresholds_df)

print(f"Selected threshold: {selected_threshold:.4f}")
print(
    f"Validation precision: "
    f"{selected_threshold_metrics['precision']:.4f}"
)
print(
    f"Validation recall: "
    f"{selected_threshold_metrics['recall']:.4f}"
)
print(
    f"Validation F1-score: "
    f"{selected_threshold_metrics['f1_score']:.4f}"
)
print(
    f"Validation false positives: "
    f"{int(selected_threshold_metrics['false_positive'])}"
)
print(
    f"Validation false negatives: "
    f"{int(selected_threshold_metrics['false_negative'])}"
)
print("Minimum recall requirement: 0.75")
print("Test dataset used: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 48: Train and validate a LightGBM classifier

import pandas as pd
import numpy as np

from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Define LightGBM preprocessing
lightgbm_numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ]
)

lightgbm_categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "one_hot_encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

lightgbm_preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            lightgbm_numeric_transformer,
            numeric_columns
        ),
        (
            "categorical",
            lightgbm_categorical_transformer,
            categorical_columns
        )
    ]
)

# Create the LightGBM pipeline
lightgbm_pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            lightgbm_preprocessor
        ),
        (
            "classifier",
            LGBMClassifier(
                objective="binary",
                n_estimators=300,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=40,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=0.5,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        )
    ]
)

# Train the model using the training dataset only
lightgbm_pipeline.fit(
    X_train,
    y_train
)

# Generate validation probabilities
lightgbm_validation_probabilities = (
    lightgbm_pipeline
    .predict_proba(X_validation)[:, 1]
)

lightgbm_decision_threshold = 0.50

lightgbm_validation_predictions = (
    lightgbm_validation_probabilities
    >= lightgbm_decision_threshold
).astype(int)

# Calculate validation metrics
lightgbm_confusion_matrix = confusion_matrix(
    y_validation,
    lightgbm_validation_predictions,
    labels=[0, 1]
)

true_negative, false_positive, false_negative, true_positive = (
    lightgbm_confusion_matrix.ravel()
)

lightgbm_validation_metrics = {
    "model_name": "LightGBM",
    "decision_threshold": lightgbm_decision_threshold,
    "pr_auc": float(
        average_precision_score(
            y_validation,
            lightgbm_validation_probabilities
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_validation,
            lightgbm_validation_probabilities
        )
    ),
    "precision": float(
        precision_score(
            y_validation,
            lightgbm_validation_predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_validation,
            lightgbm_validation_predictions,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_validation,
            lightgbm_validation_predictions,
            zero_division=0
        )
    ),
    "true_negative": int(true_negative),
    "false_positive": int(false_positive),
    "false_negative": int(false_negative),
    "true_positive": int(true_positive)
}

lightgbm_validation_metrics_df = pd.DataFrame(
    [lightgbm_validation_metrics]
)

# Validate the execution
assert len(X_train) == 12500, (
    f"Unexpected training row count: {len(X_train)}"
)

assert len(X_validation) == 2600, (
    f"Unexpected validation row count: {len(X_validation)}"
)

assert 0.0 <= lightgbm_validation_metrics["pr_auc"] <= 1.0, (
    "Invalid LightGBM validation PR-AUC"
)

assert 0.0 <= lightgbm_validation_metrics["roc_auc"] <= 1.0, (
    "Invalid LightGBM validation ROC-AUC"
)

display(lightgbm_validation_metrics_df)

print(f"Training rows: {len(X_train):,}")
print(f"Validation rows: {len(X_validation):,}")
print(f"Numeric features: {len(numeric_columns)}")
print(f"Categorical features: {len(categorical_columns)}")
print(
    f"Decision threshold: "
    f"{lightgbm_decision_threshold:.2f}"
)
print("Test dataset used: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 49: Select the LightGBM validation decision threshold

import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

threshold_results = []

for threshold in sorted(
    set(lightgbm_validation_probabilities)
):
    predictions = (
        lightgbm_validation_probabilities >= threshold
    ).astype(int)

    true_negative, false_positive, false_negative, true_positive = (
        confusion_matrix(
            y_validation,
            predictions,
            labels=[0, 1]
        ).ravel()
    )

    threshold_results.append(
        {
            "decision_threshold": float(threshold),
            "precision": float(
                precision_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "f1_score": float(
                f1_score(
                    y_validation,
                    predictions,
                    zero_division=0
                )
            ),
            "true_negative": int(true_negative),
            "false_positive": int(false_positive),
            "false_negative": int(false_negative),
            "true_positive": int(true_positive)
        }
    )

lightgbm_threshold_results_df = pd.DataFrame(
    threshold_results
)

eligible_lightgbm_thresholds_df = (
    lightgbm_threshold_results_df
    .loc[
        lightgbm_threshold_results_df["recall"] >= 0.75
    ]
    .sort_values(
        by=[
            "f1_score",
            "precision",
            "decision_threshold"
        ],
        ascending=[
            False,
            False,
            False
        ]
    )
)

assert not eligible_lightgbm_thresholds_df.empty, (
    "No LightGBM threshold satisfies the minimum recall requirement"
)

selected_lightgbm_threshold_metrics = (
    eligible_lightgbm_thresholds_df
    .iloc[0]
    .to_dict()
)

selected_lightgbm_threshold = float(
    selected_lightgbm_threshold_metrics[
        "decision_threshold"
    ]
)

display(
    eligible_lightgbm_thresholds_df
    .head(10)
    .reset_index(drop=True)
)

print(
    f"Selected LightGBM threshold: "
    f"{selected_lightgbm_threshold:.4f}"
)
print(
    f"Validation precision: "
    f"{selected_lightgbm_threshold_metrics['precision']:.4f}"
)
print(
    f"Validation recall: "
    f"{selected_lightgbm_threshold_metrics['recall']:.4f}"
)
print(
    f"Validation F1-score: "
    f"{selected_lightgbm_threshold_metrics['f1_score']:.4f}"
)
print(
    f"Validation false positives: "
    f"{int(selected_lightgbm_threshold_metrics['false_positive'])}"
)
print(
    f"Validation false negatives: "
    f"{int(selected_lightgbm_threshold_metrics['false_negative'])}"
)
print("Minimum recall requirement: 0.75")
print("Test dataset used: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 50: Perform the final LightGBM evaluation on the test dataset

import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score
)

FINAL_MODEL_NAME = "LightGBM"
FINAL_DECISION_THRESHOLD = 0.50

test_spark_df = (
    split_feature_df
    .filter(F.col("dataset_split") == "test")
)

test_pdf = (
    test_spark_df
    .select(
        *model_feature_columns,
        "failure_within_7d"
    )
    .toPandas()
)

X_test = test_pdf[model_feature_columns]
y_test = test_pdf["failure_within_7d"].astype(int)

test_probabilities = (
    lightgbm_pipeline
    .predict_proba(X_test)[:, 1]
)

test_predictions = (
    test_probabilities >= FINAL_DECISION_THRESHOLD
).astype(int)

test_confusion_matrix = confusion_matrix(
    y_test,
    test_predictions,
    labels=[0, 1]
)

true_negative, false_positive, false_negative, true_positive = (
    test_confusion_matrix.ravel()
)

test_metrics = {
    "model_name": FINAL_MODEL_NAME,
    "dataset": "test",
    "decision_threshold": FINAL_DECISION_THRESHOLD,
    "pr_auc": float(
        average_precision_score(
            y_test,
            test_probabilities
        )
    ),
    "roc_auc": float(
        roc_auc_score(
            y_test,
            test_probabilities
        )
    ),
    "precision": float(
        precision_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),
    "recall": float(
        recall_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),
    "f1_score": float(
        f1_score(
            y_test,
            test_predictions,
            zero_division=0
        )
    ),
    "true_negative": int(true_negative),
    "false_positive": int(false_positive),
    "false_negative": int(false_negative),
    "true_positive": int(true_positive)
}

test_metrics_df = pd.DataFrame([test_metrics])

assert len(test_pdf) == 2800, (
    f"Unexpected test row count: {len(test_pdf)}"
)

assert int(y_test.sum()) == 133, (
    f"Unexpected test positive count: {int(y_test.sum())}"
)

assert (
    true_negative
    + false_positive
    + false_negative
    + true_positive
    == len(test_pdf)
), "The confusion matrix does not match the test row count"

assert 0.0 <= test_metrics["pr_auc"] <= 1.0, (
    f"Invalid test PR-AUC: {test_metrics['pr_auc']}"
)

assert 0.0 <= test_metrics["roc_auc"] <= 1.0, (
    f"Invalid test ROC-AUC: {test_metrics['roc_auc']}"
)

display(test_metrics_df)

print(f"Final model: {FINAL_MODEL_NAME}")
print(f"Final decision threshold: {FINAL_DECISION_THRESHOLD:.2f}")
print(f"Test rows: {len(test_pdf):,}")
print(f"Test positive rows: {int(y_test.sum()):,}")
print(f"Test negative rows: {int((y_test == 0).sum()):,}")
print("Final test evaluation completed: True")
print("Further threshold tuning on the test dataset allowed: False")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }

# CELL ********************

# STEP 51: Log and register the final LightGBM model with MLflow

import mlflow
import mlflow.sklearn

from mlflow.models import infer_signature

REGISTERED_MODEL_NAME = "predictive_maintenance_lightgbm"
FINAL_DECISION_THRESHOLD = 0.50

model_signature = infer_signature(
    X_train,
    lightgbm_pipeline.predict_proba(X_train)[:, 1]
)

input_example = X_train.head(5)

with mlflow.start_run(
    run_name="lightgbm_champion_final"
) as final_run:

    mlflow.log_param(
        "model_type",
        "LightGBM"
    )
    mlflow.log_param(
        "decision_threshold",
        FINAL_DECISION_THRESHOLD
    )
    mlflow.log_param(
        "training_rows",
        len(X_train)
    )
    mlflow.log_param(
        "validation_rows",
        len(X_validation)
    )
    mlflow.log_param(
        "test_rows",
        len(X_test)
    )
    mlflow.log_param(
        "numeric_feature_count",
        len(numeric_columns)
    )
    mlflow.log_param(
        "categorical_feature_count",
        len(categorical_columns)
    )

    mlflow.log_metric(
        "validation_pr_auc",
        lightgbm_validation_metrics["pr_auc"]
    )
    mlflow.log_metric(
        "validation_roc_auc",
        lightgbm_validation_metrics["roc_auc"]
    )
    mlflow.log_metric(
        "validation_precision",
        lightgbm_validation_metrics["precision"]
    )
    mlflow.log_metric(
        "validation_recall",
        lightgbm_validation_metrics["recall"]
    )
    mlflow.log_metric(
        "validation_f1_score",
        lightgbm_validation_metrics["f1_score"]
    )

    mlflow.log_metric(
        "test_pr_auc",
        test_metrics["pr_auc"]
    )
    mlflow.log_metric(
        "test_roc_auc",
        test_metrics["roc_auc"]
    )
    mlflow.log_metric(
        "test_precision",
        test_metrics["precision"]
    )
    mlflow.log_metric(
        "test_recall",
        test_metrics["recall"]
    )
    mlflow.log_metric(
        "test_f1_score",
        test_metrics["f1_score"]
    )
    mlflow.log_metric(
        "test_true_negative",
        test_metrics["true_negative"]
    )
    mlflow.log_metric(
        "test_false_positive",
        test_metrics["false_positive"]
    )
    mlflow.log_metric(
        "test_false_negative",
        test_metrics["false_negative"]
    )
    mlflow.log_metric(
        "test_true_positive",
        test_metrics["true_positive"]
    )

    model_info = mlflow.sklearn.log_model(
        sk_model=lightgbm_pipeline,
        artifact_path="model",
        signature=model_signature,
        input_example=input_example,
        registered_model_name=REGISTERED_MODEL_NAME
    )

    final_run_id = final_run.info.run_id

print(f"Registered model: {REGISTERED_MODEL_NAME}")
print(f"MLflow run ID: {final_run_id}")
print(f"Model URI: {model_info.model_uri}")
print(f"Decision threshold: {FINAL_DECISION_THRESHOLD:.2f}")
print("Final champion model registered successfully.")

# METADATA ********************

# META {
# META   "language": "python",
# META   "language_group": "synapse_pyspark"
# META }
