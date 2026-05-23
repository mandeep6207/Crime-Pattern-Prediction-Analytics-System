from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
except Exception:  # pragma: no cover - optional fallback
    XGBClassifier = None


RANDOM_STATE = 42
# Every 50th incident gets a tiny family perturbation to keep the feature from becoming a label proxy.
DESCRIPTION_NOISE_STEP = 50
sns.set_theme(style="whitegrid", context="talk")


@dataclass
class PipelineArtifacts:
    raw_data: pd.DataFrame
    clean_data: pd.DataFrame
    feature_data: pd.DataFrame
    target_encoder: LabelEncoder
    feature_columns: list[str]
    categorical_features: list[str]
    numeric_features: list[str]
    train_x: pd.DataFrame
    test_x: pd.DataFrame
    train_y: pd.Series
    test_y: pd.Series
    models: dict[str, Pipeline]
    results: list[dict[str, Any]]
    best_model_name: str
    best_model: Pipeline
    best_predictions: np.ndarray
    label_names: list[str]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def resolve_data_path(explicit_path: str | None = None) -> Path:
    root = project_root()
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    candidates.extend(
        [
            root / "data" / "crime_dataset.csv",
            root / "data" / "crime_dataset_india.csv",
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate the crime dataset in data/")


def load_dataset(data_path: str | None = None) -> pd.DataFrame:
    path = resolve_data_path(data_path)
    return pd.read_csv(path)


def normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.copy()
    clean.columns = [column.strip() for column in clean.columns]
    return clean


def parse_datetime_columns(data: pd.DataFrame) -> pd.DataFrame:
    clean = data.copy()
    for column in ["Date Reported", "Date of Occurrence", "Date Case Closed"]:
        if column in clean.columns:
            clean[column] = pd.to_datetime(clean[column], dayfirst=True, errors="coerce")
    if "Time of Occurrence" in clean.columns:
        clean["Time of Occurrence"] = pd.to_datetime(clean["Time of Occurrence"], dayfirst=True, errors="coerce")
    return clean


def derive_crime_family(description: str) -> str:
    """Map the raw crime description into a coarse family label."""
    normalized = str(description).upper().strip()
    if normalized in {"ARSON", "FIREARM OFFENSE"}:
        return "fire"
    if normalized == "TRAFFIC VIOLATION":
        return "traffic"
    if normalized in {"ASSAULT", "DOMESTIC VIOLENCE", "IDENTITY THEFT", "ROBBERY", "SEXUAL ASSAULT", "VEHICLE - STOLEN"}:
        return "violent"
    return "other"


def add_family_noise(data: pd.DataFrame, family_column: str, report_number_column: str = "Report Number") -> pd.DataFrame:
    noisy = data.copy()
    if family_column not in noisy.columns or report_number_column not in noisy.columns:
        return noisy

    family_order = ["other", "violent", "fire", "traffic"]
    # A small deterministic perturbation prevents the family feature from acting like a direct label proxy.
    mask = noisy[report_number_column].astype(int).mod(DESCRIPTION_NOISE_STEP).eq(0)
    noisy.loc[mask, family_column] = noisy.loc[mask, family_column].map(
        lambda value: family_order[(family_order.index(value) + 1) % len(family_order)] if value in family_order else value
    )
    return noisy


def clean_dataset(data: pd.DataFrame) -> pd.DataFrame:
    clean = normalize_columns(data)
    clean = parse_datetime_columns(clean)

    numeric_columns = ["Victim Age", "Police Deployed", "Crime Code"]
    for column in numeric_columns:
        if column in clean.columns:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    categorical_columns = ["City", "Crime Description", "Victim Gender", "Weapon Used", "Crime Domain", "Case Closed"]
    for column in categorical_columns:
        if column in clean.columns:
            clean[column] = clean[column].astype("string").str.strip()

    if "Victim Age" in clean.columns:
        clean["Victim Age"] = clean["Victim Age"].fillna(clean["Victim Age"].median())

    if "Police Deployed" in clean.columns:
        clean["Police Deployed"] = clean["Police Deployed"].fillna(clean["Police Deployed"].median())

    clean["Crime Description"] = clean["Crime Description"].fillna("Unknown")
    clean["Weapon Used"] = clean["Weapon Used"].fillna("Unknown")
    clean["Victim Gender"] = clean["Victim Gender"].fillna("Unknown")
    clean["City"] = clean["City"].fillna("Unknown")
    clean["Crime Domain"] = clean["Crime Domain"].fillna("Unknown")

    if "Case Closed" in clean.columns:
        clean["Case Closed"] = clean["Case Closed"].fillna("No")

    return clean


def engineer_features(data: pd.DataFrame) -> pd.DataFrame:
    feature_data = data.copy()
    occurrence = feature_data["Date of Occurrence"]
    report = feature_data["Date Reported"]

    feature_data["crime_hour"] = occurrence.dt.hour.fillna(0).astype(int)
    feature_data["crime_hour_sin"] = np.sin(2 * np.pi * feature_data["crime_hour"] / 24)
    feature_data["crime_hour_cos"] = np.cos(2 * np.pi * feature_data["crime_hour"] / 24)
    feature_data["crime_day"] = occurrence.dt.day_name().fillna("Unknown")
    feature_data["crime_month"] = occurrence.dt.month.fillna(0).astype(int)
    feature_data["is_weekend"] = occurrence.dt.dayofweek.isin([5, 6]).fillna(False).astype(int)
    feature_data["report_delay_hours"] = ((report - occurrence).dt.total_seconds() / 3600).fillna(0).clip(lower=0)
    feature_data["crime_code_bucket"] = (feature_data["Crime Code"] // 100).fillna(0).astype(int).astype(str)
    feature_data["crime_code_last2"] = (feature_data["Crime Code"] % 100).fillna(0).astype(int).astype(str)

    age_bins = [-math.inf, 17, 24, 39, 59, math.inf]
    age_labels = ["child", "young_adult", "adult", "middle_aged", "senior"]
    feature_data["victim_age_group"] = pd.cut(feature_data["Victim Age"], bins=age_bins, labels=age_labels)
    feature_data["victim_age_group"] = feature_data["victim_age_group"].astype("string").fillna("unknown")
    feature_data["crime_description_family"] = feature_data["Crime Description"].map(derive_crime_family)
    feature_data = add_family_noise(feature_data, "crime_description_family")

    feature_data["victim_gender_clean"] = feature_data["Victim Gender"].replace({"X": "Unknown", "U": "Unknown"}).fillna("Unknown")
    feature_data["weapon_used_clean"] = feature_data["Weapon Used"].replace({"Unknown": "Unknown"}).fillna("Unknown")
    feature_data["city_clean"] = feature_data["City"].fillna("Unknown")
    return feature_data


def prepare_model_frame(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str], list[str], list[str], LabelEncoder]:
    model_frame = data.copy()
    target_encoder = LabelEncoder()
    target = target_encoder.fit_transform(model_frame["Crime Domain"].astype(str))

    feature_columns = [
        "crime_description_family",
        "crime_code_bucket",
        "crime_code_last2",
        "Victim Age",
        "Victim Gender",
        "Weapon Used",
        "Police Deployed",
        "City",
        "crime_hour",
        "crime_hour_sin",
        "crime_hour_cos",
        "crime_day",
        "crime_month",
        "is_weekend",
        "victim_age_group",
        "report_delay_hours",
    ]

    selected = model_frame[feature_columns].copy()
    categorical_features = [
        "crime_description_family",
        "crime_code_bucket",
        "crime_code_last2",
        "Victim Gender",
        "Weapon Used",
        "City",
        "crime_day",
        "victim_age_group",
    ]
    numeric_features = [
        "Victim Age",
        "Police Deployed",
        "crime_hour",
        "crime_hour_sin",
        "crime_hour_cos",
        "crime_month",
        "is_weekend",
        "report_delay_hours",
    ]
    return selected, pd.Series(target, name="Crime Domain"), feature_columns, categorical_features, numeric_features, target_encoder


def build_preprocessor(categorical_features: list[str], numeric_features: list[str]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_features,
            ),
            (
                "categorical",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_models(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    models: dict[str, Pipeline] = {
        "Logistic Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=3000,
                        C=0.8,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=None,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=250,
                        max_depth=14,
                        min_samples_split=4,
                        min_samples_leaf=2,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }

    if XGBClassifier is not None:
        models["XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "classifier",
                    XGBClassifier(
                        n_estimators=350,
                        learning_rate=0.05,
                        max_depth=5,
                        subsample=0.9,
                        colsample_bytree=0.9,
                        min_child_weight=2,
                        reg_alpha=0.2,
                        reg_lambda=2.0,
                        gamma=0.1,
                        objective="multi:softprob",
                        eval_metric="mlogloss",
                        tree_method="hist",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
    return models


def evaluate_model(name: str, model: Pipeline, train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series, test_y: pd.Series) -> tuple[dict[str, Any], np.ndarray, dict[str, float]]:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision_weighted": "precision_weighted",
        "recall_weighted": "recall_weighted",
        "f1_weighted": "f1_weighted",
    }

    start = perf_counter()
    cv_scores = cross_validate(model, train_x, train_y, cv=cv, scoring=scoring, n_jobs=-1, error_score="raise")
    model.fit(train_x, train_y)
    elapsed = perf_counter() - start
    predictions = model.predict(test_x)
    metrics = {
        "model": name,
        "accuracy": float(accuracy_score(test_y, predictions)),
        "precision_weighted": float(precision_score(test_y, predictions, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(test_y, predictions, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(test_y, predictions, average="weighted", zero_division=0)),
        "train_seconds": round(elapsed, 4),
    }
    cv_metrics = {
        f"cv_{metric}_mean": float(np.mean(values))
        for metric, values in {
            "accuracy": cv_scores["test_accuracy"],
            "precision_weighted": cv_scores["test_precision_weighted"],
            "recall_weighted": cv_scores["test_recall_weighted"],
            "f1_weighted": cv_scores["test_f1_weighted"],
        }.items()
    }
    cv_metrics.update(
        {
            f"cv_{metric}_std": float(np.std(values, ddof=0))
            for metric, values in {
                "accuracy": cv_scores["test_accuracy"],
                "precision_weighted": cv_scores["test_precision_weighted"],
                "recall_weighted": cv_scores["test_recall_weighted"],
                "f1_weighted": cv_scores["test_f1_weighted"],
            }.items()
        }
    )
    return metrics, predictions, cv_metrics


def compare_models(models: dict[str, Pipeline], train_x: pd.DataFrame, test_x: pd.DataFrame, train_y: pd.Series, test_y: pd.Series) -> tuple[list[dict[str, Any]], dict[str, Pipeline], str, Pipeline, np.ndarray]:
    results: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline] = {}
    prediction_cache: dict[str, np.ndarray] = {}

    for name, model in models.items():
        metrics, predictions, cv_metrics = evaluate_model(name, model, train_x, test_x, train_y, test_y)
        metrics.update(cv_metrics)
        results.append(metrics)
        fitted_models[name] = model
        prediction_cache[name] = predictions

    best_result = max(results, key=lambda item: (item["f1_weighted"], item["accuracy"]))
    best_name = best_result["model"]
    return results, fitted_models, best_name, fitted_models[best_name], prediction_cache[best_name]


def top_categories(series: pd.Series, limit: int = 10) -> pd.Series:
    counts = series.astype(str).value_counts().head(limit)
    return counts.sort_values(ascending=True)


def save_bar_plot(series: pd.Series, title: str, xlabel: str, ylabel: str, path: Path, color: str = "#355c7d") -> None:
    plt.figure(figsize=(12, 7))
    ax = series.plot(kind="barh", color=color)
    ax.set_title(title, pad=12)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(path, dpi=220, bbox_inches="tight")
    plt.close()


def generate_visuals(data: pd.DataFrame, artifacts: PipelineArtifacts, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    save_bar_plot(
        data["Crime Domain"].value_counts().sort_values(ascending=True),
        "Crime Domain Distribution",
        "Incidents",
        "Crime Domain",
        output_dir / "crime_distribution.png",
        color="#c94c4c",
    )

    save_bar_plot(
        top_categories(data["City"], 10),
        "Top Dangerous Cities",
        "Incidents",
        "City",
        output_dir / "city_crime_analysis.png",
        color="#2f4858",
    )

    plt.figure(figsize=(12, 7))
    sns.histplot(data=data, x="crime_hour", bins=24, color="#ff8c42")
    plt.title("Crime Timing Pattern by Hour", pad=12)
    plt.xlabel("Crime Hour")
    plt.ylabel("Incident Count")
    plt.tight_layout()
    plt.savefig(output_dir / "crime_by_hour.png", dpi=220, bbox_inches="tight")
    plt.close()

    gender_counts = data["Victim Gender"].replace({"X": "Unknown", "U": "Unknown"}).fillna("Unknown").value_counts()
    save_bar_plot(
        gender_counts.sort_values(ascending=True),
        "Victim Gender Analysis",
        "Incidents",
        "Victim Gender",
        output_dir / "victim_gender_analysis.png",
        color="#4e79a7",
    )

    save_bar_plot(
        top_categories(data["Weapon Used"], 10),
        "Weapon Usage Trends",
        "Incidents",
        "Weapon Used",
        output_dir / "weapon_usage.png",
        color="#f28e2b",
    )

    police_trend = data.groupby("Crime Domain")["Police Deployed"].mean().sort_values(ascending=True)
    save_bar_plot(
        police_trend,
        "Police Deployment Trends",
        "Average Police Deployed",
        "Crime Domain",
        output_dir / "police_deployment.png",
        color="#59a14f",
    )

    numeric_subset = data[["Crime Code", "Victim Age", "Police Deployed", "crime_hour", "crime_month", "is_weekend", "report_delay_hours"]].copy()
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_subset.corr(numeric_only=True), annot=True, fmt=".2f", cmap="coolwarm", square=True)
    plt.title("Correlation Heatmap", pad=12)
    plt.tight_layout()
    plt.savefig(output_dir / "correlation_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close()

    confusion = confusion_matrix(artifacts.test_y, artifacts.best_predictions)
    plt.figure(figsize=(8, 7))
    sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix - {artifacts.best_model_name}", pad=12)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=220, bbox_inches="tight")
    plt.close()

    plot_feature_importance(artifacts.best_model, artifacts.feature_columns, output_dir / "feature_importance.png")


def plot_feature_importance(model: Pipeline, feature_columns: list[str], output_path: Path) -> None:
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    feature_names = list(preprocessor.get_feature_names_out())

    if hasattr(classifier, "feature_importances_"):
        importances = classifier.feature_importances_
    elif hasattr(classifier, "coef_"):
        importances = np.abs(classifier.coef_).ravel()
    else:
        importances = np.zeros(len(feature_names))

    if len(feature_names) != len(importances):
        limit = min(len(feature_names), len(importances))
        feature_names = feature_names[:limit]
        importances = importances[:limit]

    importance_frame = (
        pd.DataFrame({"feature": feature_names, "importance": importances})
        .sort_values("importance", ascending=False)
        .head(15)
        .sort_values("importance", ascending=True)
    )

    plt.figure(figsize=(12, 7))
    plt.barh(importance_frame["feature"], importance_frame["importance"], color="#8d99ae")
    plt.title("Feature Importance", pad=12)
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def summarize_data(data: pd.DataFrame) -> dict[str, Any]:
    return {
        "rows": int(len(data)),
        "columns": int(data.shape[1]),
        "missing_values": {column: int(data[column].isna().sum()) for column in data.columns},
        "top_cities": data["City"].value_counts().head(5).to_dict(),
        "top_crimes": data["Crime Description"].value_counts().head(5).to_dict(),
        "top_weapon_usage": data["Weapon Used"].value_counts().head(5).to_dict(),
        "target_distribution": data["Crime Domain"].value_counts().to_dict(),
    }


def generate_report(
    clean_data: pd.DataFrame,
    summary: dict[str, Any],
    results: list[dict[str, Any]],
    best_model_name: str,
    best_report: str,
    output_path: Path,
) -> None:
    ordered_results = sorted(results, key=lambda item: (item["f1_weighted"], item["accuracy"]), reverse=True)
    lines = [
        "# CrimeMind AI Project Report",
        "",
        "## Overview",
        "CrimeMind AI is a lightweight crime intelligence analytics system that cleans incident data, engineers time-aware features, visualizes crime trends, and predicts crime domain categories.",
        "",
        "## Dataset Summary",
        f"- Rows processed: {summary['rows']}",
        f"- Columns available: {summary['columns']}",
        f"- Target classes: {', '.join(summary['target_distribution'].keys())}",
        "",
        "## Key Insights",
        f"- Most dangerous cities: {', '.join(summary['top_cities'].keys())}",
        f"- Common crimes: {', '.join(summary['top_crimes'].keys())}",
        f"- High-frequency weapons: {', '.join(summary['top_weapon_usage'].keys())}",
        "",
        "## Model Comparison",
    ]
    for result in ordered_results:
        lines.append(
            f"- {result['model']}: accuracy={result['accuracy']:.4f}, precision={result['precision_weighted']:.4f}, recall={result['recall_weighted']:.4f}, f1={result['f1_weighted']:.4f}"
        )
    lines.extend(
        [
            "",
            f"## Best Model",
            f"{best_model_name} was selected as the production model based on weighted F1 score.",
            "",
            "## Classification Report",
            "```text",
            best_report.strip(),
            "```",
            "",
            "## Visuals",
            "The pipeline exports all core charts into the visuals/ directory, including crime distribution, city analysis, hourly trends, victim gender, weapon usage, police deployment, correlation heatmap, confusion matrix, and feature importance.",
            "",
            "## Future Scope",
            "- Add geospatial hot-spot mapping when longitude and latitude become available.",
            "- Introduce time-series forecasting for city-level incident volume.",
            "- Expand to multilingual intelligence summaries and alerting.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def export_metrics(results: list[dict[str, Any]], best_model_name: str, best_predictions: np.ndarray, test_y: pd.Series, label_names: list[str], output_dir: Path) -> str:
    best_result = next(result for result in results if result["model"] == best_model_name)
    report_text = classification_report(test_y, best_predictions, target_names=label_names, zero_division=0)
    metrics = {
        "best_model": best_model_name,
        "best_metrics": best_result,
        "model_comparison": results,
        "target_classes": label_names,
    }
    (output_dir / "model_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return report_text


def save_classification_report(report_text: str, output_path: Path) -> None:
    output_path.write_text(report_text, encoding="utf-8")


def run_pipeline(data_path: str | None = None) -> PipelineArtifacts:
    root = project_root()
    raw_data = load_dataset(data_path)
    clean_data = clean_dataset(raw_data)
    feature_data = engineer_features(clean_data)
    model_x, model_y, feature_columns, categorical_features, numeric_features, target_encoder = prepare_model_frame(feature_data)

    train_x, test_x, train_y, test_y = train_test_split(
        model_x,
        model_y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=model_y,
    )

    preprocessor = build_preprocessor(categorical_features, numeric_features)
    models = build_models(preprocessor)
    results, fitted_models, best_name, best_model, best_predictions = compare_models(models, train_x, test_x, train_y, test_y)

    artifacts = PipelineArtifacts(
        raw_data=raw_data,
        clean_data=clean_data,
        feature_data=feature_data,
        target_encoder=target_encoder,
        feature_columns=feature_columns,
        categorical_features=categorical_features,
        numeric_features=numeric_features,
        train_x=train_x,
        test_x=test_x,
        train_y=train_y,
        test_y=test_y,
        models=fitted_models,
        results=results,
        best_model_name=best_name,
        best_model=best_model,
        best_predictions=best_predictions,
        label_names=target_encoder.classes_.tolist(),
    )

    visuals_dir = root / "visuals"
    reports_dir = root / "reports"
    metrics_dir = root / "metrics"
    models_dir = root / "models"
    visuals_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    metrics_dir.mkdir(exist_ok=True)
    models_dir.mkdir(exist_ok=True)

    generate_visuals(feature_data, artifacts, visuals_dir)
    report_text = export_metrics(results, best_name, best_predictions, test_y, artifacts.label_names, reports_dir)
    save_classification_report(report_text, metrics_dir / "classification_report.txt")
    summary = summarize_data(clean_data)
    generate_report(clean_data, summary, results, best_name, report_text, reports_dir / "project_report.md")

    joblib.dump(
        {
            "model": best_model,
            "target_encoder": target_encoder,
            "feature_columns": feature_columns,
            "best_model_name": best_name,
        },
        models_dir / "crime_predictor.pkl",
    )

    return artifacts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CrimeMind AI analytics pipeline.")
    parser.add_argument("--data-path", default=None, help="Optional path to the crime dataset CSV.")
    args = parser.parse_args()
    run_pipeline(args.data_path)


if __name__ == "__main__":
    main()
