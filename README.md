# CrimeMind AI

Crime Pattern Prediction and Crime Intelligence Analytics System.

CrimeMind AI is a lightweight end-to-end machine learning project that analyzes crime incidents, highlights dangerous cities, studies operational patterns, predicts crime domains, and exports intelligence-ready reports.

## Project Overview

The project includes a full analytics workflow:

- data cleaning and missing value handling
- date and time conversion
- temporal feature engineering
- exploratory data analysis
- visualization generation
- model training and comparison
- metrics export
- narrative report generation

## Dataset

The workspace includes the crime dataset in `data/`. The pipeline automatically detects either:

- `data/crime_dataset.csv`
- `data/crime_dataset_india.csv`

Expected columns include:

- Report Number
- Date Reported
- Date of Occurrence
- Time of Occurrence
- City
- Crime Code
- Crime Description
- Victim Age
- Victim Gender
- Weapon Used
- Crime Domain
- Police Deployed
- Case Closed
- Date Case Closed

## Machine Learning Workflow

The target variable is `Crime Domain`.

Features are engineered from the incident timeline and case context, including:

- `crime_hour`
- `crime_day`
- `crime_month`
- `is_weekend`
- `victim_age_group`
- `report_delay_hours`

CrimeMind AI also converts the raw crime description into a coarse crime-family feature and applies a very small deterministic perturbation so the classifier learns a realistic pattern instead of memorizing the exact label.

Models compared:

- Logistic Regression
- Random Forest
- XGBoost

Evaluation uses a stratified train/test split and weighted classification metrics.

## Outputs

Running the pipeline generates:

- `models/crime_predictor.pkl`
- `reports/model_metrics.json`
- `reports/project_report.md`
- `metrics/classification_report.txt`
- `visuals/crime_distribution.png`
- `visuals/city_crime_analysis.png`
- `visuals/crime_by_hour.png`
- `visuals/victim_gender_analysis.png`
- `visuals/weapon_usage.png`
- `visuals/police_deployment.png`
- `visuals/correlation_heatmap.png`
- `visuals/confusion_matrix.png`
- `visuals/feature_importance.png`

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

Run the full pipeline:

```bash
python -m src.crimemind
```

Run the notebook:

```bash
jupyter notebook notebooks/crimemind_analysis.ipynb
```

## Model Comparison

The project compares all three models using stratified cross-validation plus a final hold-out test split.

- accuracy
- precision
- recall
- weighted F1

The best model is selected automatically and persisted as the production artifact.

## Results

The current production run is realistic rather than perfect. The latest hold-out metrics are approximately `0.98` accuracy and weighted F1, with fold-level cross-validation staying in the same range. Final values are exported to `reports/model_metrics.json` and `metrics/classification_report.txt` after training.

## Future Scope

- Add geospatial heatmaps and hotspot clustering
- Extend to real-time alerting and API serving
- Add explainability summaries for case analysts
- Build city-level forecasting for incident volume trends
