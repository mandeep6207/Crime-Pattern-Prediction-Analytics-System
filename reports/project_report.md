# CrimeMind AI Project Report

## Overview
CrimeMind AI is a lightweight crime intelligence analytics system that cleans incident data, engineers time-aware features, visualizes crime trends, and predicts crime domain categories.

## Dataset Summary
- Rows processed: 40160
- Columns available: 14
- Target classes: Other Crime, Violent Crime, Fire Accident, Traffic Fatality

## Key Insights
- Most dangerous cities: Delhi, Mumbai, Bangalore, Hyderabad, Kolkata
- Common crimes: BURGLARY, VANDALISM, FRAUD, DOMESTIC VIOLENCE, FIREARM OFFENSE
- High-frequency weapons: Knife, Unknown, Explosives, Blunt Object, Poison

## Model Comparison
- Random Forest: accuracy=0.9798, precision=0.9803, recall=0.9798, f1=0.9799
- XGBoost: accuracy=0.9798, precision=0.9803, recall=0.9798, f1=0.9799
- Logistic Regression: accuracy=0.9796, precision=0.9801, recall=0.9796, f1=0.9797

Each model was also evaluated with 5-fold stratified cross-validation on the training partition, and the fold means stayed aligned with the hold-out results.

## Best Model
Random Forest was selected as the production model based on weighted F1 score.

The final hold-out metrics landed at roughly 0.98 accuracy, and the 5-fold training scores stayed within the same band.

## Classification Report
```text
precision    recall  f1-score   support

   Fire Accident       0.95      0.98      0.97       765
     Other Crime       1.00      0.98      0.99      4590
Traffic Fatality       0.96      0.98      0.97       383
   Violent Crime       0.96      0.98      0.97      2294

        accuracy                           0.98      8032
       macro avg       0.97      0.98      0.97      8032
    weighted avg       0.98      0.98      0.98      8032
```

## Visuals
The pipeline exports all core charts into the visuals/ directory, including crime distribution, city analysis, hourly trends, victim gender, weapon usage, police deployment, correlation heatmap, confusion matrix, and feature importance.

## Future Scope
- Add geospatial hot-spot mapping when longitude and latitude become available.
- Introduce time-series forecasting for city-level incident volume.
- Expand to multilingual intelligence summaries and alerting.