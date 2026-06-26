# Project Report: Spatio-Temporal Traffic Risk Engine

## 1. Introduction

This project builds a California-focused traffic risk prediction system using machine learning, spatial feature engineering, and an interactive dashboard.  
The goal is to estimate accident probability from environmental and contextual conditions rather than only describing where accidents happened historically.

The final system combines:

- data cleaning and preparation
- synthetic negative-sample generation
- H3-based spatial feature engineering
- LightGBM classification
- SHAP-based explainability
- a Streamlit dashboard for scenario exploration

## 2. Problem Statement

Accident datasets are often used only for historical summaries.  
This project reframes the problem as a predictive one:

> Given a location, time, and weather context, how likely is an accident-related scenario compared with safer scenarios?

This is useful for:

- academic modeling
- exploratory scenario analysis
- explainable dashboard communication

## 3. Dataset and Scope

The original source is the US Accidents dataset.  
The project narrows the analysis to California in order to:

- keep the geographic scope meaningful
- improve interpretability
- align the dashboard with a single deployment region

The cleaned California dataset retains only the features needed for modeling:

- `Start_Time`
- `Start_Lat`
- `Start_Lng`
- `Temperature(F)`
- `Humidity(%)`
- `Visibility(mi)`
- `Wind_Speed(mph)`
- `Weather_Condition`
- `Sunrise_Sunset`
- road-context columns retained for model compatibility

## 4. Data Cleaning

The cleaning pipeline performs the following steps:

1. Filter the national dataset to California only.
2. Keep the essential columns for spatial, temporal, and weather-based modeling.
3. Drop rows with missing values in required fields.
4. Parse `Start_Time` into datetime format.
5. Drop rows where timestamp parsing fails.
6. Save a cleaned California dataset for reproducibility.

This produces a dataset suitable for both model training and downstream dashboard support.

## 5. Modeling Strategy

### 5.1 Binary Classification Setup

The original data contains accident observations only.  
To create a binary classification problem:

- real accident rows are labeled as `target = 1`
- synthetic safe rows are generated and labeled as `target = 0`

### 5.2 Synthetic Safe Scenario Generation

Safe scenarios are created through permutation sampling:

- `Start_Time` is shuffled
- `Start_Lat` is shuffled
- `Start_Lng` is shuffled

This reduces the chance that the model simply memorizes accident density and instead encourages learning from environmental and contextual drivers.

### 5.3 Temporal Validation

To avoid optimistic evaluation:

- the modeling table is sorted by `Start_Time`
- the first 80% is used for training
- the final 20% is used for testing

This creates an out-of-time holdout that better simulates future conditions than a random split.

## 6. Feature Engineering

### 6.1 Spatial Engineering with H3

The project uses Uber H3 hexagonal indexing at resolution 7.

For each location:

- latitude and longitude are converted into an H3 cell
- neighboring H3 cells are used to compute a local historical risk score
- that score is log-transformed into `neighbor_risk`

This captures the idea of accident “contagion” in risky neighborhoods.

### 6.2 Temporal Engineering

The hour of day is encoded cyclically using:

- `hour_sin`
- `hour_cos`

This is better than using raw hour values because time is circular rather than linear.

### 6.3 Interaction Feature

The project adds:

- `wet_rush_hour = (Humidity / 100) * hour_sin`

This feature attempts to capture the interaction between moisture conditions and traffic-time effects.

## 7. Model Choice

The final model is a LightGBM classifier.

Reasons for this choice:

- strong performance on large tabular datasets
- efficient training
- ability to model non-linear interactions
- support for categorical features
- suitability for deployment in a lightweight dashboard

## 8. Results

The temporal holdout evaluation produced:

- AUC: `0.9085`
- Accuracy: `0.85`
- Positive-class Recall: `0.91`
- Positive-class Precision: `0.81`

These results suggest the model can separate higher-risk and lower-risk scenarios reasonably well under future-style validation.

## 9. Explainability

The project uses SHAP to inspect feature influence.

The explainability analysis shows:

- `neighbor_risk` is the strongest feature
- time-related features such as `hour_sin`, `hour_cos`, and `Sunrise_Sunset` are highly influential
- environmental features such as humidity, temperature, and weather conditions also contribute meaningfully

This supports the claim that the model is using interpretable spatio-temporal signals rather than purely memorizing coordinates.

## 10. Dashboard

The Streamlit dashboard has two main purposes:

1. predict accident probability for a selected scenario
2. explain the result through visuals and validation evidence

Main dashboard sections:

- Accident Probability
- Analytics & Visuals
- Model Input Audit
- Feature Schema

The dashboard is designed primarily for:

- academic demonstration
- exploratory analysis
- report support

It is not intended as a production traffic management system.

## 11. Deployment Design

For deployment, the dashboard does not scan the full raw dataset at runtime.  
Instead, it uses lightweight summary artifacts such as:

- `weather_frequency.csv`
- `dashboard_summary.json`

This makes the app easier to host and faster to load in environments like Streamlit Community Cloud.

## 12. Limitations

This project has several important limitations:

- synthetic safe scenarios may still create unrealistic combinations
- some scenario rankings are better interpreted analytically than operationally
- the deployed app emphasizes communication and interactivity over full raw-data querying

## 13. Future Improvements

Possible next steps include:

- improving the realism of synthetic safe scenario generation
- calibrating predicted probabilities
- adding more advanced SHAP visuals inside the dashboard
- expanding documentation and deployment automation

## 14. Conclusion

This project demonstrates an end-to-end traffic risk engine that combines:

- spatial modeling
- temporal reasoning
- machine learning
- explainability
- dashboard communication

Its strongest contribution is the way it links modeling rigor with presentation and interpretability.  
Rather than being only a notebook experiment, it becomes a complete analytical system that can be explained, demonstrated, and deployed.

