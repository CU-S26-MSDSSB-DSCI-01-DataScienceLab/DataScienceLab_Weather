# Spatio-Temporal Traffic Risk Engine

California traffic accident risk prediction engine built with Python, LightGBM, Uber H3, SHAP, and Streamlit.
dataset :https://www.kaggle.com/datasets/sobhanmoosavi/us-accidents?utm_source=chatgpt.com
## Project Overview

This project predicts accident risk across California using spatio-temporal feature engineering on the US Accidents dataset.  
The core idea is to move beyond simple accident frequency counting and instead estimate probability from:

- spatial neighborhood history
- time-of-day patterns
- weather conditions
- environmental interactions

The final system includes:

- a data-cleaning and training pipeline
- a LightGBM binary classification model
- SHAP-based explainability
- a Streamlit dashboard for interactive scenario analysis

## Key Highlights

- Filtered the national accident dataset to California for a focused regional analysis
- Built synthetic "safe" scenarios using permutation sampling to create a binary classification setup
- Engineered H3 hexagonal spatial features to capture neighborhood accident contagion
- Added cyclical temporal encoding with `hour_sin` and `hour_cos`
- Created an interaction feature: `wet_rush_hour`
- Used an out-of-time temporal split instead of a random split to reduce leakage
- Added SHAP analysis to interpret model behavior
- Delivered a Streamlit dashboard for live probability scoring and analytical visuals

## Model Performance

Temporal holdout validation from the training workflow:

- AUC: `0.9085`
- Accuracy: `0.85`
- Positive-class Recall: `0.91`
- Positive-class Precision: `0.81`

These results come from a future-style evaluation where the model is trained on earlier records and tested on later ones.

## Tech Stack

- Python
- pandas
- numpy
- matplotlib
- seaborn
- h3
- LightGBM
- scikit-learn
- SHAP
- Streamlit
- Altair
- PyDeck

## Repository Structure

```text
.
├── app.py
├── traffic_risk_engine.py
├── requirements.txt
├── data/
│   ├── US_Accidents.csv
│   └── california_accidents_cleaned.csv
├── traffic_risk_model.pkl
├── hex_risk_dict.pkl
├── feature_order.json
├── weather_categories.json
├── weather_frequency.csv
├── dashboard_summary.json
├── performance_curves.png
├── confusion_matrix.png
└── cyclical_risk.png
```

## Important Files

- [app.py] 
  Streamlit dashboard for risk prediction and analytics

- [traffic_risk_engine.py]
  End-to-end training and artifact-generation script

- [traffic_risk_model.pkl]
  Trained LightGBM model used by the dashboard

- [hex_risk_dict.pkl] 
  Historical H3 neighborhood-risk dictionary

- [weather_frequency.csv]  
  Lightweight deployment artifact for dashboard analytics

- [dashboard_summary.json]  
  Lightweight deployment artifact for summary cards

## How the Pipeline Works

### 1. Data Filtering and Cleaning

- Load the US Accidents dataset
- Filter rows where `State == "CA"`
- Keep only relevant features for time, location, weather, and road context
- Parse `Start_Time`
- Drop rows with missing values in required columns
- Save a cleaned California-only dataset

### 2. Synthetic Safe Scenario Creation

- Real accident rows are labeled as `target = 1`
- Synthetic safe rows are created by permuting:
  - `Start_Time`
  - `Start_Lat`
  - `Start_Lng`
- This creates a binary classification problem while reducing naive spatial-frequency learning

### 3. Feature Engineering

- Convert coordinates to H3 resolution-7 cells
- Compute `neighbor_risk` from surrounding H3 cells
- Apply `log1p` to neighborhood risk
- Encode hour cyclically:
  - `hour_sin`
  - `hour_cos`
- Create `wet_rush_hour`

### 4. Model Training

- Use LightGBM for binary classification
- Train on the first 80% of the timeline
- Test on the last 20% of the timeline

### 5. Explainability and Dashboard

- Use SHAP summary plots to explain important drivers
- Use Streamlit for interactive scenario scoring and visuals

## Running the Project

### 1. Create and activate environment

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the training pipeline

```bash
python3 traffic_risk_engine.py
```

This will:

- clean and filter the dataset
- train the model
- generate performance plots
- export dashboard artifacts

### 3. Run the Streamlit dashboard

```bash
streamlit run app.py
```

## Dashboard Features

- scenario-based accident probability prediction
- H3 neighborhood risk lookup
- California spatial risk map
- weather comparison and sensitivity visuals
- temporal heatmap
- model feature importance
- validation visuals

## Explainability

SHAP is used to inspect which features influence predictions most strongly.  
The project consistently shows that:

- `neighbor_risk` is the dominant feature
- time and light conditions matter strongly
- weather and interaction features add additional refinement

## Deployment Notes

For deployment, the dashboard uses small summary artifacts like:

- `weather_frequency.csv`
- `dashboard_summary.json`

instead of scanning the full large dataset at runtime.

This is making the app more practical for hosted environments such as Streamlit Community Cloud.

## Limitations

- Synthetic safe scenarios can still create unrealistic combinations
- Some scenario comparisons are more useful for analysis than direct operational decisions
- The deployed dashboard uses lightweight summary artifacts instead of full live row-level analytics

## Future Improvements

- improve safe-scenario generation
- calibrate predicted probabilities
- add richer SHAP-based dashboard visuals
- expand deployment and documentation

## Acknowledgements
[1] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, and Rajiv Ramnath. “A Countrywide Traffic Accident Dataset.”, 2019.
[2] Moosavi, Sobhan, Mohammad Hossein Samavatian, Srinivasan Parthasarathy, Radu Teodorescu, and Rajiv Ramnath. "Accident Risk Prediction based on Heterogeneous Sparse Data: New Dataset and Insights." In proceedings of the 27th ACM SIGSPATIAL International Conference on Advances in Geographic Information Systems, ACM, 2019.

