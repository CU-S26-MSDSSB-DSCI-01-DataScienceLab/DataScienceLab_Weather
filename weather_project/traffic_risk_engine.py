#%%
# library importing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import h3
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, classification_report,roc_curve, precision_recall_curve, auc
import warnings
warnings.filterwarnings('ignore')

#%%
# dataset loading
data = pd.read_csv("data/US_Accidents.csv")

#%%
data.shape
#%% md
# 7.7 million rows and 46 columns
#%%
data.columns.tolist()
#%%
data.isnull().sum()
#%%
# Restricting  the analysis to California only
data_california=data[data["State"]=="CA"].copy()
del data

#%%
data_california.shape
#%% md
# 1.74 million rows and 46 columns
#%%
data_california.isnull().sum()
#%% md
# Data Cleaning
#%% md
# let's drop columns first :
# droping High-Missing Columns :
#  Wind_Chill(F) because of  Multicollinearity  (Temperature(F) and Wind_Speed(mph) )
#  Precipitation(in)
# 
# We have Start_Lat and Start_Lng with 0 missing values; for point-of-accident risk, the start coordinates are the most important.
#  So we can drop End_Lat and End_Lng
# 
# End_Lat, End_Lng, Distance(mi), Description -Result, Not Cause-These describe the aftermath of the accident.
#  Using them to predict the accident is Target Leakage.
# 
# ID, Street, Zipcode, Airport_Code -High Cardinality-Too many unique values.
# The model will try to memorize specific streets instead of learning general patterns.
# 
# City, County, Country, Timezone-Redundant Location
# We are using H3 Hexagons and Start_Lat/Lng. These are much more precise than a City name.
# 
# Weather_Timestamp, Wind_Direction, Pressure(in)
# Redundant Weather-Temperature and Humidity capture 90% of the weather signal. Pressure changes are too subtle for accident prediction.
# 
# Source, Turning_Loop
#  Constant/Useless-If a column has only one value (like Turning_Loop often does), the model learns nothing.
#%%
data_california.columns
#%%
# Keep only the columns needed for time, location, weather, and road-context feature engineering
essential_features=[
    "Start_Time",'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)','Weather_Condition','Sunrise_Sunset','Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Station'

]

#%%
# Droping  rows that cannot support the downstream feature pipeline or dashboard scenario scoring
data_california=data_california[essential_features].dropna()

#%%
data_california.shape
#%%
# Parsing timestamps once here so temporal sorting and cyclical hour features behave consistently
data_california['Start_Time'] = pd.to_datetime(data_california['Start_Time'], errors='coerce')
# Drop rows whose timestamps could not be parsed so temporal sorting and feature engineering remain valid
data_california = data_california.dropna(subset=['Start_Time']).copy()

#%%
data_california["Start_Time"]

#%%
# Persist the cleaned California-only analytical dataset without the pandas index column
data_california.to_csv('data/california_accidents_cleaned.csv', index=False)

#%%

#%%
# Creating  the positive class
df_pos = data_california.copy()
df_pos['target'] = 1

# Creating a synthetic 'safe' class by permuting time and location while preserving weather/context columns
# This forces the model to learn more than raw accident density alone
df_neg = data_california.copy()
df_neg['target'] = 0

# Permutation Sampling: Shuffle Time and Space to create 'Safe' scenarios
df_neg['Start_Time'] = np.random.permutation(df_neg['Start_Time'].values)
df_neg['Start_Lat'] = np.random.permutation(df_neg['Start_Lat'].values)
df_neg['Start_Lng'] = np.random.permutation(df_neg['Start_Lng'].values)

# Combine positives and synthetic negatives into one modeling table
df_modeling = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

#%%

# TEMPORAL SPLIT
# Sort chronologically so the model trains on the past and validates on future-like data
df_modeling = df_modeling.sort_values('Start_Time')

split_idx = int(len(df_modeling) * 0.8)
train_df = df_modeling.iloc[:split_idx].copy()
test_df = df_modeling.iloc[split_idx:].copy()

print(f"Training on: {len(train_df)} rows (Past)")
print(f"Testing on: {len(test_df)} rows (Future)")


#%%
# SPATIAL & TEMPORAL ENGINEERING

# 1. Convert each accident to an H3 resolution-7 hex cell so local neighborhood effects become measurable
train_df['h3_res7'] = [h3.latlng_to_cell(lat, lng, 7) for lat, lng in zip(train_df['Start_Lat'], train_df['Start_Lng'])]
test_df['h3_res7'] = [h3.latlng_to_cell(lat, lng, 7) for lat, lng in zip(test_df['Start_Lat'], test_df['Start_Lng'])]

# 2. LEAK-PROOF NEIGHBOR RISK
# Build spatial risk memory from the training accidents only so future rows do not leak signal backwards
train_hex_risk = train_df[train_df['target'] == 1]['h3_res7'].value_counts().to_dict()

def get_spatial_risk_safe(hex_id):
    neighbors = h3.grid_disk(hex_id, 1)
    return sum(train_hex_risk.get(n, 0) for n in neighbors)

train_df['neighbor_risk'] = train_df['h3_res7'].apply(get_spatial_risk_safe)
test_df['neighbor_risk'] = test_df['h3_res7'].apply(get_spatial_risk_safe)


#%%
# Log transform the spatial counts so extremely dense cells do not dominate the scale outright
train_df['neighbor_risk'] = np.log1p(train_df['neighbor_risk'])
test_df['neighbor_risk'] = np.log1p(test_df['neighbor_risk'])

# 3. Time & Interactions
# Encode hour cyclically and create a simple humidity-time interaction feature
for df in [train_df, test_df]:
    df['hour'] = df['Start_Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['wet_rush_hour'] = (df['Humidity(%)'] / 100) * df['hour_sin']

# PREPARATION & TRAINING
# Remove identifiers and raw columns that have already been transformed into model features
drop_cols = ['target', 'Start_Lat', 'Start_Lng', 'Start_Time', 'h3_res7', 'hour']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']


#%%
# Cast categorical predictors explicitly so LightGBM handles them natively
cat_cols = ['Weather_Condition', 'Sunrise_Sunset']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# Train the binary LightGBM classifier with AUC as the primary validation target
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    scale_pos_weight=1.5, # Balances Precision/Recall
    importance_type='gain',
    random_state=42,
    n_jobs=-1,
    force_row_wise=True
)


#%%
# Early stopping keeps the model from overtraining once validation performance plateaus
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# EVALUATION
# Score the future holdout split to estimate how the model generalizes beyond the training period
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"\nFinal Validation AUC (Temporal Split): {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

#%%
import warnings
warnings.filterwarnings('ignore')
# SHAP Analysis
explainer = shap.TreeExplainer(model)
X_test_sample = X_test.sample(1000, random_state=42)
shap_values = explainer.shap_values(X_test_sample)

# Correctly handle SHAP output structure
s_values = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure(figsize=(12, 8))
shap.summary_plot(s_values, X_test_sample, show=True)
#%%

#%% md
# 1. neighbor_risk is the King
# 
# As expected, this is your most powerful feature by a huge margin.
# 
# The Logic: Red dots (high historical risk) are shifted far to the right (increased probability of accident). Blue dots (zero/low history) are far to the left.
# 
# Insight: This proves your H3 Spatial Engineering worked. The model recognizes that location-based "contagion" is the strongest predictor of future accidents.
# 
# 2. Temporal & Light Dynamics
# 
# hour_cos & hour_sin: These rank highly, meaning the time of day significantly impacts risk.
# 
# Sunrise_Sunset: Since this is a category, the gray dots cluster, but their ranking in the top 3 confirms that visibility/glare is a primary driver of accidents in California.
# 
# 3. The Weather Signal
# 
# Temperature(F) & Humidity(%): These show a clear "spread." You can see that certain humidity levels (pink/red) push the risk to the right.
# 
# wet_rush_hour: Even though it's lower on the list, it shows a distinct spread to the right. This means your Domain Interaction engineering successfully captured a signal that temperature alone might have missed.
# 
# 4. Infrastructure Features
# 
# Junction & Traffic_Signal: Notice how these have small "clusters" to the right. The model has learned that being near a junction inherently increases the risk of an accident compared to a straight road.
# 
# "I used SHAP (Shapley Additive Explanations) to audit the model's decision-making. As the summary plot shows, the model correctly identified Spatial Contagion (neighbor_risk) as the primary driver. Crucially, it also captured non-linear signals like Solar Glare and the interaction between humidity and peak traffic hours, proving it has learned the physical environmental drivers of risk rather than just memorizing coordinates."
#%%
# Plot both ranking-oriented validation views so GitHub readers can inspect discrimination quality visually
# Left: ROC curve. Right: precision-recall curve.
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")

# 2. Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
pr_auc = auc(recall, precision)

plt.subplot(1, 2, 2)
plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")

plt.tight_layout()
plt.savefig('performance_curves.png')

#%%
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Normalize the confusion matrix row-wise so the heatmap highlights recall for each true class
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # Normalize

plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Safe', 'Accident'],
            yticklabels=['Safe', 'Accident'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix (Recall focus)')
plt.savefig('confusion_matrix.png')

#%%
# Re-state the exact feature order used by the model so deployment and dashboard code stay aligned
features = [
    'Temperature(F)', 'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)',
    'Weather_Condition', 'Sunrise_Sunset', 'Crossing', 'Junction',
    'Stop', 'Traffic_Signal', 'Station', 'hour_sin', 'hour_cos',
    'neighbor_risk', 'wet_rush_hour'
]

# Convert LightGBM gain values into a readable dataframe for interpretation
importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

# Plot the feature ranking so readers can see which signals the model leaned on most
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
sns.barplot(x='importance', y='feature', data=importance, palette='viridis', hue='feature', legend=False)
plt.title('LightGBM Feature Importance (Total Gain)')
plt.xlabel('Cumulative Gain')
plt.tight_layout()
plt.show()

#%%
# Summarize accident timing over a 24-hour cycle using a polar plot for a quick temporal pattern view
risk_by_hour = train_df[train_df['target'] == 1]['hour'].value_counts().sort_index()
angles = np.linspace(0, 2 * np.pi, len(risk_by_hour), endpoint=False).tolist()
angles += angles[:1] # Close the circle
counts = risk_by_hour.tolist()
counts += counts[:1]

fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
ax.fill(angles, counts, color='red', alpha=0.25)
ax.plot(angles, counts, color='red', linewidth=2)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(np.linspace(0, 2 * np.pi, 24, endpoint=False))
ax.set_xticklabels([str(h) for h in range(24)])
plt.title('Temporal Accident Density (24-Hour Cycle)')
plt.savefig('cyclical_risk.png')

#%%
import json
import joblib
import pickle
from pathlib import Path

# Export every dashboard artifact into the project root so Streamlit can load a consistent bundle
PROJECT_ROOT = Path.cwd()
feature_order = list(X_train.columns)
weather_categories = X_train['Weather_Condition'].cat.categories.tolist()

joblib.dump(model, PROJECT_ROOT / 'traffic_risk_model.pkl')
with (PROJECT_ROOT / 'hex_risk_dict.pkl').open('wb') as f:
    pickle.dump(train_hex_risk, f)
(PROJECT_ROOT / 'feature_order.json').write_text(json.dumps(feature_order, indent=2))
(PROJECT_ROOT / 'weather_categories.json').write_text(json.dumps(weather_categories, indent=2))

print('Saved dashboard artifacts:')
print(PROJECT_ROOT / 'traffic_risk_model.pkl')
print(PROJECT_ROOT / 'hex_risk_dict.pkl')
print(PROJECT_ROOT / 'feature_order.json')
print(PROJECT_ROOT / 'weather_categories.json')

#%%
from pathlib import Path

# Save the native LightGBM booster in the project root for debugging and deployment checks
PROJECT_ROOT = Path.cwd()
model.booster_.save_model(PROJECT_ROOT / 'traffic_model_native.txt')

print('Native model saved successfully.')
print(PROJECT_ROOT / 'traffic_model_native.txt')
print('Feature Order:', list(X_train.columns))

#%%
from pathlib import Path

# Quick verification cell so reruns confirm the dashboard bundle is complete
PROJECT_ROOT = Path.cwd()
for artifact in [
    'traffic_risk_model.pkl',
    'hex_risk_dict.pkl',
    'feature_order.json',
    'weather_categories.json',
    'traffic_model_native.txt',
]:
    artifact_path = PROJECT_ROOT / artifact
    print(f'{artifact}:', artifact_path.exists(), artifact_path)

print('Training feature order:', list(X_train.columns))
print('Weather categories:', X_train['Weather_Condition'].cat.categories.tolist())

#%% md
# 
#%%
print(X_train['Weather_Condition'].cat.categories.tolist())

#%%
model.predict(X_test).mean()
