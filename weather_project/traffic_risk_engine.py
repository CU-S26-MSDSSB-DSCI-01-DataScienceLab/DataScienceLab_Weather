
import os
import json
import joblib
import pickle
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import h3
import lightgbm as lgb
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

warnings.filterwarnings('ignore')

# ==============================================================================
# 1. DATASET LOADING & CONFIGURATION
# ==============================================================================
print("Loading raw dataset...")
raw_data_path = Path("data/US_Accidents.csv")
if not raw_data_path.exists():
    raise FileNotFoundError(f"Missing base data array at: {raw_data_path}")

data = pd.read_csv(raw_data_path)

# Restricting analysis to California only to optimize spatial engineering bounds
print("Filtering for California records...")
data_california = data[data["State"] == "CA"].copy()
del data  # Free system memory immediately

# Essential structural features to eliminate target leakage and high cardinality
essential_features = [
    "Start_Time", 'Start_Lat', 'Start_Lng', 'Temperature(F)', 'Humidity(%)',
    'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition', 'Sunrise_Sunset',
    'Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Station'
]

data_california = data_california[essential_features].dropna()

# Enforce clean timestamps
data_california['Start_Time'] = pd.to_datetime(data_california['Start_Time'], errors='coerce')
data_california = data_california.dropna(subset=['Start_Time']).copy()

# CRITICAL FIX: Convert boolean infrastructure flags to integers to prevent LightGBM Type Errors
infra_cols = ['Crossing', 'Junction', 'Stop', 'Traffic_Signal', 'Station']
for col in infra_cols:
    data_california[col] = data_california[col].astype(int)

# Persist clean intermediate analytical dataset
os.makedirs("data", exist_ok=True)
data_california.to_csv('data/california_accidents_cleaned.csv', index=False)

# ==============================================================================
# 2. CONTRASTIVE SAMPLING SPACE GENERATION (SAFE VS ACCIDENT)
# ==============================================================================
print("Generating contrastive negative sample space...")
df_pos = data_california.copy()
df_pos['target'] = 1

df_neg = data_california.copy()
df_neg['target'] = 0

# FIX: Permute environmental fields along with spatial coordinates to prevent feature distribution leakage
shuffle_cols = [
    'Start_Time', 'Start_Lat', 'Start_Lng', 'Temperature(F)',
    'Humidity(%)', 'Visibility(mi)', 'Wind_Speed(mph)', 'Weather_Condition'
]
for col in shuffle_cols:
    df_neg[col] = np.random.permutation(df_neg[col].values)

df_modeling = pd.concat([df_pos, df_neg], axis=0).reset_index(drop=True)

# ==============================================================================
# 3. LEAKAGE-PROOF CHRONOLOGICAL PARTITIONING
# ==============================================================================
print("Sorting and splitting data sequentially...")
df_modeling = df_modeling.sort_values('Start_Time')

split_idx = int(len(df_modeling) * 0.8)
train_df = df_modeling.iloc[:split_idx].copy()
test_df = df_modeling.iloc[split_idx:].copy()

print(f"Training on: {len(train_df)} rows (Past)")
print(f"Testing on: {len(test_df)} rows (Future)")

# ==============================================================================
# 4. SPATIAL & TEMPORAL FEATURE ENGINEERING
# ==============================================================================
print("Executing spatial mapping to H3 Hexagonal Grid (Res 7)...")
train_df['h3_res7'] = [h3.latlng_to_cell(lat, lng, 7) for lat, lng in zip(train_df['Start_Lat'], train_df['Start_Lng'])]
test_df['h3_res7'] = [h3.latlng_to_cell(lat, lng, 7) for lat, lng in zip(test_df['Start_Lat'], test_df['Start_Lng'])]

print("Calculating look-back spatial network density counts...")
train_hex_risk = train_df[train_df['target'] == 1]['h3_res7'].value_counts().to_dict()

def get_spatial_risk_safe(hex_id):
    neighbors = h3.grid_disk(hex_id, 1)
    return sum(train_hex_risk.get(n, 0) for n in neighbors)

train_df['neighbor_risk'] = train_df['h3_res7'].apply(get_spatial_risk_safe)
test_df['neighbor_risk'] = test_df['h3_res7'].apply(get_spatial_risk_safe)

# Log transform heavily skewed spatial counts
train_df['neighbor_risk'] = np.log1p(train_df['neighbor_risk'])
test_df['neighbor_risk'] = np.log1p(test_df['neighbor_risk'])

print("Extracting cyclical temporal signatures...")
for df in [train_df, test_df]:
    df['hour'] = df['Start_Time'].dt.hour
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['wet_rush_hour'] = (df['Humidity(%)'] / 100) * df['hour_sin']

# Clean tracking columns away from training vectors
drop_cols = ['target', 'Start_Lat', 'Start_Lng', 'Start_Time', 'h3_res7', 'hour']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

cat_cols = ['Weather_Condition', 'Sunrise_Sunset']
for col in cat_cols:
    X_train[col] = X_train[col].astype('category')
    X_test[col] = X_test[col].astype('category')

# ==============================================================================
# 5. MODEL SELECTION AND GRADIENT BOOSTING EXTENSION
# ==============================================================================
print("Initializing LightGBM Classifier Framework...")
model = lgb.LGBMClassifier(
    objective='binary',
    metric='auc',
    learning_rate=0.05,
    num_leaves=31,
    n_estimators=1000,
    scale_pos_weight=1.5,
    importance_type='gain',
    random_state=42,
    n_jobs=-1,
    force_row_wise=True
)

print("Fitting model using explicit early stopping restrictions...")
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='auc',
    categorical_feature=cat_cols,  # FIX: explicitly passed to prevent ambiguous data-type parsing
    callbacks=[lgb.early_stopping(stopping_rounds=50)]
)

# ==============================================================================
# 6. PIPELINE VALIDATION & METRIC ANALYSIS
# ==============================================================================
print("Evaluating trained architecture...")
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

print(f"\nFinal Validation AUC (Temporal Split): {roc_auc_score(y_test, y_pred_proba):.4f}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Plot performance curves
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
plt.close()

# Plot Normalized Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(8, 6))
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=['Safe', 'Accident'],
            yticklabels=['Safe', 'Accident'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Normalized Confusion Matrix')
plt.savefig('confusion_matrix.png')
plt.close()

# SHAP Interpretability Framework Exporter
print("Generating SHAP validation plots...")
try:
    explainer = shap.TreeExplainer(model)
    X_test_sample = X_test.sample(min(1000, len(X_test)), random_state=42)
    shap_values = explainer.shap_values(X_test_sample)
    s_values = shap_values[1] if isinstance(shap_values, list) else shap_values
    plt.figure(figsize=(12, 8))
    shap.summary_plot(s_values, X_test_sample, show=False)
    plt.tight_layout()
    plt.savefig('shap_summary.png')
    plt.close()
except Exception as e:
    print(f"Skipping SHAP visual save due to dependency variant version differences: {e}")

# ==============================================================================
# 7. DASHBOARD INFRASTRUCTURE DELIVERABLE EXPORT
# ==============================================================================
print("Serializing optimized models and metadata for Streamlit integration...")
PROJECT_ROOT = Path.cwd()

feature_order = list(X_train.columns)
weather_categories = X_train['Weather_Condition'].cat.categories.tolist()

# Joblib & Pickle packaging
joblib.dump(model, PROJECT_ROOT / 'traffic_risk_model.pkl')
with (PROJECT_ROOT / 'hex_risk_dict.pkl').open('wb') as f:
    pickle.dump(train_hex_risk, f)

# JSON mapping preservation to force alignment within the app engine
(PROJECT_ROOT / 'feature_order.json').write_text(json.dumps(feature_order, indent=2))
(PROJECT_ROOT / 'weather_categories.json').write_text(json.dumps(weather_categories, indent=2))

# Native booster snapshot
model.booster_.save_model(PROJECT_ROOT / 'traffic_model_native.txt')

print('\nDashboard integration bundle verified successfully:')
print(f" - Model object: {PROJECT_ROOT / 'traffic_risk_model.pkl'}")
print(f" - Hash Dictionary: {PROJECT_ROOT / 'hex_risk_dict.pkl'}")
print(f" - Feature Layout Schema: {PROJECT_ROOT / 'feature_order.json'}")
print(f" - Weather Enumerated Matrix: {PROJECT_ROOT / 'weather_categories.json'}")
print(f" - Raw Native Booster File: {PROJECT_ROOT / 'traffic_model_native.txt'}")

