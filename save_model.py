"""
Run this script once to train and save the fraud detection model artifacts.

Usage:
    python save_model.py                          # uses archive/fraudTrain.csv by default
    python save_model.py /path/to/fraudTrain.csv  # custom path

Note: Upsampling to 2.5M rows (as in the notebook) can take 30-60 min and requires
~8 GB RAM. This script instead uses a balanced downsample (all fraud + equal non-fraud)
with scale_pos_weight, training in under a minute with comparable accuracy.
"""

import sys
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

train_path = sys.argv[1] if len(sys.argv) > 1 else 'archive/fraudTrain.csv'

print(f"Loading data from: {train_path}")
df = pd.read_csv(train_path)
df = df.dropna()
print(f"Loaded {len(df):,} rows after dropna.")

print("Extracting temporal features...")
df['trans_hour']  = pd.to_datetime(df['trans_date_trans_time']).dt.hour
df['trans_day']   = pd.to_datetime(df['trans_date_trans_time']).dt.day
df['trans_month'] = pd.to_datetime(df['trans_date_trans_time']).dt.month
df['trans_year']  = pd.to_datetime(df['trans_date_trans_time']).dt.year
df['birth_year']  = pd.to_datetime(df['dob']).dt.year
df['age']         = df['trans_year'] - df['birth_year']

drop_cols = [
    'Unnamed: 0', 'trans_date_trans_time', 'merchant',
    'first', 'last', 'street', 'city', 'zip',
    'job', 'trans_num', 'dob'
]
df = df.drop([c for c in drop_cols if c in df.columns], axis=1)

# Balance classes by downsampling majority to match minority count.
# This keeps the dataset small (~15k rows) so training finishes in seconds.
# scale_pos_weight in XGBoost further compensates for any residual imbalance.
print("Balancing classes (downsample majority to match minority)...")
df_majority = df[df['is_fraud'] == 0]
df_minority = df[df['is_fraud'] == 1]
n_minority  = len(df_minority)
df_majority_down = df_majority.sample(n=n_minority, random_state=42)
total = pd.concat([df_minority, df_majority_down]).sample(frac=1, random_state=42)
print(f"Balanced dataset: {len(total):,} rows  "
      f"(fraud={n_minority:,}, non-fraud={n_minority:,})")

total['gender'] = total['gender'].map({'M': 1, 'F': 0})

print("One-hot encoding category and state...")
total_enc = pd.get_dummies(total, columns=['category', 'state'], drop_first=False)

X = total_enc.drop('is_fraud', axis=1)
y = total_enc['is_fraud']

# Guarantee deterministic column order regardless of pandas version
base_cols  = ['cc_num','amt','gender','lat','long','city_pop',
              'unix_time','merch_lat','merch_long',
              'trans_hour','trans_day','trans_month','trans_year',
              'birth_year','age']
cat_cols   = sorted([c for c in X.columns if c.startswith('category_')])
state_cols = sorted([c for c in X.columns if c.startswith('state_')])
X = X[base_cols + cat_cols + state_cols]
feature_columns = list(X.columns)
print(f"Feature count: {len(feature_columns)}")

X_arr = X.values
y_arr = y.values

X_train, X_test, y_train, y_test = train_test_split(
    X_arr, y_arr, test_size=0.20, random_state=42
)

scaler        = StandardScaler()
X_train_sc    = scaler.fit_transform(X_train)
X_test_sc     = scaler.transform(X_test)

scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

print("Training XGBoost model...")
model = XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=6,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)
model.fit(X_train_sc, y_train)

y_pred   = model.predict(X_test_sc)
accuracy = accuracy_score(y_test, y_pred) * 100
print(f"\nAccuracy: {accuracy:.2f}%")
print(classification_report(y_test, y_pred))

print("Saving artifacts...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

print("\nDone! Files saved:")
print("  model.pkl")
print("  scaler.pkl")
print("  feature_columns.pkl")
print("\nNow run: python app.py")
