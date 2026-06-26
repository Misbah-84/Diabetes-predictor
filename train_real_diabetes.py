import os
import sys
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Support UTF-8 emoji printing on Windows terminals
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

print("=" * 60)
print("DOWNLOADING REAL PIMA INDIANS DIABETES DATASET")
print("=" * 60)

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
dataset_path = "diabetes_dataset.csv"

try:
    print(f"Downloading from {url}...")
    urllib.request.urlretrieve(url, dataset_path)
    print(f"✅ Saved raw data locally to: {os.path.abspath(dataset_path)}")
except Exception as e:
    print(f"❌ Failed to download dataset: {e}")
    sys.exit(1)

# Define column names
columns = [
    'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome'
]

# Load dataset
df = pd.read_csv(dataset_path, header=None, names=columns)
print(f"Loaded dataset with shape: {df.shape}")

# Print class distribution
counts = df['Outcome'].value_counts()
print(f"Outcome distribution:\n{counts}")

# Compute median values of the features
print("\n" + "=" * 60)
print("COMPUTING FEATURE MEDIAN VALUES FOR IMPUTATION")
print("=" * 60)
medians = df.drop(columns=['Outcome']).median()
for col, val in medians.items():
    print(f"Median of {col:25}: {val}")
print("=" * 60 + "\n")

# Save medians to a file or print so we can copy them to server.py
# Pregnancies: 3.0, BloodPressure: 72.0, SkinThickness: 23.0, Insulin: 30.5, DiabetesPedigreeFunction: 0.3725
# Let's save the medians dictionary to a pickle or json just in case, or write it into server.py directly.
# Let's print out the exact values so we can embed them in server.py!

# Separate X and y
X = df.drop(columns=['Outcome'])
y = df['Outcome']

# Fit StandardScaler
print("Fitting StandardScaler on all 8 features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("✅ StandardScaler fit complete.")

# Train SVM Classifier
print("Training SVM Classifier...")
# SVC(kernel='rbf', probability=True, random_state=42)
svm_model = SVC(kernel='rbf', probability=True, random_state=42)
svm_model.fit(X_scaled, y)
print("✅ SVM Classifier training complete.")

# Save assets
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(models_dir, exist_ok=True)

scaler_path = os.path.join(models_dir, "scaler.pkl")
model_path = os.path.join(models_dir, "svm_model.pkl")

joblib.dump(scaler, scaler_path)
joblib.dump(svm_model, model_path)

print("\n" + "=" * 60)
print("SAVED PRODUCTION ASSETS")
print("=" * 60)
print(f"Saved scaler to: {os.path.abspath(scaler_path)}")
print(f"Saved SVM model to: {os.path.abspath(model_path)}")
print("=" * 60)
