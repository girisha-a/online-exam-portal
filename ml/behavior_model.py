import pandas as pd
from sklearn.cluster import KMeans
import joblib
import os

# Ensure the script runs from the 'ml' directory or handles paths correctly
base_path = os.path.dirname(os.path.abspath(__file__))
dataset_path = os.path.join(base_path, "../dataset/student_behavior.csv")
model_path = os.path.join(base_path, "behavior_model.pkl")

data = pd.read_csv(dataset_path)

X = data[['time_spent', 'answer_changes', 'tab_switches']]

from sklearn.ensemble import IsolationForest

# Train Isolation Model
# contamination=0.1 means we expect ~10% of students to be potential outliers
model = IsolationForest(n_estimators=100, contamination=0.1, random_state=42)
model.fit(X)

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
