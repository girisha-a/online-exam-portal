import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

base_path = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(base_path, "performance_model.pkl")

data = pd.DataFrame({
    'score': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'result': [40, 45, 50, 60, 65, 70, 75, 80, 85, 95] # Percentage prediction
})

X = data[['score']]
y = data['result']

model = LinearRegression()
model.fit(X, y)

joblib.dump(model, model_path)
print(f"Model saved to {model_path}")
