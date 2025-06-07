import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model/risk_predictor.pkl")
scaler = joblib.load("model/scaler.pkl")

# Sample inputs: [Age, SystolicBP, DiastolicBP, BodyTemp, HeartRate]
test_inputs = [
    [30, 120, 80, 98.6, 70],  # Normal values, expect Low Risk
    [40, 140, 90, 99.0, 80],  # Elevated BP, expect Mid/High Risk
    [25, 100, 60, 97.8, 65],  # Low BP, expect Low Risk
]

# Test predictions
labels = ['Low Risk', 'Mid Risk', 'High Risk']
for input_data in test_inputs:
    # Scale input
    scaled_data = scaler.transform(np.array([input_data]))
    # Predict
    prediction = model.predict(scaled_data)[0]
    print(f"Input: {input_data}")
    print(f"Prediction: {labels[prediction]}\n")