from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("model/risk_predictor.pkl")
scaler = joblib.load("model/scaler.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        input_data = np.array([[
            data['age'],
            data['bp_systolic'],
            data['bp_diastolic'],
            data['body_temp'],
            data['heart_rate']
        ]])
        scaled_data = scaler.transform(input_data)
        prediction = model.predict(scaled_data)[0]
        labels = ['Low Risk', 'Mid Risk', 'High Risk']
        return jsonify({'prediction': labels[prediction]})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)