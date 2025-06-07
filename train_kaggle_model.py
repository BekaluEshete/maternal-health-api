import os
import zipfile
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import joblib

# ---------------------------------------------
# STEP 1: Verify Kaggle API setup
# ---------------------------------------------
kaggle_json_path = os.path.expanduser("~/.kaggle/kaggle.json")
if not os.path.exists(kaggle_json_path):
    raise FileNotFoundError(f"kaggle.json not found at {kaggle_json_path}. Please set up Kaggle API credentials.")

# ---------------------------------------------
# STEP 2: Download dataset from Kaggle
# ---------------------------------------------
print("Downloading dataset from Kaggle...")
dataset = "csafrit2/maternal-health-risk-data"  # Correct dataset identifier
download_command = f"kaggle datasets download -d {dataset}"
if os.system(download_command) != 0:
    raise Exception(f"Failed to download dataset {dataset}. Ensure you have accepted the dataset terms at https://www.kaggle.com/datasets/{dataset} and have valid credentials.")

# ---------------------------------------------
# STEP 3: Extract dataset
# ---------------------------------------------
print("Extracting dataset...")
os.makedirs("data", exist_ok=True)
try:
    with zipfile.ZipFile("maternal-health-risk-data.zip", "r") as zip_ref:
        zip_ref.extractall("data")
except Exception as e:
    raise Exception(f"Extraction failed: {e}")

# ---------------------------------------------
# STEP 4: Load and prepare dataset
# ---------------------------------------------
print("Preparing dataset...")
try:
    df = pd.read_csv("data/Maternal Health Risk Data Set.csv")
except FileNotFoundError:
    raise FileNotFoundError("Dataset file not found. Ensure the dataset was downloaded and extracted correctly.")

# Check for missing values
if df.isnull().any().any():
    print("Warning: Missing values detected. Filling with mean...")
    df.fillna(df.mean(numeric_only=True), inplace=True)

# Encode risk level as integers
df['RiskLevel'] = df['RiskLevel'].map({'low risk': 0, 'mid risk': 1, 'high risk': 2})

# Verify no unmapped categories
if df['RiskLevel'].isnull().any():
    raise ValueError("Unmapped categories in RiskLevel")

# Features and target (excluding BS to align with Flutter app, assuming Age is added)
X = df[['Age', 'SystolicBP', 'DiastolicBP', 'BodyTemp', 'HeartRate']]
y = df['RiskLevel']

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler for later use
os.makedirs("model", exist_ok=True)
joblib.dump(scaler, "model/scaler.pkl")

# Print scaler parameters for Flutter
print("Scaler Means:", scaler.mean_)
print("Scaler Std Devs:", scaler.scale_)

# ---------------------------------------------
# STEP 5: Train the model
# ---------------------------------------------
print("Training the model...")
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print("Model Evaluation:")
print(classification_report(y_test, y_pred, target_names=['Low Risk', 'Mid Risk', 'High Risk']))

# ---------------------------------------------
# STEP 6: Save the model
# ---------------------------------------------
joblib.dump(model, "model/risk_predictor.pkl")
print("✅ Model trained and saved as model/risk_predictor.pkl")

# ---------------------------------------------
# STEP 7: Convert to ONNX (optional for Flutter integration)
# ---------------------------------------------
try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    print("Converting model to ONNX...")
    initial_type = [('float_input', FloatTensorType([None, 5]))]
    onnx_model = convert_sklearn(model, initial_types=initial_type)
    with open("model/risk_predictor.onnx", "wb") as f:
        f.write(onnx_model.SerializeToString())
    print("✅ ONNX model saved as model/risk_predictor.onnx")
except ImportError:
    print("⚠️ skl2onnx not installed. Skipping ONNX conversion.")