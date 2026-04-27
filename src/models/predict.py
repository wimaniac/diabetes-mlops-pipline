import joblib
import pandas as pd
import os

def load_artifacts(models_dir="models/"):
    # Tải cả model và scaler
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    return model, scaler

def predict(input_data: pd.DataFrame):
    model, scaler = load_artifacts()
    
    # Áp dụng chuẩn hoá cho luồng dự đoán
    input_scaled = scaler.transform(input_data)
    
    predictions = model.predict(input_scaled)
    probabilities = model.predict_proba(input_scaled)
    return predictions, probabilities