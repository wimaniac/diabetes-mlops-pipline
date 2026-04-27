import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_features(input_path, output_dir, models_dir):
    df = pd.read_csv(input_path)
    
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    # Train-test split (đảm bảo tỷ lệ nhãn y đồng đều qua stratify)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Chuẩn hoá dữ liệu (StandardScaler) cho Logistic Regression
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X.columns)
    
    # Lưu Scaler làm artifact
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
    
    # Lưu features đã xử lý
    os.makedirs(output_dir, exist_ok=True)
    X_train_scaled.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
    X_test_scaled.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
    print("Features built and scaler saved successfully.")

if __name__ == "__main__":
    build_features("data/processed/cleaned_diabetes.csv", "data/processed/", "models/")