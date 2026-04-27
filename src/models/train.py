import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train_model(data_dir, models_dir):
    # Thiết lập MLflow với backend SQLite để hỗ trợ Model Registry
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Diabetes_Prediction_Experiment")
    
    # Load tập huấn luyện
    X_train = pd.read_csv(os.path.join(data_dir, 'X_train.csv'))
    y_train = pd.read_csv(os.path.join(data_dir, 'y_train.csv')).values.ravel()
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    with mlflow.start_run():
        # Khai báo tham số
        params = {
            "n_estimators": 100,
            "max_depth": 7,
            "min_samples_split": 10,
            "min_samples_leaf": 4,
            "random_state": 42,
            "class_weight": "balanced"
        }
        mlflow.log_params(params)
        
        # Khởi tạo và huấn luyện Random Forest
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Đánh giá trên tập test để log vào MLflow
        y_pred = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred)
        }
        mlflow.log_metrics(metrics)
        print(f"Model trained & logged. Test Accuracy: {metrics['accuracy']:.4f}")
        
        # Đăng ký phiên bản mô hình (Model Versioning) lên MLflow Registry
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
            registered_model_name="Diabetes_Risk_Predictor"
        )
        
        # Vẫn lưu file pkl local phục vụ API fallback
        joblib.dump(model, os.path.join(models_dir, 'model.pkl'))

if __name__ == "__main__":
    train_model("data/processed/", "models/")