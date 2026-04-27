import pandas as pd
import os
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def evaluate_model(data_dir, models_dir, reports_dir="reports/"):
    # Load tập test
    X_test = pd.read_csv(os.path.join(data_dir, 'X_test.csv'))
    y_test = pd.read_csv(os.path.join(data_dir, 'y_test.csv')).values.ravel()
    
    # Load model đã được huấn luyện
    model = joblib.load(os.path.join(models_dir, 'model.pkl'))
    
    # Thực hiện dự đoán
    y_pred = model.predict(X_test)
    
    # Tính toán các độ đo
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print("ĐÁNH GIÁ MÔ HÌNH TRÊN TẬP TEST:")
    print("-" * 30)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-Score  : {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    
    # Lưu các độ đo vào file json
    os.makedirs(reports_dir, exist_ok=True)
    metrics = {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm.tolist()}
    
    with open(os.path.join(reports_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=4)
    print(f"\n✅ Đã lưu kết quả đánh giá vào {reports_dir}metrics.json")

if __name__ == "__main__":
    evaluate_model("data/processed/", "models/")