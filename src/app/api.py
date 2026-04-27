from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import sys
import os

# Chỉ định lại sys path để gọi được module từ src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import hàm predict (Hỗ trợ fallback do khác biệt thư mục cấu trúc)
try:
    from src.models.predict import predict
except ImportError:
    from src.data.predict import predict

# Khởi tạo ứng dụng FastAPI
app = FastAPI(
    title="Diabetes Prediction API 🩺",
    description="Hệ thống API MLOps phục vụ dự đoán nguy cơ mắc bệnh tiểu đường",
    version="1.0.0"
)

# Định nghĩa schema cho dữ liệu đầu vào
class DiabetesInput(BaseModel):
    Pregnancies: float
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: float

@app.get("/")
def read_root():
    return {"message": "Chào mừng đến với Diabetes Prediction API. Truy cập /docs để xem tài liệu Swagger UI."}

@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    try:
        # Chuyển đổi Payload (JSON) thành DataFrame để đưa vào model
        input_df = pd.DataFrame([data.dict()])
        
        pred, prob = predict(input_df)
        
        return {
            "prediction": int(pred[0]),
            "probability_0": float(prob[0][0]),
            "probability_1": float(prob[0][1]),
            "risk_status": "Có nguy cơ mắc tiểu đường" if int(pred[0]) == 1 else "An toàn"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))