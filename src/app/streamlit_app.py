import streamlit as st
import pandas as pd
import requests
import os

st.title("Hệ thống dự đoán bệnh Tiểu đường 🩺")
st.markdown("Vui lòng nhập các thông số lâm sàng bên dưới để kiểm tra nguy cơ.")

# Giao diện nhập liệu tương ứng với 8 features
col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Số lần mang thai (Pregnancies)", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Đường huyết (Glucose)", min_value=0, max_value=300, value=100)
    blood_pressure = st.number_input("Huyết áp (BloodPressure)", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Độ dày nếp gấp da (SkinThickness)", min_value=0, max_value=150, value=20)

with col2:
    insulin = st.number_input("Lượng Insulin", min_value=0, max_value=1000, value=79)
    bmi = st.number_input("Chỉ số khối cơ thể (BMI)", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Chỉ số phả hệ (DiabetesPedigree)", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Tuổi (Age)", min_value=1, max_value=120, value=30)

if st.button("Dự đoán nguy cơ"):
    # Đóng gói dữ liệu thành dạng Dictionary (JSON)
    payload = {
        'Pregnancies': pregnancies, 'Glucose': glucose, 'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness, 'Insulin': insulin, 'BMI': bmi,
        'DiabetesPedigreeFunction': dpf, 'Age': age
    }
    
    # Dùng biến môi trường API_URL (cho Docker), mặc định fallback về localhost
    api_url = os.getenv("API_URL", "http://localhost:8000/predict")
    
    try:
        # Gửi HTTP POST request sang FastAPI
        response = requests.post(api_url, json=payload)
        result = response.json()
        
        if result["prediction"] == 1:
            st.error(f"⚠️ CẢNH BÁO: {result['risk_status']}! (Xác suất: {result['probability_1']:.2%})")
        else:
            st.success(f"✅ AN TOÀN: {result['risk_status']}. (Xác suất: {result['probability_0']:.2%})")
    except Exception as e:
        st.error(f"Lỗi: Không thể kết nối tới API tại `{api_url}`. Chi tiết: {e}")