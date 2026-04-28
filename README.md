# 🩺 Diabetes Risk Prediction - End-to-End MLOps Pipeline

## 📖 Giới thiệu
Dự án này là một hệ thống Machine Learning hoàn chỉnh (End-to-End MLOps) giúp dự đoán nguy cơ mắc bệnh tiểu đường dựa trên các chỉ số lâm sàng của bệnh nhân. 
Hệ thống được thiết kế theo kiến trúc Microservices, tách biệt hoàn toàn giữa Frontend (Giao diện người dùng) và Backend API, đồng thời đóng gói toàn diện bằng Docker.

## 🚀 Công nghệ sử dụng (Tech Stack)
- **Machine Learning:** Scikit-Learn (Random Forest Classifier)
- **Experiment Tracking & Model Registry:** MLflow (với SQLite Backend)
- **Backend/API (Serving Layer):** FastAPI & Uvicorn
- **Frontend/UI:** Streamlit
- **Dependency Management:** `uv` (Siêu tốc độ)
- **Containerization:** Docker & Docker Compose

## 🧠 Kiến trúc hệ thống
1. **Model Training Pipeline:** Tự động tiền xử lý dữ liệu, chuẩn hóa (StandardScaler), huấn luyện mô hình và lưu lại lịch sử/phiên bản mô hình thông qua MLflow.
2. **FastAPI Backend:** Tải mô hình đã huấn luyện và cung cấp RESTful API endpoint (`/predict`) để suy luận (Inference).
3. **Streamlit Frontend:** Cung cấp giao diện trực quan cho người dùng cuối (Bác sĩ/Bệnh nhân) nhập liệu và nhận kết quả cảnh báo.

## 🛠️ Hướng dẫn cài đặt và khởi chạy (Local)

Dự án được đóng gói hoàn toàn bằng Docker, bạn không cần cài đặt Python thủ công trên máy.

### Bước 1: Clone Repository
```bash
git clone https://github.com/your-username/diabetes-mlops-pipeline.git
cd diabetes-mlops-pipeline
```

### Bước 2: Khởi chạy hệ thống với Docker Compose
Đảm bảo bạn đã mở Docker Desktop, sau đó chạy lệnh:
```bash
docker-compose up --build -d
```

### Bước 3: Trải nghiệm ứng dụng
Sau khi quá trình build hoàn tất, hệ thống được triển khai trên nền tảng PaaS - Render: https://diabetes-app-2ku5.onrender.com/

## 📂 Cấu trúc thư mục (Project Structure)
```text
diabetes-mlops/
│
├── src/                    # Mã nguồn chính
│   ├── app/                # Chứa code API (FastAPI) và UI (Streamlit)
│   ├── data/               # Scripts tiền xử lý dữ liệu
│   ├── features/           # Scripts tạo đặc trưng (Feature Engineering)
│   └── models/             # Scripts huấn luyện & dự đoán (train.py, predict.py)
│
├── data/                   # Chứa dữ liệu (raw, processed)
├── models/                 # Nơi lưu Artifacts cục bộ (.pkl)
├── docker-compose.yml      # Cấu hình Microservices
├── Dockerfile              # Hướng dẫn build Container
├── pyproject.toml          # Cấu hình dependency cho uv
└── README.md
```
