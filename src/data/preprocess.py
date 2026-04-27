import pandas as pd
import numpy as np
import os

def preprocess_data(input_path, output_path):
    df = pd.read_csv(input_path)
    
    # 1. Xử lý giá trị 0 (Thay bằng NaN)
    zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for col in zero_cols:
        df[col] = df[col].replace(0, np.nan)
        
    # 2. Điền giá trị khuyết thiếu bằng Median
    for col in zero_cols:
        df[col] = df[col].fillna(df[col].median())
        
    # 3. Xử lý ngoại lai bằng phương pháp IQR (Capping)
    features = df.columns.drop('Outcome')
    for col in features:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower_bound, upper_bound)
        
    # Lưu dữ liệu đã tiền xử lý
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data preprocessed and saved to {output_path}")

if __name__ == "__main__":
    input_file = "data/raw/diabetes.csv"
    output_file = "data/processed/cleaned_diabetes.csv"
    if os.path.exists(input_file):
        preprocess_data(input_file, output_file)
    else:
        print(f"File not found: {input_file}. Vui lòng đặt dữ liệu raw vào đúng thư mục.")