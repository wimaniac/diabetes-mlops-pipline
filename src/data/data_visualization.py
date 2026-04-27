import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('data/raw/diabetes.csv')
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

print(f"Kích thước X: {X.shape}, Kích thước y: {y.shape}")

for col in df.columns[:-1]:
    plt.figure(figsize=(8, 4))
    sns.histplot(data=df, x=col, hue='Outcome')
    plt.title(f'Phân phối {col} theo kết quả bệnh tiểu đường')
    plt.tight_layout()
    plt.show()