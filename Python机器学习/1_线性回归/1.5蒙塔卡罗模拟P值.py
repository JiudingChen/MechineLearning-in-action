# 导入必要的库
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# 自定义函数计算 F 值
def calculate_f_statistic(y_true, y_pred, n_features):
    # 计算总的样本数量
    n_samples = len(y_true)

    # 计算残差平方和
    rss = np.sum((y_true - y_pred) ** 2)

    # 计算总平方和
    tss = np.sum((y_true - np.mean(y_true)) ** 2)

    # 计算模型平方和
    mss = tss - rss

    # 计算 F 统计量
    return (mss / n_features) / (rss / (n_samples - n_features - 1))


# 加载数据集
file_path = 'D:/python/Python机器学习/1_线性回归/Real_estate.csv'
data = pd.read_csv(file_path)

# 相关性分析并绘制热力图
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=2)
plt.title("Correlation Matrix")
plt.show()

# 选择特征和目标变量
features = data.drop(columns=['No', 'Y house price of unit area'])
target = data['Y house price of unit area']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型并计算原始模型的 F 值
model = LinearRegression().fit(X_train_scaled, y_train_np)
y_train_pred = model.predict(X_train_scaled)
original_f_value = calculate_f_statistic(y_train_np, y_train_pred, n_features=X_train_scaled.shape[1])

# 蒙特卡洛模拟计算 p 值
num_simulations = 1000
f_values_dist = np.zeros(num_simulations)

for i in range(num_simulations):
    shuffled_target = np.random.permutation(y_train_np)
    model = LinearRegression().fit(X_train_scaled, shuffled_target)
    y_shuffled_pred = model.predict(X_train_scaled)
    f_value = calculate_f_statistic(shuffled_target, y_shuffled_pred, n_features=X_train_scaled.shape[1])
    f_values_dist[i] = f_value

# 计算 p 值
p_value_estimate = np.mean(f_values_dist >= original_f_value)

# 可视化 F 值的分布
plt.figure(figsize=(12, 8))
sns.histplot(f_values_dist, kde=True, color='blue', label='Random F Values Distribution')
plt.axvline(x=original_f_value, color='red', linestyle='--', label='Original F Value')
plt.xlabel("F Value")
plt.ylabel("Frequency")
plt.title("Distribution of F Values and Original F Value")
plt.legend()
plt.grid(True)
plt.show()

# 输出 F 值和 p 值
print(f"Original F Value: {original_f_value:.4f}")
print(f"Estimated P Value: {p_value_estimate:.4f}")
