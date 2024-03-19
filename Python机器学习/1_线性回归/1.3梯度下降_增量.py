import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error


# 梯度下降算法（带动量）
def gradient_descent_with_momentum(X, y, learning_rate=0.01, iterations=1000, momentum=0.9):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # 添加截距项
    theta = np.zeros((n + 1, 1))  # 初始化参数
    velocity = np.zeros((n + 1, 1))  # 初始化动量项
    cost_history = []  # 记录每次迭代的成本

    for i in range(iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        velocity = momentum * velocity - learning_rate * gradients
        theta += velocity
        cost = np.mean((X_b.dot(theta) - y) ** 2) / 2
        cost_history.append(cost)

    return theta, cost_history


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

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 将 y_train 和 y_test 转换为 numpy 数组并确保是二维数组
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 使用学习率 0.001 进行梯度下降（带动量）
theta_momentum, cost_history_momentum = gradient_descent_with_momentum(X_train_scaled, y_train_np, learning_rate=0.001,
                                                                       iterations=1000)

# 可视化成本历史
plt.figure(figsize=(10, 6))
plt.plot(cost_history_momentum, label='Gradient Descent with Momentum')
plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Variation with Gradient Descent using Momentum")
plt.legend()
plt.grid(True)
plt.show()

# 使用训练好的模型进行预测
X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]
y_pred = X_test_b.dot(theta_momentum)

# 计算测试集上的均方误差
test_mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {test_mse}")

# 可视化测试集的预测结果
plt.figure(figsize=(10, 6))
plt.scatter(range(len(y_test)), y_test, color='blue', label='Actual Value')
plt.scatter(range(len(y_pred)), y_pred, color='red', label='Predicted Value')
plt.xlabel("Sample")
plt.ylabel("House Price")
plt.title("Prediction Performance on Test Set")
plt.legend()
plt.grid(True)
plt.show()
