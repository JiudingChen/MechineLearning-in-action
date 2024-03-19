import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 加载数据集
file_path = 'D:\python\Python机器学习\\1_线性回归\Real_estate.csv'
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

# 将 y_train 和 y_test 转换为 numpy 数组并确保是二维数组
y_train_np = y_train.values.reshape(-1, 1)
y_test_np = y_test.values.reshape(-1, 1)

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# 梯度下降算法
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m, n = X.shape
    X_b = np.c_[np.ones((m, 1)), X]  # 添加截距项
    theta = np.zeros((n + 1, 1))  # 初始化参数
    cost_history = []  # 记录每次迭代的成本

    for i in range(iterations):
        predictions = X_b.dot(theta)
        errors = predictions - y
        gradients = 2 / m * X_b.T.dot(errors)
        theta -= learning_rate * gradients
        cost = np.mean(errors ** 2)/2
        cost_history.append(cost)

    return theta, cost_history


# 比较不同学习率
learning_rates = [0.001, 0.01, 0.1]
plt.figure(figsize=(10, 6))

for lr in learning_rates:
    _, cost_history = gradient_descent(X_train_scaled, y_train_np, learning_rate=lr, iterations=1000)
    plt.plot(cost_history, label=f"LR = {lr}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Variation with Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

# 使用学习率 0.001 进行梯度下降
theta, _ = gradient_descent(X_train_scaled, y_train_np, learning_rate=0.001, iterations=1000)

# 构建多元线性回归方程字符串
equation_parts = [f"{theta[0][0]:.4f}"]
for coef, feature in zip(theta[1:], features.columns):
    equation_parts.append(f"{coef[0]:.4f}*{feature}")

linear_regression_equation = " + ".join(equation_parts)
print("Linear Regression Equation:\n")
print(f"y = {linear_regression_equation}")

# 使用训练好的模型进行预测
X_test_b = np.c_[np.ones((len(X_test_scaled), 1)), X_test_scaled]
y_pred = X_test_b.dot(theta)

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
# 使用拟合出的函数对所有数据进行预测，并进行折线图可视化比较

# 首先对所有数据进行特征标准化
all_features_scaled = scaler.transform(features)

# 添加截距项并进行预测
X_all_b = np.c_[np.ones((len(all_features_scaled), 1)), all_features_scaled]
y_all_pred = X_all_b.dot(theta)

# 排序以便于绘制折线图
sorted_indices = np.argsort(target.values)
sorted_y_all = target.values[sorted_indices]
sorted_y_all_pred = y_all_pred[sorted_indices, 0]
