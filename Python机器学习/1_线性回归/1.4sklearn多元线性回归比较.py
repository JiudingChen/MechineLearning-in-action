import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


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
        cost = np.mean(errors ** 2) / 2
        cost_history.append(cost)

    return theta, cost_history


# 加载数据集
file_path = 'D:\python\Python机器学习\\1_线性回归\Real_estate.csv'
data = pd.read_csv(file_path)

# # 相关性分析并绘制热力图
# correlation_matrix = data.corr()
# plt.figure(figsize=(12, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=2)
# plt.title("Correlation Matrix")
# plt.show()

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

# 使用 scikit-learn 的 LinearRegression 类训练多元线性回归模型
sklearn_model = LinearRegression().fit(X_train_scaled, y_train_np)

# 获取 scikit-learn 模型的系数
sklearn_intercept = sklearn_model.intercept_[0]
sklearn_coefficients = sklearn_model.coef_[0]

# 构建 scikit-learn 多元线性回归方程字符串
sklearn_equation = f"{sklearn_intercept:.4f}"
for coef, feature in zip(sklearn_coefficients, features.columns):
    sklearn_equation += f" + {coef:.4f}*{feature}"

print("scikit-learn Linear Regression Equation:\n")
print(f"y = {sklearn_equation}")

# 使用梯度下降算法
theta, _ = gradient_descent(X_train_scaled, y_train_np, learning_rate=0.001, iterations=1000)

# 构建梯度下降多元线性回归方程字符串
gradient_descent_equation = f"{theta[0][0]:.4f}"
for i, feature in enumerate(features.columns):
    gradient_descent_equation += f" + {theta[i + 1][0]:.4f}*{feature}"

print("Gradient Descent Linear Regression Equation:\n")
print(f"y = {gradient_descent_equation}")

# 比较 scikit-learn 和梯度下降法的系数
plt.figure(figsize=(12, 8))
bar_width = 0.35
index = np.arange(len(sklearn_coefficients) + 1)

bar1 = plt.bar(index, np.insert(sklearn_coefficients, 0, sklearn_intercept), bar_width, label='scikit-learn')
bar2 = plt.bar(index + bar_width, theta.flatten(), bar_width, label='Gradient Descent')

plt.xlabel('Coefficient Index')
plt.ylabel('Coefficient Value')
plt.title('Comparison of Coefficients between scikit-learn and Gradient Descent')
plt.xticks(index)
plt.legend()
plt.grid(True)
plt.show()
