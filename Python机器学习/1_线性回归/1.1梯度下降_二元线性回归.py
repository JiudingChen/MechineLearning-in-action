# 首先使用二维的线性回归数据集应用梯度下降算法并评估其学习率对收敛的影响
# 由于想要采用的经典波士顿房价数据集已在sklearn的1.2版本中被移除
# 因此以使用其他内置数据集或者创建一个合成数据集来演示梯度下降算法。我们使用 make_regression 函数来创建一个适用于回归分析的合成数据集。
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# 创建合成数据集
X, y = make_regression(n_samples=500, n_features=1, noise=10, random_state=42)
y = y.reshape(-1, 1)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# 梯度下降算法
def gradient_descent(X, y, learning_rate=0.01, iterations=1000):
    m = len(X)
    X_b = np.c_[np.ones((m, 1)), X]  # 添加截距项
    theta = np.zeros((2, 1))
    cost_history = []

    for i in range(iterations):
        gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
        theta -= learning_rate * gradients
        cost = 1 / (2 * m) * np.sum((X_b.dot(theta) - y) ** 2)
        cost_history.append(cost)

    return theta, cost_history


# 设置不同的学习率进行实验
learning_rates = [0.01, 0.05, 0.1]
iterations = 1000
plt.figure(figsize=(12, 8))

# 对每个学习率进行梯度下降并绘图
for lr in learning_rates:
    _, cost_history = gradient_descent(X_train, y_train, learning_rate=lr, iterations=iterations)
    plt.plot(cost_history, label=f"LR = {lr}")

plt.xlabel("Iterations")
plt.ylabel("Cost")
plt.title("Cost Variation with Different Learning Rates")
plt.legend()
plt.grid(True)
plt.show()

# 使用一个学习率进行梯度下降并输出函数表达式
lr = 0.1
theta, _ = gradient_descent(X_train, y_train, learning_rate=lr, iterations=iterations)
print(f"Linear Regression Equation: y = {theta[1][0]} * x + {theta[0][0]}")

# 增加额外的可视化：最终模型的拟合效果
plt.figure(figsize=(12, 8))
plt.scatter(X_train, y_train, color='blue', label='Training Data')
plt.scatter(X_test, y_test, color='green', label='Test Data')
plt.plot(X_train, np.c_[np.ones((len(X_train), 1)), X_train].dot(theta), color='red', label='Fitted Line')
plt.xlabel("Feature X")
plt.ylabel("Target y")
plt.title("Linear Regression Fit")
plt.legend()
plt.grid(True)
plt.show()
