import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report

# 数据是否需要标准化
scale = True

# 载入数据
data = np.genfromtxt("D:\python\Python机器学习\\3_逻辑回归\LR-testSet.csv", delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1]


# 数据可视化
def plot_data():
    x0 = x_data[y_data == 0]
    x1 = x_data[y_data == 1]
    plt.scatter(x0[:, 0], x0[:, 1], c='b', marker='o', label='Label 0')
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='x', label='Label 1')
    plt.legend()
    plt.xlabel('Feature1')
    plt.ylabel('Feature2')
    plt.title('Data Distribution')
    plt.show()


plot_data()


# Sigmoid函数
def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))


# 代价函数
def compute_cost(xMat, yMat, ws):
    left = np.multiply(yMat, np.log(sigmoid(xMat * ws)))
    right = np.multiply(1 - yMat, np.log(1 - sigmoid(xMat * ws)))
    return np.sum(left + right) / (-len(xMat))


# 梯度上升算法
def gradient_ascent(xArr, yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).transpose()
    lr = 0.001  # 学习率
    epochs = 10000  # 迭代次数
    m, n = np.shape(xMat)
    ws = np.mat(np.ones((n, 1)))
    cost_list = []  # 记录每次迭代的代价值

    for i in range(epochs + 1):
        h = sigmoid(xMat * ws)
        ws_grad = xMat.T * (h - yMat) / m
        ws = ws - lr * ws_grad
        if i % 50 == 0:
            cost_list.append(compute_cost(xMat, yMat, ws))
    return ws, cost_list


# 添加偏置项
X_data = np.concatenate((np.ones((100, 1)), x_data), axis=1)

# 训练模型，获取权重和代价列表
ws, cost_list = gradient_ascent(X_data, y_data)


# 绘制代价变化曲线
def plot_cost(cost_list):
    plt.plot(cost_list)
    plt.xlabel('Iterations')
    plt.ylabel('Cost')
    plt.title('Cost Trend')
    plt.show()


plot_cost(cost_list)


# 预测函数
def predict(x_data, ws):
    xMat = np.mat(x_data)
    ws = np.mat(ws)
    return [1 if x >= 0.5 else 0 for x in sigmoid(xMat * ws)]


# 执行预测并输出分类报告
predictions = predict(X_data, ws)
print(classification_report(y_data, predictions))
