import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

data_path = "D:\\python\\Python机器学习\\3_逻辑回归\\LR-testSet.csv"
data = np.genfromtxt(data_path, delimiter=",")
x_data = data[:, :-1]
y_data = data[:, -1].astype(int)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(x_data, y_data)


# 可视化数据和决策边界
def plot_data_with_decision_boundary(x_data, y_data, model):
    # 分类数据点
    x0 = x_data[y_data == 0]
    x1 = x_data[y_data == 1]

    # 绘制数据点
    plt.scatter(x0[:, 0], x0[:, 1], c='b', marker='o', label='Label 0')
    plt.scatter(x1[:, 0], x1[:, 1], c='r', marker='x', label='Label 1')
    plt.title('Data Distribution')
    # 绘制决策边界
    x_min, x_max = x_data[:, 0].min() - 1, x_data[:, 0].max() + 1
    y_min, y_max = x_data[:, 1].min() - 1, x_data[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 500), np.linspace(y_min, y_max, 500))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z, levels=[0.5], cmap="Greys_r")

    # 图表设置
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.show()


plot_data_with_decision_boundary(x_data, y_data, model)

# 打印分类报告
predictions = model.predict(x_data)
print(classification_report(y_data, predictions))
