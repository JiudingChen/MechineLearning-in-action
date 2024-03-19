import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score

# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target

# 将数据集分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建KNN分类器实例
knn = KNeighborsClassifier(n_neighbors=3)

# 使用训练集数据训练模型
knn.fit(X_train, y_train)

# 在测试集上进行预测
predictions = knn.predict(X_test)

# 计算并打印模型的准确率和分类报告
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions)
print(f'Accuracy: {accuracy * 100:.2f}%')
print('Classification Report:\n', report)


# 可视化函数
def plot_iris_knn_results(X_train, y_train, X_test, y_test, model):
    plt.figure(figsize=(12, 8))

    # 设置颜色映射
    cmap_light = plt.cm.Spectral
    cmap_bold = ['darkorange', 'c', 'darkblue']

    # 设置边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # 构建网格
    xx, yy = np.meshgrid(np.arange(x_min, x_max, .02),
                         np.arange(y_min, y_max, .02))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, cmap=cmap_light, alpha=0.8)

    # 绘制训练集和测试集数据点
    for i, color in zip(range(3), cmap_bold):
        idx = np.where(y_train == i)
        plt.scatter(X_train[idx, 0], X_train[idx, 1], c=color, label=f'Iris class {i}',
                    cmap=cmap_bold, edgecolor='black', s=20)
    for i, color in zip(range(3), cmap_bold):
        idx = np.where(y_test == i)
        plt.scatter(X_test[idx, 0], X_test[idx, 1], c=color, marker='x', label=f'Iris class {i} (Test)',
                    cmap=cmap_bold, edgecolor='black', s=100)

    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('3-Class Classification (k = 3, weights = uniform)')
    plt.legend(loc='upper left')
    plt.show()


# 只使用前两个特征（萼片长度和宽度）进行可视化
X_train_2d, X_test_2d = X_train[:, :2], X_test[:, :2]
knn_2d = KNeighborsClassifier(n_neighbors=3)
knn_2d.fit(X_train_2d, y_train)

# 进行可视化
plot_iris_knn_results(X_train_2d, y_train, X_test_2d, y_test, knn_2d)
