import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.datasets import load_iris


# 加载鸢尾花数据集
def load_iris_data():
    iris = load_iris()
    X, y = iris.data, iris.target
    return X, y


# KNN模型
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return np.array(predictions)

    def _predict(self, x):
        # 计算距离
        distances = [np.sqrt(np.sum((x_train - x) ** 2)) for x_train in self.X_train]
        # 获取最近k个点的索引
        k_indices = np.argsort(distances)[:self.k]
        # 获取这些点的标签
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # 多数投票
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# 数据集分割函数
def train_test_split(X, y, test_size=0.25):
    # 混洗数据集
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X, y = X[indices], y[indices]

    # 分割点
    split_idx = int(X.shape[0] * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    return X_train, X_test, y_train, y_test


# 可视化函数
def plot_iris_classification(X_train, y_train, X_test, y_test, predictions):
    plt.figure(figsize=(10, 6))

    # 绘制训练集数据
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Training Data')

    # 绘制测试集数据
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', label='Testing Data (True Label)')

    # 标记预测错误的点
    incorrect = (predictions != y_test)
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', marker='x', label='Misclassified')

    # 图表细节
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Classification using KNN')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 显示图表
    plt.show()


# 加载数据
X, y = load_iris_data()

# 分割数据集
X_train, X_test, y_train, y_test = train_test_split(X, y)

# 创建KNN实例并训练
knn = KNN(k=3)
knn.fit(X_train, y_train)

# 在测试集上进行预测
predictions = knn.predict(X_test)

# 计算准确率
accuracy = np.sum(predictions == y_test) / len(y_test)
print(f'Accuracy: {accuracy * 100:.2f}%')

# 输出每个测试样本的真实标签和预测标签
for i, (true_label, pred_label) in enumerate(zip(y_test, predictions)):
    print(f"Sample {i}: True Label = {true_label}, Predicted Label = {pred_label}")

# 改进可视化函数，添加决策边界
def plot_iris_classification_with_decision_boundary(X_train, y_train, X_test, y_test, predictions, model):
    plt.figure(figsize=(12, 8))

    # 生成网格点，用于绘制决策边界
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))

    # 预测网格点的类别
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # 绘制决策边界
    plt.contourf(xx, yy, Z, alpha=0.4)

    # 绘制训练集数据
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, marker='o', label='Training Data')

    # 绘制测试集数据
    plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, marker='s', label='Testing Data (True Label)')

    # 标记预测错误的点
    incorrect = (predictions != y_test)
    plt.scatter(X_test[incorrect, 0], X_test[incorrect, 1], c='red', marker='x', label='Misclassified')

    # 图表细节
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Sepal Width (cm)')
    plt.title('Iris Classification using KNN with Decision Boundary')
    plt.legend(loc='upper right')
    plt.grid(True)

    # 显示图表
    plt.show()

# 重新训练KNN模型，只使用前两个特征（为了绘制决策边界）
knn = KNN(k=3)
knn.fit(X_train[:, :2], y_train)
predictions = knn.predict(X_test[:, :2])

# 使用前两个特征（萼片长度和宽度）进行可视化，包括决策边界
plot_iris_classification_with_decision_boundary(X_train[:, :2], y_train, X_test[:, :2], y_test, predictions, knn)
