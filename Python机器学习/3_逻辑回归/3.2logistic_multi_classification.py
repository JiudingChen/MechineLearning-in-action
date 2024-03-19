from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data[:, :2]  # 为了方便可视化，仅采用两个特征
y = iris.target


# 二分类逻辑回归模型
class BinaryLogisticRegression:
    def __init__(self, alpha=0.01, epochs=1000):
        self.alpha = alpha  # 学习率
        self.epochs = epochs  # 迭代次数
        self.weights = None  # 权重

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # 梯度下降法
        for _ in range(self.epochs):
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self._sigmoid(linear_model)

            # 计算梯度
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)

            # 更新权重
            self.weights -= self.alpha * dw
            self.bias -= self.alpha * db

    def predict(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(linear_model)
        y_predicted_cls = [1 if i > 0.5 else 0 for i in y_predicted]
        return y_predicted_cls

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-z))


# 一对其余（One-vs-Rest）策略
def train_one_vs_rest(X, y):
    classifiers = []
    for i in np.unique(y):
        y_binary = np.where(y == i, 1, 0)
        clf = BinaryLogisticRegression()
        clf.fit(X, y_binary)
        classifiers.append(clf)
    return classifiers


# 预测多分类结果
def predict_one_vs_rest(classifiers, X):
    predictions = [clf.predict(X) for clf in classifiers]
    predictions = np.array(predictions)
    predictions = np.argmax(predictions, axis=0)
    return predictions


# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
classifiers = train_one_vs_rest(X_train, y_train)

# 预测
predictions = predict_one_vs_rest(classifiers, X_test)

# 准确率
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# 可视化结果
plt.figure(figsize=(12, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)
plt.title('Iris Data - Logistic Regression One-vs-Rest')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.show()
