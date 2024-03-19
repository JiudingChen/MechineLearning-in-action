import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# 决策树节点类
class TreeNode:
    def __init__(self, gini, num_samples, num_samples_per_class, predicted_class):
        self.gini = gini
        self.num_samples = num_samples
        self.num_samples_per_class = num_samples_per_class
        self.predicted_class = predicted_class
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None


# 计算基尼不纯度
def gini(y):
    m = len(y)
    return 1.0 - sum((np.sum(y == c) / m) ** 2 for c in np.unique(y))


# 寻找最佳分割
def best_split(X, y):
    """找到最佳分割"""
    m, n = X.shape
    if m <= 1:
        return None, None

    # 获取类别及其映射到整数索引的字典
    unique_classes = np.unique(y)
    class_indices = {c: i for i, c in enumerate(unique_classes)}

    num_parent = [np.sum(y == c) for c in unique_classes]
    best_gini = 1.0 - sum((n / m) ** 2 for n in num_parent)
    best_idx, best_thr = None, None

    for idx in range(n):
        thresholds, class_labels = zip(*sorted(zip(X[:, idx], y)))
        num_left = [0] * len(unique_classes)
        num_right = num_parent.copy()
        for i in range(1, m):
            c = class_indices[class_labels[i - 1]]
            num_left[c] += 1
            num_right[c] -= 1
            gini_left = 1.0 - sum((num_left[x] / i) ** 2 for x in range(len(unique_classes)))
            gini_right = 1.0 - sum((num_right[x] / (m - i)) ** 2 for x in range(len(unique_classes)))
            gini = (i * gini_left + (m - i) * gini_right) / m
            if thresholds[i] == thresholds[i - 1]:
                continue

            if gini < best_gini:
                best_gini = gini
                best_idx = idx
                best_thr = (thresholds[i] + thresholds[i - 1]) / 2
    return best_idx, best_thr


# 递归构建决策树
def grow_tree(X, y, depth=0, max_depth=3):
    num_samples_per_class = [np.sum(y == i) for i in np.unique(y)]
    predicted_class = np.argmax(num_samples_per_class)
    node = TreeNode(
        gini=gini(y),
        num_samples=len(y),
        num_samples_per_class=num_samples_per_class,
        predicted_class=predicted_class,
    )

    if depth < max_depth:
        idx, thr = best_split(X, y)
        if idx is not None:
            indices_left = X[:, idx] < thr
            X_left, y_left = X[indices_left], y[indices_left]
            X_right, y_right = X[~indices_left], y[~indices_left]
            node.feature_index = idx
            node.threshold = thr
            node.left = grow_tree(X_left, y_left, depth + 1, max_depth)
            node.right = grow_tree(X_right, y_right, depth + 1, max_depth)
    return node


# 对单个样本进行分类
def predict(sample, node):
    while node.left:
        if sample[node.feature_index] < node.threshold:
            node = node.left
        else:
            node = node.right
    return node.predicted_class


# 加载鸢尾花数据集
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 构建决策树并进行预测
tree = grow_tree(X_train, y_train, max_depth=3)
y_pred = [predict(x, tree) for x in X_test]

# 计算并打印性能指标
report = classification_report(y_test, y_pred, target_names=iris.target_names)
print('分类报告:\n', report)

# 绘制混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix for Iris Dataset Classification')
plt.show()

# 可视化分类结果
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis', edgecolor='k', s=50, label='Actual')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Actual Classification')
plt.legend()

plt.subplot(1, 2, 2)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', edgecolor='k', s=50, label='Predicted')
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')
plt.title('Predicted Classification')
plt.legend()

plt.show()
