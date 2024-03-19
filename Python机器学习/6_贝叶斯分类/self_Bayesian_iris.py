import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


# 定义高斯朴素贝叶斯分类器类
class GaussianNaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = []
        for c in self.classes:
            X_c = X[y == c]
            self.parameters.append(
                {
                    'mean': X_c.mean(axis=0),
                    'var': X_c.var(axis=0)
                }
            )

    def _pdf(self, x, mean, var):
        return 1 / (np.sqrt(2 * np.pi * var)) * np.exp(-((x - mean) ** 2) / (2 * var))

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        posteriors = []
        for i, c in enumerate(self.classes):
            prior = np.log(len(X[y == c]) / len(X))
            posterior = np.sum(np.log(self._pdf(x, self.parameters[i]['mean'], self.parameters[i]['var'])))
            posterior += prior
            posteriors.append(posterior)
        return self.classes[np.argmax(posteriors)]


# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建并训练手动实现的高斯朴素贝叶斯分类器
gnb_manual = GaussianNaiveBayes()
gnb_manual.fit(X_train, y_train)

# 进行预测
y_pred_manual = gnb_manual.predict(X_test)

# 生成分类报告
classification_rep_manual = classification_report(y_test, y_pred_manual, target_names=iris.target_names)

# 生成混淆矩阵
conf_matrix_manual = confusion_matrix(y_test, y_pred_manual)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_manual, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of Iris Dataset Classification (Manual Implementation)')
plt.show()

# 输出分类报告和混淆矩阵
print("分类报告 (Manual Implementation):\n", classification_rep_manual)
print("混淆矩阵 (Manual Implementation):\n", conf_matrix_manual)
