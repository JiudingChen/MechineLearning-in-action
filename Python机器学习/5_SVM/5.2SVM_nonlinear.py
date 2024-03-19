import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

# 加载数据集
file_path = 'D:\python\Python机器学习\\5_SVM\LR-testSet2.txt'
data = pd.read_csv(file_path, sep='\t', header=None)
data = data[0].str.split(',', expand=True)
data.columns = ['Feature1', 'Feature2', 'Label']
data = data.astype({'Feature1': 'float', 'Feature2': 'float', 'Label': 'int'})

# 原始数据的可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Feature1', y='Feature2', hue='Label', data=data, palette='coolwarm')
plt.title('Data Distribution')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.show()

# 将数据集分割为特征和标签
X = data[['Feature1', 'Feature2']].values
y = data['Label'].values

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建具有RBF核的SVM分类器并训练
svm_classifier = SVC(kernel='rbf')
svm_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred = svm_classifier.predict(X_test)

# 输出分类报告和混淆矩阵
print("分类报告:\n", classification_report(y_test, y_pred))
conf_matrix = confusion_matrix(y_test, y_pred)
print("混淆矩阵:\n", conf_matrix)

# 绘制并展示混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# 函数：创建决策边界网格
def make_meshgrid(x, y, h=.02):
    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    return xx, yy


# 函数：绘制决策边界
def plot_contours(ax, clf, xx, yy, **params):
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    out = ax.contourf(xx, yy, Z, **params)
    return out


# 创建网格
X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

# 绘制决策边界和数据点
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
plot_contours(ax, svm_classifier, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# 添加图表标题和坐标轴标签
ax.set_title('SVM Decision Region Boundary', size=16)
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')

# 显示图形
plt.show()
