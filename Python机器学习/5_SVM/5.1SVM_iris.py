from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建SVM分类器并训练
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# 预测测试集结果
y_pred = svm_classifier.predict(X_test)

# 输出分类报告
print("分类报告:\n", classification_report(y_test, y_pred))

# 绘制并展示混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 使用PCA将数据降维到二维
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

# 重新训练SVM模型（使用降维后的数据）
svm_classifier_pca = SVC(kernel='linear')
svm_classifier_pca.fit(X_train_pca, y_train)


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
X0, X1 = X_test_pca[:, 0], X_test_pca[:, 1]
xx, yy = make_meshgrid(X0, X1)

# 绘制决策边界和测试数据点
plt.figure(figsize=(10, 8))
ax = plt.subplot(111)
plot_contours(ax, svm_classifier_pca, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
scatter = ax.scatter(X0, X1, c=y_test, cmap=plt.cm.coolwarm, s=20, edgecolors='k')

# 添加图表标题和坐标轴标签
ax.set_title('SVM Decision Region Boundary', size=16)
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')

# 设置图例
classes = ['Setosa', 'Versicolor', 'Virginica']
legend = ax.legend(*scatter.legend_elements(), title='Iris Species', loc='upper left')
for i, text in enumerate(legend.get_texts()):
    text.set_text(classes[i])
ax.add_artist(legend)

# 显示图形
plt.show()
