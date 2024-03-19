from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载鸢尾花数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 划分数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用高斯贝叶斯分类器进行模型训练
gnb = GaussianNB()
gnb.fit(X_train, y_train)

# 进行预测
y_pred = gnb.predict(X_test)

# 生成分类报告
classification_rep = classification_report(y_test, y_pred, target_names=iris.target_names)

# 生成混淆矩阵
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix of Iris Dataset Classification')
plt.show()

# 为了可视化，选择鸢尾花数据集的前两个特征
X_train_viz = X_train[:, :2]
X_test_viz = X_test[:, :2]
gnb_viz = GaussianNB()
gnb_viz.fit(X_train_viz, y_train)

# 在测试集上进行预测
y_pred_viz = gnb_viz.predict(X_test_viz)

# 可视化测试数据点
plt.figure(figsize=(10, 6))
for i, label in enumerate(iris.target_names):
    # 绘制实际类别
    plt.scatter(X_test_viz[y_test == i, 0], X_test_viz[y_test == i, 1], label=f'Actual {label}')
    # 绘制预测类别
    plt.scatter(X_test_viz[y_pred_viz == i, 0], X_test_viz[y_pred_viz == i, 1], marker='x', alpha=0.8,
                label=f'Predicted {label}')

plt.xlabel('Sepal Length (cm)')
plt.ylabel('Sepal Width (cm)')
plt.title('Iris Classification Results Visualization')
plt.legend()
plt.show()
