import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from matplotlib.colors import ListedColormap
import seaborn as sns

# 加载数据
file_path = 'D:\python\Python机器学习\\4_决策树\LR-testSet2.txt'
data = pd.read_csv(file_path, header=None, names=["Feature1", "Feature2", "Label"])

# 分离特征和标签
X = data[["Feature1", "Feature2"]].values
y = data["Label"].values

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 使用决策树算法进行分类
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

# 创建决策边界网格
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                     np.arange(y_min, y_max, 0.01))

# 预测网格点的分类
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 设置颜色图
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# 绘制决策边界和数据点
plt.figure(figsize=(10, 6))
plt.contourf(xx, yy, Z, alpha=0.8, cmap=cmap_light)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold, edgecolor='k', s=20)
plt.title("Decision Tree Classification with Uploaded Dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 在测试集上进行预测
y_pred = dt.predict(X_test)

# 计算并输出准确率和其他指标
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# 打印性能指标
print(f"Accuracy: {accuracy}")
print("\\nClassification Report:\\n", report)
