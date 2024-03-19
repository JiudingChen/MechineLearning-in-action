import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

sns.set(style="whitegrid")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器，并设置参数
clf = DecisionTreeClassifier(
    criterion='entropy',  # 选择分裂节点的标准，可以是'gini'或者'entropy'
    splitter='best',  # 选择分裂策略，可以是'best'或者'random'
    max_depth=3,  # 树的最大深度，可以是任何整数或者None
    min_samples_split=4,  # 分裂内部节点所需的最小样本数
    min_samples_leaf=2,  # 叶节点必须有的最小样本数量
    max_features=None,  # 分裂时考虑的最大特征数量
    random_state=42  # 控制随机性的种子值
)

# 训练分类器
clf.fit(X_train, y_train)

# 进行预测
y_pred = clf.predict(X_test)

# 评估模型
report = classification_report(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names.tolist(), fontsize=12)
plt.title("Decision Tree Visualization", fontsize=24)
plt.show()

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names,
            yticklabels=iris.target_names)
plt.xlabel('Predicted Labels', fontsize=14)
plt.ylabel('True Labels', fontsize=14)
plt.title("Confusion Matrix", fontsize=16)
plt.show()

# 打印评估报告
print(report)
