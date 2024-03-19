import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, f1_score

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化参数范围
max_depths = range(1, 11)
min_samples_splits = range(2, 11)
min_samples_leaves = range(1, 11)

# 初始化性能指标的存储列表
performance_metrics = {
    'max_depth': {'accuracy': [], 'recall': [], 'f1': []},
    'min_samples_split': {'accuracy': [], 'recall': [], 'f1': []},
    'min_samples_leaf': {'accuracy': [], 'recall': [], 'f1': []}
}

# 测试不同的 max_depth
for max_depth in max_depths:
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    performance_metrics['max_depth']['accuracy'].append(accuracy_score(y_test, y_pred))
    performance_metrics['max_depth']['recall'].append(recall_score(y_test, y_pred, average='macro'))
    performance_metrics['max_depth']['f1'].append(f1_score(y_test, y_pred, average='macro'))

# 测试不同的 min_samples_split
for min_samples_split in min_samples_splits:
    clf = DecisionTreeClassifier(min_samples_split=min_samples_split, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    performance_metrics['min_samples_split']['accuracy'].append(accuracy_score(y_test, y_pred))
    performance_metrics['min_samples_split']['recall'].append(recall_score(y_test, y_pred, average='macro'))
    performance_metrics['min_samples_split']['f1'].append(f1_score(y_test, y_pred, average='macro'))

# 测试不同的 min_samples_leaf
for min_samples_leaf in min_samples_leaves:
    clf = DecisionTreeClassifier(min_samples_leaf=min_samples_leaf, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    performance_metrics['min_samples_leaf']['accuracy'].append(accuracy_score(y_test, y_pred))
    performance_metrics['min_samples_leaf']['recall'].append(recall_score(y_test, y_pred, average='macro'))
    performance_metrics['min_samples_leaf']['f1'].append(f1_score(y_test, y_pred, average='macro'))


# 绘制不同参数下的性能指标比较图
def plot_metrics(param_range, metrics, title):
    plt.figure(figsize=(12, 6))
    plt.plot(param_range, metrics['accuracy'], label='Accuracy', marker='o')
    plt.plot(param_range, metrics['recall'], label='Recall', marker='s')
    plt.plot(param_range, metrics['f1'], label='F1 Score', marker='^')
    plt.xlabel('Parameter Value')
    plt.ylabel('Score')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()


plot_metrics(max_depths, performance_metrics['max_depth'], 'Performance Metrics at Different Tree Depths')
plot_metrics(min_samples_splits, performance_metrics['min_samples_split'],
             'Performance Metrics at Different Min Samples Split')
plot_metrics(min_samples_leaves, performance_metrics['min_samples_leaf'],
             'Performance Metrics at Different Min Samples Leaf')
