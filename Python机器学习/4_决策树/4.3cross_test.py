import numpy as np
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 设置决策树的参数范围
param_grid = {
    'max_depth': np.arange(1, 11),
    'min_samples_split': range(2, 11),
    'min_samples_leaf': range(1, 11)
}

# 创建决策树分类器实例
dt = DecisionTreeClassifier(random_state=42)

# 创建GridSearchCV实例
grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='accuracy') # 可以更换为其他评分标准

# 执行网格搜索
grid_search.fit(X_train, y_train)

# 获取最佳参数和评分
best_params = grid_search.best_params_
best_score = grid_search.best_score_

print("Best Parameters:", best_params)
print("Best Score:", best_score)

# 使用最佳参数在测试集上进行预测
best_dt = grid_search.best_estimator_
y_pred = best_dt.predict(X_test)

# 打印分类报告
print(classification_report(y_test, y_pred))
