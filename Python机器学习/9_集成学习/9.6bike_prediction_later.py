import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 加载数据集
file_path = 'D:\python\Python机器学习\9_集成学习\\bike_sharing_daily.xlsx'
bike_data = pd.read_excel(file_path)

# 选择特征和目标变量
X = bike_data.drop(['instant', 'dteday', 'casual', 'registered', 'cnt'], axis=1)
y = bike_data['cnt']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化梯度提升回归器模型
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

# 训练模型
gbr.fit(X_train, y_train)

# 预测测试集
y_pred = gbr.predict(X_test)

# 计算均方误差（MSE）和决定系数（R^2）
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 特征重要性可视化
features = X.columns
importances = gbr.feature_importances_
indices = np.argsort(importances)

plt.figure(figsize=(12, 8))
plt.title('Feature Importances')
plt.barh(range(len(indices)), importances[indices], color='b', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('Relative Importance')
plt.show()

# 预测值与实际值的对比可视化
plt.figure(figsize=(12, 8))
plt.scatter(y_test, y_pred, color='darkorange', label='Predicted Values')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Actual Values')
plt.xlabel('True Values')
plt.ylabel('Predictions')
plt.title('Comparison of True Values and Predictions')
plt.legend()
plt.show()

# 输出模型性能指标
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2:.2f}')
