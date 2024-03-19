import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
file_path = 'D:\python\Python机器学习\9_集成学习\\bike_sharing_daily.xlsx'
bike_data = pd.read_excel(file_path)

# 数据的统计描述，确保数据的合理性
stats_description = bike_data.describe()

# 选择特征和目标变量
features = bike_data.drop(['dteday', 'cnt'], axis=1) # 除去日期和目标变量的其他所有列作为特征
target = bike_data['cnt']

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.25, random_state=42)

# 初始化梯度提升回归器模型
gbr = GradientBoostingRegressor(n_estimators=100, random_state=42)

# 训练模型
gbr.fit(X_train, y_train)

# 预测测试集
y_pred = gbr.predict(X_test)

# 计算性能指标
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# 输出性能指标结果
print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared Score: {r2:.2f}')

# 可视化特征重要性
feature_importance = gbr.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + .5
plt.figure(figsize=(12, 6))
plt.barh(pos, feature_importance[sorted_idx], align='center')
plt.yticks(pos, features.columns[sorted_idx])
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Predicting Daily Bike Rentals')
plt.show()

# 可视化实际值与预测值
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted Daily Bike Rentals')
plt.show()
