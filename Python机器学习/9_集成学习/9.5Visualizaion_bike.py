import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 重新加载数据
bike_data_path = 'D:\\python\\Python机器学习\\9_集成学习\\bike_sharing_daily.xlsx'
bike_data = pd.read_excel(bike_data_path)

# 第一部分：前两个特征分布图
plt.figure(figsize=(24, 12))

plt.subplot(1, 2, 1)
sns.boxplot(x='season', y='cnt', data=bike_data)
plt.title('Bike Rentals by Season')
plt.xlabel('Season')
plt.ylabel('Total Rentals')

plt.subplot(1, 2, 2)
sns.boxplot(x='weekday', y='cnt', data=bike_data)
plt.title('Bike Rentals by Weekday')
plt.xlabel('Weekday')
plt.ylabel('Total Rentals')

plt.tight_layout()
plt.show()

# 第二部分：中间四个关系图
plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
sns.boxplot(x='weathersit', y='cnt', data=bike_data)
plt.title('Bike Rentals by Weather Situation')
plt.xlabel('Weather Situation')
plt.ylabel('Total Rentals')

plt.subplot(2, 2, 2)
sns.scatterplot(x='temp', y='cnt', data=bike_data)
plt.title('Bike Rentals vs. Temperature')
plt.xlabel('Temperature')
plt.ylabel('Total Rentals')

plt.subplot(2, 2, 3)
sns.scatterplot(x='atemp', y='cnt', data=bike_data)
plt.title('Bike Rentals vs. Feeling Temperature')
plt.xlabel('Feeling Temperature')
plt.ylabel('Total Rentals')

plt.subplot(2, 2, 4)
sns.scatterplot(x='hum', y='cnt', data=bike_data)
plt.title('Bike Rentals vs. Humidity')
plt.xlabel('Humidity')
plt.ylabel('Total Rentals')

plt.tight_layout()
plt.show()

# 第三部分：租赁数量随时间变化的趋势图
plt.figure(figsize=(15, 7))
sns.lineplot(x=bike_data.index, y='cnt', data=bike_data)
plt.title('Trend of Bike Rentals Over Time')
plt.xlabel('Time')
plt.ylabel('Total Rentals')
plt.show()
