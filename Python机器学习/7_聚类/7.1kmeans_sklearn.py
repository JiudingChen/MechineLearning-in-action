import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# 加载数据
data_path = 'D:\python\Python机器学习\\7_聚类\kmeans.txt'
data = pd.read_csv(data_path, sep='\s+', header=None)

# 数据集可视化
plt.figure(figsize=(10, 6))
plt.scatter(data[0], data[1], s=50)
plt.title('Data Visualization', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()

# 使用肘部方法确定最佳簇数
sse = {}
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(data)
    sse[k] = kmeans.inertia_

# 绘制肘部图形
plt.figure(figsize=(10, 6))
plt.plot(list(sse.keys()), list(sse.values()))
plt.title("Elbow Method for Optimal Clusters")
plt.xlabel("Number of Clusters")
plt.ylabel("SSE")
plt.show()

# 进行KMeans聚类
k = 4  # 选择的簇数
kmeans = KMeans(n_clusters=k, random_state=42)
data['cluster'] = kmeans.fit_predict(data)

# 聚类结果可视化
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[0], y=data[1], hue=data['cluster'], palette="deep", s=50)
plt.title('KMeans Clustering Results', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()

