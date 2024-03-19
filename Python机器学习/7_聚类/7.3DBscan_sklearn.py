from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# 加载数据
data_path = 'D:\python\Python机器学习\\7_聚类\kmeans.txt'
data = pd.read_csv(data_path, sep='\s+', header=None)

# 应用DBSCAN聚类算法
dbscan = DBSCAN(eps=0.5, min_samples=5)  # eps为邻域大小，min_samples为形成簇所需的最小样本数
data['dbscan_cluster'] = dbscan.fit_predict(data)

# 聚类结果报告
n_clusters = len(set(data['dbscan_cluster'])) - (1 if -1 in data['dbscan_cluster'] else 0)
n_noise = list(data['dbscan_cluster']).count(-1)

print(f"Estimated number of clusters: {n_clusters}")
print(f"Estimated number of noise points: {n_noise}")

# 可视化DBSCAN聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[0], y=data[1], hue=data['dbscan_cluster'], palette="deep", s=50)
plt.title('DBSCAN Clustering Results', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()
