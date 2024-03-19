from sklearn import datasets
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
import numpy as np

# 生成数据
x1, y1 = datasets.make_circles(n_samples=2000, factor=0.5, noise=0.05)
x2, y2 = datasets.make_blobs(n_samples=1000, centers=[[1.2, 1.2]], cluster_std=[0.1])

# 合并数据集
X = np.vstack((x1, x2))
y = np.hstack((y1, y2 + 3))  # +3为了区分y1和y2的标签

# 可视化原始数据
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette="deep")
plt.title("Original Data Visualization")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 应用KMeans聚类算法
kmeans = KMeans(n_clusters=2)
kmeans_labels = kmeans.fit_predict(X)

# 应用DBSCAN聚类算法
dbscan = DBSCAN(eps=0.1, min_samples=5)
dbscan_labels = dbscan.fit_predict(X)

# 可视化KMeans聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=kmeans_labels, palette="deep")
plt.title("KMeans Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

# 可视化DBSCAN聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=dbscan_labels, palette="deep")
plt.title("DBSCAN Clustering Results")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

'''
数据生成：

make_circles：生成了一个具有环形结构的数据集，这种结构对于基于距离的聚类算法是一种挑战。
make_blobs：生成了一个更传统的、基于中心点的聚类数据集。
K-means聚类结果：

K-means算法通过最小化簇内距离的总和来寻找簇中心。
在环形数据集上，K-means无法识别非凸形状的簇，因此不能正确分离环形结构。
在基于中心点的数据集上，K-means表现良好，能够准确地识别出簇。
DBSCAN聚类结果：

DBSCAN算法基于密度，能够识别任意形状的簇。
在环形数据集上，DBSCAN成功地识别出了环形结构，展现了其在处理复杂形状数据上的优势。
在基于中心点的数据集上，DBSCAN同样有效，能够正确识别出簇。
结论：

K-means适合于簇大小相似、簇形状凸的场景，但在处理复杂形状或大小不一的簇时可能效果不佳。
DBSCAN在处理任意形状的簇、尤其是非凸形状的簇时表现更好，但它对参数选择（如eps和min_samples）非常敏感。
这个比较展示了不同聚类算法在不同类型的数据集上的适用性和局限性。选择合适的聚类算法需要考虑数据的特性和聚类的目标。

'''
