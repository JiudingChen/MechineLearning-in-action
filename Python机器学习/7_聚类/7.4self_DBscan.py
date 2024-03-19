import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class DBSCANManual:
    def __init__(self, eps, min_samples):
        self.eps = eps  # 邻域的半径
        self.min_samples = min_samples  # 形成簇所需的最小样本数
        self.labels = None

    def fit_predict(self, X):
        labels = np.full(X.shape[0], -1)  # 初始化所有点的标签为-1
        C = 0  # 初始化簇的编号

        for i in range(X.shape[0]):
            if labels[i] != -1:
                continue  # 如果点i已经被访问过，则跳过

            # 寻找点i的邻域内的所有点
            neighbors = self._region_query(X, i)

            if len(neighbors) < self.min_samples:
                labels[i] = -1  # 标记为噪声点
            else:
                C += 1  # 创建一个新的簇
                self._expand_cluster(X, labels, i, neighbors, C)

        self.labels = labels
        return labels

    def _expand_cluster(self, X, labels, i, neighbors, C):
        labels[i] = C  # 将点i标记为簇C
        k = 0
        while k < len(neighbors):
            j = neighbors[k]
            if labels[j] == -1:
                labels[j] = C  # 将噪声点标记为簇C
            elif labels[j] == 0:
                labels[j] = C  # 将未访问的点标记为簇C
                neighbors_j = self._region_query(X, j)
                if len(neighbors_j) >= self.min_samples:
                    neighbors += neighbors_j  # 添加新的邻域点到队列
            k += 1

    def _region_query(self, X, i):
        # 计算点i与数据集中所有点的距离
        distances = np.linalg.norm(X - X[i], axis=1)
        neighbors = np.where(distances < self.eps)[0].tolist()
        return neighbors


# 加载数据
data_path = 'D:\python\Python机器学习\\7_聚类\kmeans.txt'
data = pd.read_csv(data_path, sep='\s+', header=None)

# 使用手写的DBSCAN算法进行聚类
dbscan_manual = DBSCANManual(eps=0.5, min_samples=5)
data['dbscan_manual_cluster'] = dbscan_manual.fit_predict(data.values[:, :2])

# 可视化手写DBSCAN聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[0], y=data[1], hue=data['dbscan_manual_cluster'], palette="deep", s=50)
plt.title('Manual DBSCAN Clustering Results', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()
