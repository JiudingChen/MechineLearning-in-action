import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


class KMeansManual:
    def __init__(self, n_clusters, max_iter=300, random_state=42):
        self.n_clusters = n_clusters  # 聚类数
        self.max_iter = max_iter  # 最大迭代次数
        self.random_state = random_state  # 随机种子
        self.centers = None  # 聚类中心点

    def initialize_centers(self, X):
        # 初始化聚类中心
        np.random.seed(self.random_state)
        random_idx = np.random.permutation(X.shape[0])
        centers = X[random_idx[:self.n_clusters]]
        return centers

    def compute_distance(self, X, centers):
        # 计算每个点到各个中心的距离
        distances = np.zeros((X.shape[0], self.n_clusters))
        for k in range(self.n_clusters):
            row_norm = np.linalg.norm(X - centers[k, :], axis=1)
            distances[:, k] = np.square(row_norm)
        return distances

    def update_centers(self, X, labels):
        # 更新聚类中心
        centers = np.zeros((self.n_clusters, X.shape[1]))
        for k in range(self.n_clusters):
            centers[k, :] = np.mean(X[labels == k, :], axis=0)
        return centers

    def compute_labels(self, distances):
        # 根据最近的中心分配标签
        return np.argmin(distances, axis=1)

    def fit(self, X):
        # 训练模型
        self.centers = self.initialize_centers(X)
        for _ in range(self.max_iter):
            distances = self.compute_distance(X, self.centers)
            labels = self.compute_labels(distances)
            new_centers = self.update_centers(X, labels)
            if np.all(new_centers == self.centers):
                break
            self.centers = new_centers
        return self

    def predict(self, X):
        # 对新数据进行分类
        distances = self.compute_distance(X, self.centers)
        return self.compute_labels(distances)


# 加载数据
data_path = 'D:\python\Python机器学习\\7_聚类\kmeans.txt'
data = pd.read_csv(data_path, sep='\s+', header=None)

# 应用手写的KMeans算法
kmeans_manual = KMeansManual(n_clusters=4)
kmeans_manual.fit(data.values)
data['cluster_manual'] = kmeans_manual.predict(data.values)

# 可视化手写KMeans算法的聚类结果
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data[0], y=data[1], hue=data['cluster_manual'], palette="deep", s=50)
plt.scatter(kmeans_manual.centers[:, 0], kmeans_manual.centers[:, 1], c='red', marker='X', s=200)  # 中心点
plt.title('Manual KMeans Clustering Results', fontsize=16)
plt.xlabel('Feature 1', fontsize=14)
plt.ylabel('Feature 2', fontsize=14)
plt.show()
