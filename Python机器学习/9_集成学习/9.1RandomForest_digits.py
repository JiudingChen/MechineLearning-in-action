from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据集
digits = load_digits()

# 分割数据集为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(digits.data, digits.target, test_size=0.25, random_state=42)

# 初始化随机森林分类器
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 训练模型
rf.fit(X_train, y_train)

# 预测测试集
y_pred = rf.predict(X_test)

# 计算准确率
accuracy = metrics.accuracy_score(y_test, y_pred)

# 可视化混淆矩阵
conf_matrix = metrics.confusion_matrix(y_test, y_pred)

# 使用Seaborn绘制混淆矩阵的热图
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix Visualization')

# 可视化数字示例
# 显示测试集中的前10个数字图像及其预测标签
_, axes = plt.subplots(nrows=1, ncols=10, figsize=(16, 4))
for ax, image, prediction in zip(axes, X_test, y_pred):
    ax.set_axis_off()
    image = image.reshape(8, 8)
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title(f'Pred: {prediction}')

# 打印准确率
print(f'Accuracy: {accuracy:.2f}')

# 显示图表
plt.show()
