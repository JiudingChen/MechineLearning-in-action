from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# 加载Iris数据集
iris = datasets.load_iris()
X = iris.data
y = iris.target

# 将数据集分割为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 定义具有不同核函数的SVM分类器
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
f1_scores = []

for kernel in kernels:
    # 创建并训练SVM分类器
    svm_classifier = SVC(kernel=kernel, gamma='scale')  # 使用'scale'作为gamma的默认值
    svm_classifier.fit(X_train, y_train)

    # 在测试集上进行预测
    y_pred = svm_classifier.predict(X_test)

    # 计算并记录F1分数
    f1 = f1_score(y_test, y_pred, average='weighted')  # 用加权平均方式计算多类分类的F1分数
    f1_scores.append(f1)

# 绘制不同核函数F1分数的比较图
plt.figure(figsize=(10, 6))
plt.bar(kernels, f1_scores, color='skyblue')
plt.xlabel('Kernel Function')
plt.ylabel('Weighted F1 Score')
plt.title('Comparison of SVM Kernel Functions on Iris Dataset')
plt.ylim([min(f1_scores) - 0.1, 1.0])  # 设置y轴限制以清晰显示差异
plt.show()

# 将F1分数以字典形式输出，便于访问
kernel_f1_dict = dict(zip(kernels, f1_scores))
print(kernel_f1_dict)
