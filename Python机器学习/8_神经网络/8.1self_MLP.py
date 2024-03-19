import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 数据加载和预处理
digits = load_digits()
X, y = digits.data, digits.target

# 标准化特征值
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


# 将标签转换为 one-hot 编码
def one_hot(y, num_classes):
    return np.eye(num_classes)[y]


num_classes = len(np.unique(y))
y_train_one_hot = one_hot(y_train, num_classes)
y_test_one_hot = one_hot(y_test, num_classes)


# 神经网络类
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        self.b2 = np.zeros((1, output_size))

    def forward(self, X):
        # 前向传播
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = np.tanh(self.Z1)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = self.softmax(self.Z2)
        return self.A2

    @staticmethod
    def softmax(z):
        # Softmax 函数
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def compute_loss(self, Y_pred, Y_true):
        # 计算交叉熵损失
        m = Y_true.shape[0]
        loss = -np.sum(Y_true * np.log(Y_pred + 1e-8)) / m
        return loss

    def backward(self, X, Y):
        # 反向传播算法
        m = X.shape[0]

        # 输出层梯度
        dZ2 = self.A2 - Y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        # 隐藏层梯度
        dZ1 = np.dot(dZ2, self.W2.T) * (1 - np.power(self.A1, 2))
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        return dW1, db1, dW2, db2

    def update_params(self, dW1, db1, dW2, db2, learning_rate):
        # 更新权重和偏置
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2


# 创建神经网络实例
input_size = 64  # 输入层大小（特征数）
hidden_size = 32  # 隐藏层大小
output_size = 10  # 输出层大小（类别数）
nn = NeuralNetwork(input_size, hidden_size, output_size)


# 训练函数
def train(nn, X_train, y_train, X_test, y_test, epochs, learning_rate):
    loss_history = []
    accuracy_history = []

    for epoch in range(epochs):
        # 前向传播
        Y_pred = nn.forward(X_train)

        # 计算损失
        loss = nn.compute_loss(Y_pred, y_train)
        loss_history.append(loss)

        # 反向传播和参数更新
        dW1, db1, dW2, db2 = nn.backward(X_train, y_train)
        nn.update_params(dW1, db1, dW2, db2, learning_rate)

        # 计算准确率
        accuracy = compute_accuracy(Y_pred, y_train)
        accuracy_history.append(accuracy)

        # 打印训练进度
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Loss: {loss:.4f} - Accuracy: {accuracy:.4f}")

    test_accuracy = compute_accuracy(nn.forward(X_test), y_test)
    print(f"Test Accuracy: {test_accuracy:.4f}")

    return loss_history, accuracy_history


def compute_accuracy(Y_pred, Y_true):
    predictions = np.argmax(Y_pred, axis=1)
    labels = np.argmax(Y_true, axis=1)
    return np.mean(predictions == labels)


# 训练参数
epochs = 100
learning_rate = 0.1

# 训练模型
loss_history, accuracy_history = train(nn, X_train, y_train_one_hot, X_test, y_test_one_hot, epochs, learning_rate)

# 可视化损失和准确率
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(loss_history, label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(accuracy_history, label='Accuracy', color='orange')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.tight_layout()
plt.show()
