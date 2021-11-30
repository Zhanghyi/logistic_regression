import numpy as np


# 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegressionBinaryClassifier():
    def __init__(self, iterations=10000, learning_rate=0.01, weight_decay=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.weight = None  # 权重 + 偏置 长度是特征个数 + 1
        self.train_loss_history = []
        self.val_loss_history = []
        self.n_feature = 0  # 特征个数
        self.n_samples = 0  # 训练集样本个数
        return

    def train(self, X, y, X_val, y_val):
        self.n_feature = len(X[0])
        self.n_samples = len(X)
        X_1 = np.hstack((X, np.ones((self.n_samples, 1))))
        y = np.array(y)

        X_val_1 = np.hstack((X_val, np.ones((len(X_val), 1))))
        y_val = np.array(y_val)
        if self.weight is None:
            self.weight = np.random.rand(self.n_feature + 1)
        for iteration in range(self.iterations):
            self.weight = self.weight - self.learning_rate * self._gradient(X_1, y)
            self.train_loss_history.append(self._loss(X_1, y))
            self.val_loss_history.append(self._loss(X_val_1, y_val))
        return

    def predict(self, X):
        X_1 = np.hstack((X, np.ones((len(X), 1))))
        return np.round(sigmoid(np.matmul(X_1, self.weight)))

    # loss: -(1/m)*(y * log(sigmoid(wx+b)) + (1-y) * log(1-sigmoid(wx+b))) + 1/2*λw^2
    def _loss(self, X_1, y):
        res = sigmoid(np.matmul(X_1, self.weight))
        res = np.matmul(y.T, np.log(res)) + np.matmul((1 - y).T, np.log(1 - res))
        res = - res / len(X_1)
        res += 0.5 * self.weight_decay * np.sum(np.square(self.weight[:-1]))
        return res

    # gradient: (1/m)*((sigmoid(wx+b) - y) * x) + λw
    def _gradient(self, X_1, y):
        res = np.matmul(X_1, self.weight)
        res = sigmoid(res)
        res = res - y
        res = np.matmul(X_1.T, res)
        res = res / len(X_1)
        res = res + self.weight_decay * np.hstack((self.weight[:-1], np.zeros((1,))))
        return res
