import numpy as np


# 1/(1+e^-x)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


class LogisticRegressionBinaryClassifier():
    def __init__(self, iterations=10000, learning_rate=0.01):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.weight = None  # 权重 + 偏置 长度是特征个数 + 1
        self.loss_history = []
        self.n_feature = 0  # 特征个数
        self.n_samples = 0  # 训练集样本个数
        return

    def train(self, X, y):
        self.n_feature = len(X[0])
        self.n_samples = len(X)
        X_with_bias = np.hstack((X, np.ones((self.n_samples, 1))))
        y = np.array(y)
        if self.weight is None:
            self.weight = np.random.rand(self.n_feature + 1)
        for iteration in range(self.iterations):
            self.weight = self.weight - self.learning_rate * self._gradient(X_with_bias, y)
            self.loss_history.append(self._loss(X_with_bias, y))
        return

    def predict(self, X):
        X_with_bias = np.hstack((X, np.ones((len(X), 1))))
        return np.round(sigmoid(np.dot(X_with_bias, self.weight)))

    # loss: (1/m)*(y * log(sigmoid(wx+b)) + (1-y) * log(1-sigmoid(wx+b)))
    def _loss(self, X, y):
        res = sigmoid(np.matmul(X, self.weight))
        res = np.matmul(y.T, np.log(res)) + np.matmul((1 - y).T, np.log(1 - res))
        res = - res / self.n_samples
        return res

    # gradient: (1/m)*((sigmoid(wx+b) - y) * x)
    def _gradient(self, X, y):
        res = np.matmul(X, self.weight)
        res = sigmoid(res)
        res = res - y
        res = np.matmul(X.T, res)
        res = res / self.n_samples
        return res
