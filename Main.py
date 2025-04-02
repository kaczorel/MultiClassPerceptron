import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from mlxtend.plotting import plot_decision_regions


class PerceptronMultiClass:
    def __init__(self, eta=0.01, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}

        for cls in self.classes_:
            binary_y = np.where(y == cls, 1, -1)
            model = Perceptron(eta=self.eta, n_iter=self.n_iter)
            model.fit(X, binary_y)
            self.models_[cls] = model

    def predict(self, X):
        scores = {cls: model.net_input(X) for cls, model in self.models_.items()}
        return np.array([max(scores, key=lambda cls: scores[cls][i]) for i in range(len(X))])


class Perceptron:
    def __init__(self, eta=0.01, n_iter=10):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(self.n_iter):
            for xi, target in zip(X, y):
                update = self.eta * (target - self.predict(xi))
                self.w_[1:] += update * xi
                self.w_[0] += update
        return self
    # pewnosc
    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, -1)


class LogisticRegressionMultiClass:
    def __init__(self, eta=0.05, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.models_ = {}

        for cls in self.classes_:
            binary_y = np.where(y == cls, 1, 0)
            model = LogisticRegressionGD(eta=self.eta, n_iter=self.n_iter)
            model.fit(X, binary_y)
            self.models_[cls] = model

    def predict(self, X):
        probs = {cls: model.probability(X) for cls, model in self.models_.items()}
        return np.array([max(probs, key=lambda cls: probs[cls][i]) for i in range(len(X))])

    def probability(self, X):
        probs = {cls: model.probability(X) for cls, model in self.models_.items()}
        return probs


class LogisticRegressionGD:
    def __init__(self, eta=0.05, n_iter=100):
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, X, y):
        self.w_ = np.zeros(1 + X.shape[1])
        for _ in range(self.n_iter):
            output = self.activation(self.net_input(X))
            errors = y - output
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
        return self

    def net_input(self, X):
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, z):
        return 1. / (1. + np.exp(-z))

    def predict(self, X):
        return np.where(self.net_input(X) >= 0.0, 1, 0)

    def probability(self, X):
        return self.activation(self.net_input(X))


iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

#Perceptron
perceptron = PerceptronMultiClass(eta=0.1, n_iter=1000)
perceptron.fit(X_train, y_train)
y_pred = perceptron.predict(X_test)
print("Perceptron accuracy:", np.mean(y_pred == y_test))

#Regression
log_reg = LogisticRegressionMultiClass(eta=0.05, n_iter=1000)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
print("Logistic Regression accuracy:", np.mean(y_pred == y_test))

plot_decision_regions(X_train, y_train, clf=perceptron)
plt.title("Perceptron Decision Boundaries")
plt.show()

plot_decision_regions(X_train, y_train, clf=log_reg)
plt.title("Logistic Regression Decision Boundaries")
plt.show()
