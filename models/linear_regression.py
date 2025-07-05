import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class LinearRegressionX:
    def __init__(self, epochs=10000, alpha=0.01, tol=1e-6, normalize=True, plot_cost=False, l2_penalty=1e-4):
        self.epochs = epochs
        self.alpha = alpha
        self.tol = tol
        self.normalize = normalize
        self.plot_cost = plot_cost
        self.l2_penalty = l2_penalty

        self.costH = []
        self.intercept_ = None
        self.coef_ = None
        self.mean_ = None
        self.std_ = None
        self.w = None
        self.epsilon = 1e-8

    def costFunction(self, X, Y, w):
        m = len(Y)
        predictions = X @ w
        errors = predictions - Y
        reg_term = self.l2_penalty * np.sum(w[1:] ** 2)
        return (np.sum(errors ** 2) + reg_term) / (2 * m)

    def fit_transform(self, X):
        self.mean_ = np.mean(X, axis=0)
        self.std_ = np.std(X, axis=0) + self.epsilon
        return (X - self.mean_) / self.std_

    def transform(self, X):
        if self.mean_ is None or self.std_ is None:
            raise ValueError("Must fit the model before transforming data.")
        return (X - self.mean_) / self.std_

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y).ravel()

        if self.normalize:
            X = self.fit_transform(X)

        X_ = np.c_[np.ones((X.shape[0], 1)), X]
        w = np.zeros(X_.shape[1])

        for i in range(self.epochs):
            predictions = X_ @ w
            errors = predictions - Y
            gradient = (X_.T @ errors) / len(Y)

            gradient[1:] += (self.l2_penalty / len(Y)) * w[1:]

            w -= self.alpha * gradient
            cost = self.costFunction(X_, Y, w)
            self.costH.append(cost)

            if i > 0 and abs(self.costH[-2] - cost) < self.tol:
                if self.plot_cost:
                    print(f"Converged at epoch {i} with cost {cost:.6f}")
                break

        self.w = w
        if self.normalize:
            self.coef_ = w[1:] / self.std_
            self.intercept_ = w[0] - np.sum(self.coef_ * self.mean_)
        else:
            self.intercept_ = w[0]
            self.coef_ = w[1:]

        if self.plot_cost:
            plt.plot(self.costH)
            plt.xlabel("Epoch")
            plt.ylabel("Cost")
            plt.title("Cost History")
            plt.grid(True)
            plt.tight_layout()
            plt.show()

        return self

    def predict(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        index = X.index
        X_values = X.values

        if self.normalize:
            X_values = self.transform(X_values)

        X_ = np.c_[np.ones((X_values.shape[0], 1)), X_values]
        predictions = X_ @ self.w

        return pd.Series(predictions, index=index, name="Prediction")
