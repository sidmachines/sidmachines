import numpy as np

class Perceptron:
    """Online perceptron for labels in {+1, -1}. Bias is included in w[-1]."""
    def __init__(self):
        self.w = None

    def _augment(self, X):
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y, epochs=25, seed=42):
        rng = np.random.default_rng(seed)
        X_aug = self._augment(X)
        self.w = np.zeros(X_aug.shape[1])
        history = []

        for epoch in range(1, epochs + 1):
            perm = rng.permutation(len(X_aug))
            Xs, ys = X_aug[perm], y[perm]
            errors = 0
            for xi, yi in zip(Xs, ys):
                y_hat = 1.0 if (xi @ self.w) >= 0.0 else -1.0
                if yi != y_hat:
                    self.w = self.w + yi * xi
                    errors += 1
            pred = self.predict(X)
            acc = float((pred == y).mean())
            history.append({"epoch": epoch, "errors": int(errors), "accuracy": acc})
        return history

    def decision_function(self, X):
        return self._augment(X) @ self.w

    def predict(self, X):
        s = self.decision_function(X)
        return np.where(s >= 0.0, 1.0, -1.0)
