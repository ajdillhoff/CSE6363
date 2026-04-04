import numpy as np


class LinearRegression:
    def __init__(self, batch_size=32, regularization=0.0, max_epochs=100, patience=3, lr=0.01, seed=42):
        self.batch_size = int(batch_size)
        self.regularization = float(regularization)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        self.lr = float(lr)
        self.seed = int(seed)
        self.weights = None
        self.bias = None
        self.loss_history = []
        self.val_history = []

    @staticmethod
    def _to_2d_y(y):
        y = np.asarray(y, dtype=float)
        if y.ndim == 1:
            return y.reshape(-1, 1)
        return y

    def fit(self, X, y, batch_size=32, regularization=0.0, max_epochs=100, patience=3, lr=None, seed=None):
        self.batch_size = int(batch_size)
        self.regularization = float(regularization)
        self.max_epochs = int(max_epochs)
        self.patience = int(patience)
        if lr is not None:
            self.lr = float(lr)
        if seed is not None:
            self.seed = int(seed)

        X = np.asarray(X, dtype=float)
        y = self._to_2d_y(y)

        n, d = X.shape
        m = y.shape[1]

        rng = np.random.default_rng(self.seed)
        idx = np.arange(n)
        rng.shuffle(idx)

        val_size = max(1, int(0.1 * n))
        val_idx = idx[:val_size]
        train_idx = idx[val_size:]

        X_val, y_val = X[val_idx], y[val_idx]
        X_train, y_train = X[train_idx], y[train_idx]

        self.weights = rng.normal(0.0, 0.01, size=(d, m))
        self.bias = np.zeros((m,), dtype=float)

        self.loss_history = []
        self.val_history = []

        best_W = self.weights.copy()
        best_b = self.bias.copy()
        best_val_loss = float("inf")
        bad_steps = 0

        for _ in range(self.max_epochs):
            perm = rng.permutation(X_train.shape[0])
            X_train = X_train[perm]
            y_train = y_train[perm]

            for start in range(0, X_train.shape[0], self.batch_size):
                end = start + self.batch_size
                Xb = X_train[start:end]
                yb = y_train[start:end]
                bs = Xb.shape[0]
                if bs == 0:
                    continue

                preds = Xb @ self.weights + self.bias
                err = preds - yb

                mse = np.mean(err ** 2)
                train_loss = mse + self.regularization * np.sum(self.weights ** 2)

                scale = 2.0 / (bs * m)
                grad_W = scale * (Xb.T @ err) + 2.0 * self.regularization * self.weights
                grad_b = scale * np.sum(err, axis=0)

                self.weights -= self.lr * grad_W
                self.bias -= self.lr * grad_b

                val_preds = X_val @ self.weights + self.bias
                val_err = val_preds - y_val
                val_mse = np.mean(val_err ** 2)
                val_loss = val_mse + self.regularization * np.sum(self.weights ** 2)

                self.loss_history.append(train_loss)
                self.val_history.append(val_loss)

                if val_loss < best_val_loss - 1e-12:
                    best_val_loss = val_loss
                    best_W = self.weights.copy()
                    best_b = self.bias.copy()
                    bad_steps = 0
                else:
                    bad_steps += 1
                    if bad_steps >= self.patience:
                        self.weights = best_W
                        self.bias = best_b
                        return self

        self.weights = best_W
        self.bias = best_b
        return self

    def predict(self, X):
        if self.weights is None or self.bias is None:
            raise ValueError("Model is not trained. Call fit() first.")
        X = np.asarray(X, dtype=float)
        y_hat = X @ self.weights + self.bias
        if y_hat.shape[1] == 1:
            return y_hat.reshape(-1)
        return y_hat

    def score(self, X, y):
        X = np.asarray(X, dtype=float)
        y = self._to_2d_y(y)
        y_hat = self.predict(X)
        if np.asarray(y_hat).ndim == 1:
            y_hat = np.asarray(y_hat).reshape(-1, 1)
        return float(np.mean((y - y_hat) ** 2))

    def save(self, path):
        if self.weights is None or self.bias is None:
            raise ValueError("No parameters to save. Train first.")
        np.savez(path, weights=self.weights, bias=self.bias)

    def load(self, path):
        data = np.load(path)
        self.weights = data["weights"]
        self.bias = data["bias"]
        return self
