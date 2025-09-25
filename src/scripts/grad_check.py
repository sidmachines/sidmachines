import numpy as np

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_loss_and_grad(w, X, y):
    # Forward pass
    z = X @ w
    p = sigmoid(z)

    # Loss: binary cross-entropy
    eps = 1e-9
    loss = -np.mean(y*np.log(p+eps) + (1-y)*np.log(1-p+eps))

    # Gradient
    grad = X.T @ (p - y) / X.shape[0]
    return loss, grad

def numerical_grad(f, w, eps=1e-5):
    g = np.zeros_like(w)
    for i in range(len(w)):
        w_pos = w.copy(); w_pos[i] += eps
        w_neg = w.copy(); w_neg[i] -= eps
        g[i] = (f(w_pos) - f(w_neg)) / (2*eps)
    return g

def main():
    rng = np.random.default_rng(0)
    n, d = 50, 3   # 50 samples, 3 features
    X = rng.normal(size=(n, d))
    true_w = rng.normal(size=(d,))
    y = (X @ true_w + 0.1*rng.normal(size=n) > 0).astype(float)

    w = rng.normal(size=(d,))

    # Analytic gradient
    loss, grad = logistic_loss_and_grad(w, X, y)

    # Numeric gradient
    def loss_only(wv): return logistic_loss_and_grad(wv, X, y)[0]
    numg = numerical_grad(loss_only, w)

    print("Analytic grad:", np.round(grad, 4))
    print("Numeric  grad:", np.round(numg, 4))
    print("Max abs diff:", float(np.max(np.abs(grad - numg))))

if __name__ == "__main__":
    main()
