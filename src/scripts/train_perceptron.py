import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from src.models.perceptron_numpy import Perceptron

def make_toy(seed=42, n_per_class=100):
    rng = np.random.default_rng(seed)
    mean_pos = np.array([2.0, 2.0])
    mean_neg = np.array([-2.0, -1.5])
    cov = np.array([[1.0, 0.2], [0.2, 1.0]])
    X_pos = rng.multivariate_normal(mean_pos, cov, size=n_per_class)
    X_neg = rng.multivariate_normal(mean_neg, cov, size=n_per_class)
    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    perm = rng.permutation(len(X))
    return X[perm], y[perm]

def plot_boundary(X, y, w, out_path):
    xx = np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200)
    if abs(w[1]) < 1e-8:
        x_vert = -w[2]/w[0] if abs(w[0]) > 1e-8 else 0.0
        plt.figure(figsize=(6,6))
        plt.scatter(X[y==1,0], X[y==1,1], label="+1")
        plt.scatter(X[y==-1,0], X[y==-1,1], label="-1")
        plt.axvline(x_vert, label="decision boundary")
    else:
        yy = -(w[0]/w[1])*xx - (w[2]/w[1])
        plt.figure(figsize=(6,6))
        plt.scatter(X[y==1,0], X[y==1,1], label="+1")
        plt.scatter(X[y==-1,0], X[y==-1,1], label="-1")
        plt.plot(xx, yy, label="decision boundary")
    plt.legend()
    plt.title("Perceptron on Toy Data — Decision Boundary")
    plt.savefig(out_path, bbox_inches="tight")

def main():
    X, y = make_toy()
    p = Perceptron()
    history = p.fit(X, y, epochs=25)

    from pathlib import Path

    # Always resolve from project root
    root = Path(__file__).resolve().parents[2]

    # Define proper subfolders
    out_logs = root / "media/logs"
    out_pngs = root / "media/pngs"

    # Make sure they exist
    out_logs.mkdir(parents=True, exist_ok=True)
    out_pngs.mkdir(parents=True, exist_ok=True)

    # Save outputs to the right place
    pd.DataFrame(history).to_csv(out_logs / "perceptron_training_log.csv", index=False)
    plot_boundary(X, y, p.w, out_pngs / "perceptron_decision_boundary.png")

    print("Final weights:", np.round(p.w, 3))
    print("Final accuracy:", history[-1]["accuracy"])
    print("Saved: media/perceptron_decision_boundary.png, media/perceptron_training_log.csv")

if __name__ == "__main__":
    main()
