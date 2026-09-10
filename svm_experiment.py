"""
SVM Experiment: Effect of the penalty parameter C on an RBF-kernel SVM.

Dataset : Iris (versicolor vs virginica), features = petal length, petal width.
Method  : soft-margin SVM with RBF kernel, SVC(kernel="rbf", gamma="scale", C=C).
Main    : compare C = 1, 50, 100, 1000 (only C varies; everything else is fixed).
Extra   : C x gamma grid search with 5-fold cross-validation (Additional Analysis).

Usage:
    python svm_experiment.py            # run the full experiment
    python svm_experiment.py --smoke    # quick smoke test (~20 samples, C=1 only)

Outputs (output/experiment/):
    metrics.csv, support_vectors.csv, confusion_matrices.csv,
    method_comparison.csv, grid_search.csv,
    decision_boundaries.png, metrics_comparison.png, support_vectors.png,
    confusion_matrices.png, method_comparison.png,
    experiment_results.md
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# --------------------------------------------------------------------------- #
# Fixed configuration (control variables)
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "iris" / "iris.data"
OUT_DIR = ROOT / "output" / "experiment"

C_VALUES = [1, 50, 100, 1000]
GAMMA_GRID = [0.01, 0.1, 1, 10]
KERNEL = "rbf"
GAMMA = "scale"
TEST_SIZE = 0.30
RANDOM_STATE = 42

POS_CLASS = "Iris-virginica"     # +1
NEG_CLASS = "Iris-versicolor"    # -1
FEATURE_COLS = [2, 3]            # petal length, petal width (0-indexed in iris.data)
FEATURE_NAMES = ["petal length", "petal width"]


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def load_iris_binary():
    """Load the two target classes and the two petal features from iris.data."""
    cols = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    df = pd.read_csv(DATA_PATH, header=None, names=cols).dropna()
    df = df[df["class"].isin([NEG_CLASS, POS_CLASS])].reset_index(drop=True)

    X = df[["petal_length", "petal_width"]].to_numpy(dtype=float)
    y = np.where(df["class"].to_numpy() == POS_CLASS, 1, -1)
    label_names = df["class"].str.replace("Iris-", "", regex=False).to_numpy()
    return X, y, label_names


# --------------------------------------------------------------------------- #
# Core experiment
# --------------------------------------------------------------------------- #
def run_models(X_train, y_train, X_test, y_test, name_train, c_values):
    """Train one SVC per C value and collect metrics + support-vector counts."""
    metric_rows, sv_rows, cm_rows, models = [], [], [], {}
    for C in c_values:
        clf = SVC(kernel=KERNEL, gamma=GAMMA, C=C, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)
        models[C] = clf

        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)

        metric_rows.append({
            "C": C,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, test_pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_test, test_pred, pos_label=1, zero_division=0),
            "support_vectors": int(clf.support_.shape[0]),
        })

        sv_labels = name_train[clf.support_]
        sv_rows.append({
            "C": C,
            "sv_versicolor": int(np.sum(sv_labels == "versicolor")),
            "sv_virginica": int(np.sum(sv_labels == "virginica")),
            "total_sv": int(clf.support_.shape[0]),
        })

        cm = confusion_matrix(y_test, test_pred, labels=[-1, 1])
        cm_rows.append({
            "C": C,
            "true_versicolor_pred_versicolor": int(cm[0, 0]),
            "true_versicolor_pred_virginica": int(cm[0, 1]),
            "true_virginica_pred_versicolor": int(cm[1, 0]),
            "true_virginica_pred_virginica": int(cm[1, 1]),
        })
    return pd.DataFrame(metric_rows), pd.DataFrame(sv_rows), pd.DataFrame(cm_rows), models


def run_method_comparison(X_train, y_train, X_test, y_test):
    """Compare several SVM variants on the same split."""
    methods = [
        ("Linear SVM", SVC(kernel="linear", C=1, random_state=RANDOM_STATE),
         "linear", "-", "-"),
        ("Polynomial SVM", SVC(kernel="poly", degree=3, gamma=GAMMA, C=1,
                               random_state=RANDOM_STATE),
         "poly", GAMMA, "3"),
        ("RBF SVM", SVC(kernel="rbf", gamma=GAMMA, C=1, random_state=RANDOM_STATE),
         "rbf", GAMMA, "-"),
    ]
    rows = []
    for name, clf, kernel, gamma, degree in methods:
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        rows.append({
            "method": name,
            "kernel": kernel,
            "C": 1,
            "gamma": gamma,
            "degree": degree,
            "train_accuracy": accuracy_score(y_train, train_pred),
            "test_accuracy": accuracy_score(y_test, test_pred),
            "precision": precision_score(y_test, test_pred, pos_label=1, zero_division=0),
            "recall": recall_score(y_test, test_pred, pos_label=1, zero_division=0),
            "f1": f1_score(y_test, test_pred, pos_label=1, zero_division=0),
            "support_vectors": int(clf.support_.shape[0]),
        })
    return pd.DataFrame(rows)


def run_grid_search(X_train, y_train, X_test, y_test):
    """C x gamma grid search with 5-fold CV on the training set (Additional Analysis)."""
    param_grid = {"C": C_VALUES, "gamma": GAMMA_GRID}
    grid = GridSearchCV(
        SVC(kernel=KERNEL, random_state=RANDOM_STATE),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=1,
    )
    grid.fit(X_train, y_train)

    res = grid.cv_results_
    rows = []
    for i in range(len(res["params"])):
        C = res["params"][i]["C"]
        gamma = res["params"][i]["gamma"]
        clf = SVC(kernel=KERNEL, C=C, gamma=gamma, random_state=RANDOM_STATE).fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        rows.append({
            "C": C,
            "gamma": gamma,
            "cv_mean_accuracy": res["mean_test_score"][i],
            "cv_std": res["std_test_score"][i],
            "test_accuracy": accuracy_score(y_test, test_pred),
            "f1": f1_score(y_test, test_pred, pos_label=1, zero_division=0),
        })
    df = pd.DataFrame(rows).sort_values(["C", "gamma"]).reset_index(drop=True)
    return df, grid.best_params_, grid.best_score_


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def plot_decision_boundaries(path, models, X_train, y_train, X_test, y_test):
    cmap_bg = ListedColormap(["#f5c6cb", "#cfe2ff"])   # versicolor / virginica regions
    color_neg, color_pos = "#b2182b", "#2166ac"

    all_x = np.vstack([X_train, X_test])
    x_min, x_max = all_x[:, 0].min() - 0.7, all_x[:, 0].max() + 0.7
    y_min, y_max = all_x[:, 1].min() - 0.7, all_x[:, 1].max() + 0.7
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 400),
                         np.linspace(y_min, y_max, 400))
    grid = np.c_[xx.ravel(), yy.ravel()]

    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    for ax, C in zip(axes.ravel(), sorted(models)):
        clf = models[C]
        Z = clf.predict(grid).reshape(xx.shape)
        ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)
        ax.contour(xx, yy, clf.decision_function(grid).reshape(xx.shape),
                   levels=[-1, 0, 1], colors="k",
                   linestyles=["--", "-", "--"], linewidths=[0.8, 1.4, 0.8])

        for label, color in [(-1, color_neg), (1, color_pos)]:
            m_tr = y_train == label
            ax.scatter(X_train[m_tr, 0], X_train[m_tr, 1], c=color, s=35,
                       edgecolors="white", linewidths=0.5, zorder=3)
            m_te = y_test == label
            ax.scatter(X_test[m_te, 0], X_test[m_te, 1], c=color, s=45,
                       marker="s", edgecolors="black", linewidths=0.6, zorder=3)

        sv = clf.support_vectors_
        ax.scatter(sv[:, 0], sv[:, 1], s=140, facecolors="none",
                   edgecolors="black", linewidths=1.6, zorder=2)

        ax.set_title(f"Decision Boundary of RBF-SVM (C = {C})")
        ax.set_xlabel(f"{FEATURE_NAMES[0]} (standardized)")
        ax.set_ylabel(f"{FEATURE_NAMES[1]} (standardized)")

    fig.suptitle("Decision Boundaries of RBF-SVM under Different C Values", fontsize=14)
    fig.text(0.5, 0.005,
             "Red = versicolor (-1), Blue = virginica (+1); circles = train, "
             "squares = test, black rings = support vectors.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_metrics_comparison(path, metrics_df):
    labels = metrics_df["C"].astype(str).tolist()
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, metrics_df["train_accuracy"], width, label="Train Accuracy", color="#4e79a7")
    ax.bar(x, metrics_df["test_accuracy"], width, label="Test Accuracy", color="#f28e2b")
    ax.bar(x + width, metrics_df["f1"], width, label="F1-score", color="#59a14f")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("Penalty parameter C")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Performance Metrics under Different C Values")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_support_vectors(path, metrics_df):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(metrics_df))
    ax.plot(x, metrics_df["support_vectors"], marker="o", color="#4e79a7", linewidth=2)
    for xi, val in zip(x, metrics_df["support_vectors"]):
        ax.annotate(str(val), (xi, val), textcoords="offset points",
                    xytext=(0, 8), ha="center")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_df["C"].astype(str))
    ax.set_xlabel("Penalty parameter C")
    ax.set_ylabel("Number of support vectors")
    ax.set_title("Number of Support Vectors under Different C Values")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion_matrices(path, cm_df):
    """Plot test-set confusion matrices for the four main C values."""
    fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.3))
    max_count = cm_df[[
        "true_versicolor_pred_versicolor",
        "true_versicolor_pred_virginica",
        "true_virginica_pred_versicolor",
        "true_virginica_pred_virginica",
    ]].to_numpy().max()

    for ax, (_, row) in zip(axes.ravel(), cm_df.iterrows()):
        cm = np.array([
            [row["true_versicolor_pred_versicolor"], row["true_versicolor_pred_virginica"]],
            [row["true_virginica_pred_versicolor"], row["true_virginica_pred_virginica"]],
        ])
        ax.imshow(cm, cmap="Blues", vmin=0, vmax=max_count)
        ax.set_title(f"C = {int(row['C'])}")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["versicolor", "virginica"], rotation=20, ha="right")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["versicolor", "virginica"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > max_count / 2 else "black",
                        fontsize=12, fontweight="bold")

    fig.suptitle("Test-set Confusion Matrices under Different C Values", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_method_comparison(path, method_df):
    labels = method_df["method"].tolist()
    x = np.arange(len(labels))
    width = 0.26

    fig, ax = plt.subplots(figsize=(8.4, 5))
    ax.bar(x - width, method_df["train_accuracy"], width,
           label="Train Accuracy", color="#4e79a7")
    ax.bar(x, method_df["test_accuracy"], width,
           label="Test Accuracy", color="#f28e2b")
    ax.bar(x + width, method_df["f1"], width,
           label="F1-score", color="#59a14f")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title("Comparison of Several SVM Methods")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def write_report(path, metrics_df, sv_df, cm_df, method_df,
                 grid_df, best_params, best_score, gamma_value):
    md = metrics_df.copy()
    for c in ["train_accuracy", "test_accuracy", "precision", "recall", "f1"]:
        md[c] = md[c].map(lambda v: f"{v:.4f}")

    method_md = method_df.copy()
    for c in ["train_accuracy", "test_accuracy", "precision", "recall", "f1"]:
        method_md[c] = method_md[c].map(lambda v: f"{v:.4f}")

    lines = [
        "# SVM Experiment Results",
        "",
        "## Main Settings (fixed control variables)",
        "",
        "- Dataset: Iris dataset (`iris/iris.data`)",
        "- Classes: `versicolor` (-1) and `virginica` (+1)",
        "- Features: petal length and petal width",
        "- Split: 70% train / 30% test, stratified, `random_state = 42`",
        "- Preprocessing: `StandardScaler` fitted on the training set only",
        f'- Model: `SVC(kernel="rbf", gamma="scale", C=C)`, effective gamma = {gamma_value:.6f}',
        "- Compared penalty parameters: C = 1, 50, 100, 1000",
        "",
        "## 3.1 Metrics Table",
        "",
        "| C | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | # Support Vectors |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in md.iterrows():
        lines.append(f"| {r['C']} | {r['train_accuracy']} | {r['test_accuracy']} | "
                     f"{r['precision']} | {r['recall']} | {r['f1']} | {r['support_vectors']} |")

    lines += ["", "## Support Vectors by Class", "",
              "| C | SV in versicolor | SV in virginica | Total SV |",
              "|---:|---:|---:|---:|"]
    for _, r in sv_df.iterrows():
        lines.append(f"| {r['C']} | {r['sv_versicolor']} | {r['sv_virginica']} | {r['total_sv']} |")

    lines += ["", "## Test-set Confusion Matrices", "",
              "| C | True versicolor -> Pred versicolor | True versicolor -> Pred virginica | True virginica -> Pred versicolor | True virginica -> Pred virginica |",
              "|---:|---:|---:|---:|---:|"]
    for _, r in cm_df.iterrows():
        lines.append(f"| {r['C']} | {r['true_versicolor_pred_versicolor']} | "
                     f"{r['true_versicolor_pred_virginica']} | "
                     f"{r['true_virginica_pred_versicolor']} | "
                     f"{r['true_virginica_pred_virginica']} |")

    lines += ["", "## Additional Method Comparison", "",
              "The main experiment focuses on the RBF-kernel SVM and varies C. "
              "To satisfy the method-comparison requirement, the same train-test split "
              "is also evaluated using three SVM variants with C = 1.",
              "",
              "| Method | Kernel | C | Gamma | Degree | Train Accuracy | Test Accuracy | Precision | Recall | F1-score | # Support Vectors |",
              "|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for _, r in method_md.iterrows():
        lines.append(f"| {r['method']} | {r['kernel']} | {r['C']} | {r['gamma']} | "
                     f"{r['degree']} | {r['train_accuracy']} | {r['test_accuracy']} | "
                     f"{r['precision']} | {r['recall']} | {r['f1']} | {r['support_vectors']} |")

    lines += ["", "## 3.2 Figures", "",
              "- `decision_boundaries.png` (Figure 1): four boundary plots for C = 1, 50, 100, 1000.",
              "- `metrics_comparison.png` (Figure 2): train accuracy, test accuracy, F1-score.",
              "- `support_vectors.png` (Figure 3): number of support vectors vs C.",
              "- `confusion_matrices.png` (Figure 4): test-set confusion matrices for the four C values.",
              "- `method_comparison.png` (Figure 5): comparison of linear, polynomial, and RBF SVM.",
              "",
              "## 5. Additional Analysis - C x gamma Grid Search (5-fold CV)",
              "",
              "| C | gamma | CV Mean Accuracy | CV Std | Test Accuracy | F1-score |",
              "|---:|---:|---:|---:|---:|---:|"]
    for _, r in grid_df.iterrows():
        lines.append(f"| {int(r['C'])} | {r['gamma']:g} | {r['cv_mean_accuracy']:.4f} | "
                     f"{r['cv_std']:.4f} | {r['test_accuracy']:.4f} | {r['f1']:.4f} |")
    lines += ["",
              f"Best CV combination: C = {best_params['C']}, gamma = {best_params['gamma']:g} "
              f"(CV accuracy = {best_score:.4f}).",
              "",
              "## Short Discussion",
              "",
              "Only the penalty parameter C varies; the dataset, features, split, scaler, kernel "
              "and gamma are held fixed, so differences below are attributable to C. A larger C "
              "penalizes margin violations more strongly and fits the training set more strictly, "
              "which is reflected in the training accuracy and the support-vector count. Test "
              "metrics should be read directly from the table rather than assumed to improve "
              "monotonically with C; conclusions are limited to this dataset, feature set, split "
              "and RBF-kernel setting.",
              ""]
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Entry points
# --------------------------------------------------------------------------- #
def prepare_data(smoke=False):
    X, y, label_names = load_iris_binary()
    if smoke:
        rng = np.random.default_rng(RANDOM_STATE)
        idx = np.concatenate([
            rng.choice(np.where(y == -1)[0], 10, replace=False),
            rng.choice(np.where(y == 1)[0], 10, replace=False),
        ])
        X, y, label_names = X[idx], y[idx], label_names[idx]

    X_tr, X_te, y_tr, y_te, name_tr, _ = train_test_split(
        X, y, label_names, test_size=TEST_SIZE,
        stratify=y, random_state=RANDOM_STATE,
    )
    scaler = StandardScaler().fit(X_tr)          # fit on training data ONLY
    return scaler.transform(X_tr), y_tr, scaler.transform(X_te), y_te, name_tr


def effective_gamma(X_train):
    # sklearn gamma="scale" == 1 / (n_features * X.var())
    return 1.0 / (X_train.shape[1] * X_train.var())


def smoke_test():
    print("[smoke] running quick pipeline check (~20 samples, C=1)...")
    X_tr, y_tr, X_te, y_te, name_tr = prepare_data(smoke=True)
    metrics_df, sv_df, cm_df, models = run_models(X_tr, y_tr, X_te, y_te, name_tr, [1])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_decision_boundaries(OUT_DIR / "smoke_boundary.png", models, X_tr, y_tr, X_te, y_te)
    plot_confusion_matrices(OUT_DIR / "smoke_confusion_matrix.png", cm_df)
    r = metrics_df.iloc[0]
    print(f"[smoke] train_acc={r['train_accuracy']:.4f} test_acc={r['test_accuracy']:.4f} "
          f"f1={r['f1']:.4f} sv={r['support_vectors']}")
    print("[smoke] OK - pipeline runs end to end. See output/experiment/smoke_boundary.png")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_tr, y_tr, X_te, y_te, name_tr = prepare_data(smoke=False)
    gamma_value = effective_gamma(X_tr)

    metrics_df, sv_df, cm_df, models = run_models(X_tr, y_tr, X_te, y_te, name_tr, C_VALUES)
    method_df = run_method_comparison(X_tr, y_tr, X_te, y_te)
    grid_df, best_params, best_score = run_grid_search(X_tr, y_tr, X_te, y_te)

    metrics_df.to_csv(OUT_DIR / "metrics.csv", index=False)
    sv_df.to_csv(OUT_DIR / "support_vectors.csv", index=False)
    cm_df.to_csv(OUT_DIR / "confusion_matrices.csv", index=False)
    method_df.to_csv(OUT_DIR / "method_comparison.csv", index=False)
    grid_df.to_csv(OUT_DIR / "grid_search.csv", index=False)

    plot_decision_boundaries(OUT_DIR / "decision_boundaries.png", models, X_tr, y_tr, X_te, y_te)
    plot_metrics_comparison(OUT_DIR / "metrics_comparison.png", metrics_df)
    plot_support_vectors(OUT_DIR / "support_vectors.png", metrics_df)
    plot_confusion_matrices(OUT_DIR / "confusion_matrices.png", cm_df)
    plot_method_comparison(OUT_DIR / "method_comparison.png", method_df)
    write_report(OUT_DIR / "experiment_results.md", metrics_df, sv_df, cm_df, method_df,
                 grid_df, best_params, best_score, gamma_value)

    print("Experiment completed.")
    print(f"Output directory: {OUT_DIR}")
    print(f"Effective gamma (scale) = {gamma_value:.6f}\n")
    print(metrics_df.to_string(index=False))
    print(f"\nBest grid combination: {best_params} (CV acc = {best_score:.4f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RBF-SVM C comparison on Iris.")
    parser.add_argument("--smoke", action="store_true", help="run a quick smoke test only")
    args = parser.parse_args()
    smoke_test() if args.smoke else main()
