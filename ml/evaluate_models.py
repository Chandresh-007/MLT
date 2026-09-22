"""
MindfulTech – Model Evaluation
===============================
Evaluates all trained classifiers on the held-out test set and performs
5-fold stratified cross-validation.

Outputs:
  - model_metrics.json      (accuracy, precision, recall, F1, cross-val)
  - confusion matrix PNGs   (one per classifier)
  - model_comparison.png    (grouped bar chart)
"""

import os
import sys
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix,
)
from sklearn.model_selection import cross_val_score, StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing import preprocess, FEATURE_COLS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")

MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
}


def evaluate_all(X_train, X_test, y_train, y_test, label_encoder):
    """Evaluate every classifier and return metrics dict."""
    os.makedirs(IMG_DIR, exist_ok=True)
    class_names = list(label_encoder.classes_)
    results = {}
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        model = joblib.load(path)

        # --- test-set metrics ---
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec_w = precision_score(y_test, y_pred, average="weighted",
                                 zero_division=0)
        rec_w = recall_score(y_test, y_pred, average="weighted",
                             zero_division=0)
        f1_w = f1_score(y_test, y_pred, average="weighted",
                        zero_division=0)
        prec_m = precision_score(y_test, y_pred, average="macro",
                                 zero_division=0)
        rec_m = recall_score(y_test, y_pred, average="macro",
                             zero_division=0)
        f1_m = f1_score(y_test, y_pred, average="macro",
                        zero_division=0)

        # --- 5-fold cross-validation ---
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv,
                                    scoring="accuracy")

        results[name] = {
            "accuracy": round(acc, 4),
            "precision": round(prec_w, 4),
            "recall": round(rec_w, 4),
            "f1_score": round(f1_w, 4),
            "precision_macro": round(prec_m, 4),
            "recall_macro": round(rec_m, 4),
            "f1_score_macro": round(f1_m, 4),
            "cv_mean": round(float(cv_scores.mean()), 4),
            "cv_std": round(float(cv_scores.std()), 4),
            "cv_scores": [round(float(s), 4) for s in cv_scores],
        }

        print(f"\n{name}")
        print(f"  Accuracy       : {acc:.4f}")
        print(f"  Precision (w)  : {prec_w:.4f}")
        print(f"  Recall    (w)  : {rec_w:.4f}")
        print(f"  F1-score  (w)  : {f1_w:.4f}")
        print(f"  Precision (m)  : {prec_m:.4f}")
        print(f"  Recall    (m)  : {rec_m:.4f}")
        print(f"  F1-score  (m)  : {f1_m:.4f}")
        print(f"  CV accuracy    : {cv_scores.mean():.4f} "
              f"+/- {cv_scores.std():.4f}")

        # --- confusion matrix ---
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names, ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name} - Confusion Matrix")
        fig.tight_layout()
        img_name = fname.replace(".pkl", "_cm.png")
        fig.savefig(os.path.join(IMG_DIR, img_name), dpi=120)
        plt.close(fig)
        print(f"  Confusion matrix saved -> static/img/{img_name}")

    _plot_comparison(results)

    metrics_path = os.path.join(MODEL_DIR, "model_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nMetrics saved -> {metrics_path}")
    return results


def _plot_comparison(results: dict):
    model_names = list(results.keys())
    metrics = ["accuracy", "precision", "recall", "f1_score"]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(model_names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (m, label) in enumerate(zip(metrics, labels)):
        values = [results[mn][m] for mn in model_names]
        ax.bar(x + i * width, values, width, label=label)

    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=120)
    plt.close(fig)
    print("  Model comparison chart saved -> static/img/model_comparison.png")


def main():
    print("=" * 60)
    print("MindfulTech - Model Evaluation")
    print("=" * 60)

    (X_train, X_test, y_train, y_test,
     feature_names, label_encoder, _, _) = preprocess()
    evaluate_all(X_train, X_test, y_train, y_test, label_encoder)
    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
