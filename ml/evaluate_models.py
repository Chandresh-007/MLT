import os
import sys
import json
import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing import preprocess
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "img")
MODEL_FILES = {
    "Logistic Regression": "logistic_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "SVM": "svm_model.pkl",
}

def evaluate_all(X_test, y_test, label_encoder):
    os.makedirs(IMG_DIR, exist_ok=True)
    class_names = list(label_encoder.classes_)
    results = {}
    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        model = joblib.load(path)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
        results[name] = {
            "accuracy": round(acc, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "f1_score": round(f1, 4),
        }
        print(f"\n{name}")
        print(f"  Accuracy : {acc:.4f}")
        print(f"  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}")
        print(f"  F1-score : {f1:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names, ax=ax)
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
    (X_train, X_test, y_train, y_test, feature_names, label_encoder, _, _) = preprocess()
    evaluate_all(X_test, y_test, label_encoder)
    print("\nEvaluation complete.")

if __name__ == "__main__":
    main()
