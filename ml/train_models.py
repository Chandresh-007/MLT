import os
import sys
import json
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ml.preprocessing import preprocess, FEATURE_COLS
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

def train_classifiers(X_train, y_train):
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=8, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }
    trained = {}
    for name, model in models.items():
        print(f"  Training {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained

def train_kmeans(X_train, k=4):
    print(f"  Training K-Means (k={k}) ...")
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_train)
    return km

def train_regression(X_train, y_train_wb):
    if y_train_wb is None:
        print("  Skipping regression (no wellbeing_score column).")
        return None
    print("  Training Linear Regression (wellbeing score) ...")
    lr = LinearRegression()
    lr.fit(X_train, y_train_wb)
    return lr

def describe_clusters(km_model, feature_names, scaler):
    centroids_scaled = km_model.cluster_centers_
    centroids = scaler.inverse_transform(centroids_scaled)
    descriptions = {}
    for i, centroid in enumerate(centroids):
        profile = dict(zip(feature_names, np.round(centroid, 1)))
        label_parts = []
        if profile["screen_time"] > 10:
            label_parts.append("High Screen-Time")
        elif profile["screen_time"] < 5:
            label_parts.append("Low Screen-Time")
        if profile["gaming"] > 3 or profile["short_video"] > 3:
            label_parts.append("Entertainment-Heavy")
        if profile["focus_time"] < 2:
            label_parts.append("Low Focus")
        elif profile["focus_time"] > 5:
            label_parts.append("High Focus")
        if profile["phone_unlocks"] > 100:
            label_parts.append("High Distraction")
        if profile["sleep"] > 7 and profile["exercise"] > 40:
            label_parts.append("Balanced Lifestyle")
        if profile["social_media"] > 4:
            label_parts.append("Social-Media Heavy")
        if not label_parts:
            label_parts.append("Balanced Digital User")
        descriptions[i] = {"label": " / ".join(label_parts), "profile": profile}
    return descriptions

def save_models(classifiers, km_model, reg_model, cluster_desc):
    os.makedirs(MODEL_DIR, exist_ok=True)
    file_map = {
        "Logistic Regression": "logistic_model.pkl",
        "Decision Tree": "decision_tree_model.pkl",
        "Random Forest": "random_forest_model.pkl",
        "SVM": "svm_model.pkl",
    }
    for name, model in classifiers.items():
        path = os.path.join(MODEL_DIR, file_map[name])
        joblib.dump(model, path)
        print(f"  Saved {name} -> {path}")
    km_path = os.path.join(MODEL_DIR, "kmeans_model.pkl")
    joblib.dump(km_model, km_path)
    print(f"  Saved K-Means -> {km_path}")
    desc_path = os.path.join(MODEL_DIR, "cluster_descriptions.json")
    serialisable = {}
    for k, v in cluster_desc.items():
        serialisable[str(k)] = {
            "label": v["label"],
            "profile": {fk: float(fv) for fk, fv in v["profile"].items()},
        }
    with open(desc_path, "w") as f:
        json.dump(serialisable, f, indent=2)
    print(f"  Saved cluster descriptions -> {desc_path}")
    if reg_model is not None:
        reg_path = os.path.join(MODEL_DIR, "linear_regression_model.pkl")
        joblib.dump(reg_model, reg_path)
        print(f"  Saved Linear Regression -> {reg_path}")

def main():
    print("=" * 60)
    print("MindfulTech - Model Training")
    print("=" * 60)
    print("\n[1/4] Preprocessing ...")
    (X_train, X_test, y_train, y_test, feature_names, label_encoder, wb_train, wb_test) = preprocess()
    print("\n[2/4] Training classifiers ...")
    classifiers = train_classifiers(X_train, y_train)
    print("\n[3/4] Training K-Means clustering ...")
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
    km_model = train_kmeans(X_train, k=4)
    cluster_desc = describe_clusters(km_model, feature_names, scaler)
    print("\n[4/4] Training regression model ...")
    reg_model = train_regression(X_train, wb_train)
    print("\nSaving models ...")
    save_models(classifiers, km_model, reg_model, cluster_desc)
    print("\nAll models trained and saved.")
    return (classifiers, X_test, y_test, label_encoder, km_model, reg_model, wb_test)

if __name__ == "__main__":
    main()
