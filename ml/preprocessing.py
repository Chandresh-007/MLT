import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODEL_DIR = os.path.join(BASE_DIR, "models")
DATASET_PATH = os.path.join(DATA_DIR, "dataset.csv")
PROCESSED_PATH = os.path.join(DATA_DIR, "processed_data.csv")
SCALER_PATH = os.path.join(MODEL_DIR, "scaler.pkl")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.pkl")
FEATURE_COLS = [
    "age", "screen_time", "social_media", "gaming", "short_video",
    "phone_unlocks", "notifications", "night_usage", "sleep",
    "focus_time", "exercise", "stress", "productivity",
]
TARGET_COL = "risk_level"
WELLBEING_COL = "wellbeing_score"
VALID_RANGES = {
    "age": (10, 80), "screen_time": (0, 24), "social_media": (0, 18),
    "gaming": (0, 18), "short_video": (0, 18), "phone_unlocks": (0, 500),
    "notifications": (0, 1000), "night_usage": (0, 10), "sleep": (0, 16),
    "focus_time": (0, 18), "exercise": (0, 300), "stress": (1, 10),
    "productivity": (1, 10),
}

def load_data(path: str = DATASET_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows from {path}")
    return df

def handle_missing(df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df[col].isnull().sum() > 0:
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"  Filled {col} missing values with median ({median_val})")
    return df

def validate_and_clip(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)
    print("  Value ranges validated and clipped.")
    return df

def preprocess(test_size: float = 0.2, random_state: int = 42):
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_data()
    print("Handling missing values ...")
    df = handle_missing(df)
    print("Validating value ranges ...")
    df = validate_and_clip(df)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Processed data saved to {PROCESSED_PATH}")
    X = df[FEATURE_COLS].values
    y_raw = df[TARGET_COL].values
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    joblib.dump(le, LABEL_ENCODER_PATH)
    print(f"Label encoder saved ({dict(zip(le.classes_, le.transform(le.classes_)))})")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, SCALER_PATH)
    print("Scaler fitted and saved.")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")
    wb_train, wb_test = None, None
    if WELLBEING_COL in df.columns:
        wb = df[WELLBEING_COL].values
        _, _, wb_train, wb_test = train_test_split(
            X_scaled, wb, test_size=test_size, random_state=random_state
        )
    return (X_train, X_test, y_train, y_test, FEATURE_COLS, le, wb_train, wb_test)

if __name__ == "__main__":
    results = preprocess()
    print("\nPreprocessing complete")
