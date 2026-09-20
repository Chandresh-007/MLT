import os
import numpy as np
import pandas as pd
NUM_ROWS = 1500
RANDOM_SEED = 42
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")
np.random.seed(RANDOM_SEED)

def generate_features(n: int) -> pd.DataFrame:
    data = {
        "age": np.random.randint(16, 45, size=n),
        "screen_time": np.round(np.random.uniform(1, 16, size=n), 1),
        "social_media": np.round(np.random.uniform(0, 8, size=n), 1),
        "gaming": np.round(np.random.uniform(0, 6, size=n), 1),
        "short_video": np.round(np.random.uniform(0, 6, size=n), 1),
        "phone_unlocks": np.random.randint(10, 200, size=n),
        "notifications": np.random.randint(20, 350, size=n),
        "night_usage": np.round(np.random.uniform(0, 5, size=n), 1),
        "sleep": np.round(np.random.uniform(3, 10, size=n), 1),
        "focus_time": np.round(np.random.uniform(0, 8, size=n), 1),
        "exercise": np.random.randint(0, 120, size=n),
        "stress": np.random.randint(1, 11, size=n),
        "productivity": np.random.randint(1, 11, size=n),
    }
    return pd.DataFrame(data)

def derive_risk_label(row: pd.Series) -> str:
    score = 0.0
    if row["screen_time"] > 10: score += 3
    elif row["screen_time"] > 6: score += 1.5
    if row["social_media"] > 5: score += 2.5
    elif row["social_media"] > 3: score += 1
    if row["gaming"] > 4: score += 2
    elif row["gaming"] > 2: score += 0.8
    if row["short_video"] > 4: score += 2
    elif row["short_video"] > 2: score += 0.8
    if row["phone_unlocks"] > 120: score += 2
    elif row["phone_unlocks"] > 70: score += 1
    if row["notifications"] > 250: score += 1.5
    elif row["notifications"] > 150: score += 0.5
    if row["night_usage"] > 3: score += 2.5
    elif row["night_usage"] > 1.5: score += 1
    if row["sleep"] < 5: score += 2.5
    elif row["sleep"] < 6.5: score += 1
    if row["focus_time"] < 1.5: score += 2
    elif row["focus_time"] < 3: score += 0.8
    if row["exercise"] < 15: score += 1.5
    elif row["exercise"] < 30: score += 0.5
    if row["stress"] > 7: score += 2
    elif row["stress"] > 5: score += 0.8
    if row["productivity"] < 4: score += 2
    elif row["productivity"] < 6: score += 0.8
    if score >= 12: return "High"
    elif score >= 6: return "Moderate"
    else: return "Low"

def derive_wellbeing_score(row: pd.Series) -> float:
    score = 50.0
    score += min(row["sleep"], 8) * 2
    score += min(row["focus_time"], 6) * 2
    score += min(row["exercise"], 60) * 0.15
    score += row["productivity"] * 1.5
    score -= max(row["screen_time"] - 4, 0) * 1.5
    score -= max(row["social_media"] - 2, 0) * 1.5
    score -= max(row["night_usage"] - 1, 0) * 2
    score -= max(row["stress"] - 4, 0) * 2
    score -= max(row["phone_unlocks"] - 50, 0) * 0.05
    return round(max(0, min(100, score)), 1)

def main():
    print("Generating synthetic dataset ...")
    df = generate_features(NUM_ROWS)
    df["risk_level"] = df.apply(derive_risk_label, axis=1)
    df["wellbeing_score"] = df.apply(derive_wellbeing_score, axis=1)
    print("\nRisk-level distribution:")
    print(df["risk_level"].value_counts())
    print(f"\nWellbeing-score range: {df['wellbeing_score'].min()} - {df['wellbeing_score'].max()}")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nDataset saved to {OUTPUT_FILE}  ({len(df)} rows)")

if __name__ == "__main__":
    main()
