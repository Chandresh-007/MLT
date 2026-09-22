"""
MindfulTech – Synthetic Dataset Generator
==========================================
Generates a reproducible synthetic dataset for the digital-wellbeing
classification task.

IMPORTANT: This dataset is SYNTHETIC and created for educational /
academic demonstration purposes only.  It must NOT be presented as
real-world survey data.

Methodology
-----------
1. 13 primary features are sampled from bounded uniform / integer
   distributions that mimic plausible ranges of self-reported digital
   habits (see `generate_features`).
2. 6 derived features are computed deterministically from the primary
   features (see `add_derived_features`).
3. A **Digital Wellbeing Risk Score** is calculated from weighted
   behavioural thresholds (see `_raw_risk_score`).  A small amount of
   Gaussian noise (σ = 1.5) is added to prevent the ML model from
   perfectly memorising the rule set.
4. The noisy score is discretised into three classes:
        score >= 12  →  High
        score >= 6   →  Moderate
        score <  6   →  Low
5. A continuous **Wellbeing Score** (0-100) is also computed for the
   optional regression target.
"""

import os
import numpy as np
import pandas as pd

# --------------- configuration ---------------
NUM_ROWS = 2000
RANDOM_SEED = 42
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "dataset.csv")

np.random.seed(RANDOM_SEED)


# --------------- primary features ---------------

def generate_features(n: int) -> pd.DataFrame:
    """Sample 13 primary digital-habit features."""
    data = {
        "age":            np.random.randint(16, 45, size=n),
        "screen_time":    np.round(np.random.uniform(1, 16, size=n), 1),
        "social_media":   np.round(np.random.uniform(0, 8, size=n), 1),
        "gaming":         np.round(np.random.uniform(0, 6, size=n), 1),
        "short_video":    np.round(np.random.uniform(0, 6, size=n), 1),
        "phone_unlocks":  np.random.randint(10, 200, size=n),
        "notifications":  np.random.randint(20, 350, size=n),
        "night_usage":    np.round(np.random.uniform(0, 5, size=n), 1),
        "sleep":          np.round(np.random.uniform(3, 10, size=n), 1),
        "focus_time":     np.round(np.random.uniform(0, 8, size=n), 1),
        "exercise":       np.random.randint(0, 120, size=n),
        "stress":         np.random.randint(1, 11, size=n),
        "productivity":   np.random.randint(1, 11, size=n),
    }
    return pd.DataFrame(data)


# --------------- derived features ---------------

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 6 derived features from the primary ones.

    Feature Engineering Rationale
    -----------------------------
    entertainment_hours            – Total time in gaming + short-video;
                                     captures recreational digital consumption.
    total_recreational_screen_time – social_media + gaming + short_video;
                                     broader recreational usage indicator.
    night_usage_ratio              – Proportion of screen time occurring at
                                     night (0-1, clamped); higher values
                                     indicate more late-night usage.
    focus_to_screen_ratio          – Proportion of screen time spent on
                                     focused / productive activities (0-1);
                                     lower values suggest more passive usage.
    sleep_deficit_indicator        – How far below 7.5 h the user sleeps
                                     (0 = no deficit); captures sleep debt.
    usage_intensity                – Composite of phone unlocks and
                                     notification volume; reflects
                                     distraction / dependency signals.
    """
    df = df.copy()
    df["entertainment_hours"] = np.round(df["gaming"] + df["short_video"], 1)
    df["total_recreational_screen_time"] = np.round(
        df["social_media"] + df["gaming"] + df["short_video"], 1
    )
    df["night_usage_ratio"] = np.round(
        np.clip(df["night_usage"] / df["screen_time"].replace(0, 1), 0, 1), 3
    )
    df["focus_to_screen_ratio"] = np.round(
        np.clip(df["focus_time"] / df["screen_time"].replace(0, 1), 0, 1), 3
    )
    df["sleep_deficit_indicator"] = np.round(
        np.clip(7.5 - df["sleep"], 0, None), 1
    )
    df["usage_intensity"] = np.round(
        df["phone_unlocks"] + df["notifications"] / 10.0, 1
    )
    return df


# --------------- target labels ---------------

def _raw_risk_score(row: pd.Series) -> float:
    """
    Compute a Digital Wellbeing Risk Score using weighted thresholds.

    Each factor contributes 0-3 points.  The total possible score is
    roughly 0-24.  See the table below for the threshold rationale.

    Factor                | Threshold     | Points
    ----------------------|---------------|-------
    screen_time > 10 h    | Very high     | +3
    screen_time > 6 h     | Above-average | +1.5
    social_media > 5 h    | Excessive     | +2.5
    social_media > 3 h    | High          | +1
    gaming > 4 h          | Heavy         | +2
    gaming > 2 h          | Moderate      | +0.8
    short_video > 4 h     | Heavy         | +2
    short_video > 2 h     | Moderate      | +0.8
    phone_unlocks > 120   | Compulsive    | +2
    phone_unlocks > 70    | Frequent      | +1
    notifications > 250   | Overwhelming  | +1.5
    notifications > 150   | High          | +0.5
    night_usage > 3 h     | Excessive     | +2.5
    night_usage > 1.5 h   | Elevated      | +1
    sleep < 5 h           | Very low      | +2.5
    sleep < 6.5 h         | Below target  | +1
    focus_time < 1.5 h    | Very low      | +2
    focus_time < 3 h      | Low           | +0.8
    exercise < 15 min     | Sedentary     | +1.5
    exercise < 30 min     | Minimal       | +0.5
    stress > 7            | High          | +2
    stress > 5            | Moderate      | +0.8
    productivity < 4      | Low           | +2
    productivity < 6      | Below-average | +0.8
    """
    score = 0.0

    # Screen time
    if row["screen_time"] > 10:    score += 3
    elif row["screen_time"] > 6:   score += 1.5

    # Social media
    if row["social_media"] > 5:    score += 2.5
    elif row["social_media"] > 3:  score += 1

    # Gaming
    if row["gaming"] > 4:          score += 2
    elif row["gaming"] > 2:        score += 0.8

    # Short videos
    if row["short_video"] > 4:     score += 2
    elif row["short_video"] > 2:   score += 0.8

    # Phone unlocks
    if row["phone_unlocks"] > 120: score += 2
    elif row["phone_unlocks"] > 70: score += 1

    # Notifications
    if row["notifications"] > 250: score += 1.5
    elif row["notifications"] > 150: score += 0.5

    # Night usage
    if row["night_usage"] > 3:     score += 2.5
    elif row["night_usage"] > 1.5: score += 1

    # Sleep
    if row["sleep"] < 5:           score += 2.5
    elif row["sleep"] < 6.5:       score += 1

    # Focus time
    if row["focus_time"] < 1.5:    score += 2
    elif row["focus_time"] < 3:    score += 0.8

    # Exercise
    if row["exercise"] < 15:       score += 1.5
    elif row["exercise"] < 30:     score += 0.5

    # Stress
    if row["stress"] > 7:          score += 2
    elif row["stress"] > 5:        score += 0.8

    # Productivity
    if row["productivity"] < 4:    score += 2
    elif row["productivity"] < 6:  score += 0.8

    return score


def derive_risk_label(row: pd.Series) -> str:
    """
    Derive the risk class by adding Gaussian noise (σ=1.5) to the raw
    risk score before applying thresholds.  This prevents the ML model
    from perfectly memorising the deterministic rules and introduces
    realistic variation.
    """
    raw = _raw_risk_score(row)
    noisy = raw + np.random.normal(0, 2.0)

    if noisy >= 13:
        return "High"
    elif noisy >= 7:
        return "Moderate"
    else:
        return "Low"


def derive_wellbeing_score(row: pd.Series) -> float:
    """Continuous 0-100 wellbeing indicator (higher is better)."""
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


# --------------- main ---------------

def main():
    print("=" * 60)
    print("MindfulTech - Synthetic Dataset Generator")
    print("=" * 60)
    print(f"\nRandom seed : {RANDOM_SEED}")
    print(f"Rows        : {NUM_ROWS}")

    print("\n[1/4] Generating primary features ...")
    df = generate_features(NUM_ROWS)

    print("[2/4] Computing derived features ...")
    df = add_derived_features(df)

    print("[3/4] Deriving target labels ...")
    df["risk_level"] = df.apply(derive_risk_label, axis=1)
    df["wellbeing_score"] = df.apply(derive_wellbeing_score, axis=1)

    print("\nRisk-level distribution:")
    print(df["risk_level"].value_counts().to_string())
    print(f"\nWellbeing-score range: {df['wellbeing_score'].min()}"
          f" to {df['wellbeing_score'].max()}")

    print("\n[4/4] Saving dataset ...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Dataset saved to {OUTPUT_FILE}  ({len(df)} rows, "
          f"{len(df.columns)} columns)")


if __name__ == "__main__":
    main()
