"""
MindfulTech – Recommendation Engine
=====================================
Generates non-medical, general wellbeing suggestions based on the
user's actual input values and detected behavioural patterns.

Architecture
------------
1. Rule-based recommendations (always available)
2. Template-based natural-language explanation (always available)
3. Optional AWS Bedrock explanation layer (only if configured)

The application works completely without AWS Bedrock.
"""

import os

# --------------- rule-based recommendations ---------------

def generate_recommendations(user_input: dict, patterns: list[str]) -> list[str]:
    """
    Produce a list of actionable, non-medical recommendations
    based on the user's actual habit values.
    """
    recs = []

    # Night usage
    if user_input.get("night_usage", 0) > 2:
        recs.append(
            "Consider reducing recreational screen use before bedtime "
            "to support better sleep quality."
        )

    # Social media
    if user_input.get("social_media", 0) > 4:
        recs.append(
            "Consider scheduling short distraction-free periods during "
            "study or work hours to balance social media usage."
        )

    # Focus time
    if user_input.get("focus_time", 0) < 2:
        recs.append(
            "Try starting with a short focused session (e.g., 25 minutes) "
            "and gradually increase its duration."
        )

    # Sleep
    if user_input.get("sleep", 0) < 6:
        recs.append(
            "Consider maintaining a consistent sleep schedule and aiming "
            "for 7-9 hours of sleep."
        )

    # Screen time
    if user_input.get("screen_time", 0) > 8:
        recs.append(
            "Take regular screen breaks — a 5-minute break every hour "
            "can help reduce eye strain and mental fatigue."
        )

    # Phone unlocks
    if user_input.get("phone_unlocks", 0) > 100:
        recs.append(
            "Try batching your phone checks at set intervals rather than "
            "responding to every notification immediately."
        )

    # Exercise
    if user_input.get("exercise", 0) < 15:
        recs.append(
            "Even a short daily walk or stretch routine can improve "
            "focus and reduce stress."
        )

    # Stress
    if user_input.get("stress", 0) > 7:
        recs.append(
            "Consider incorporating brief relaxation activities such as "
            "deep-breathing exercises into your daily routine."
        )

    # Gaming / short video
    if user_input.get("gaming", 0) > 3 or user_input.get("short_video", 0) > 3:
        recs.append(
            "Setting a daily time limit for entertainment activities "
            "can help balance recreation with other priorities."
        )

    # Notifications
    if user_input.get("notifications", 0) > 200:
        recs.append(
            "Reducing non-essential app notifications may help lower "
            "distractions throughout the day."
        )

    # Productivity
    if user_input.get("productivity", 0) < 4:
        recs.append(
            "Try breaking your tasks into smaller goals and celebrating "
            "small wins to build momentum."
        )

    # Sleep + night usage combo
    if user_input.get("sleep", 0) < 6.5 and user_input.get("night_usage", 0) > 1.5:
        recs.append(
            "Your sleep duration and night-time usage suggest the two "
            "may be linked — consider a digital curfew 30 minutes "
            "before bed."
        )

    if not recs:
        recs.append(
            "Your digital habits appear balanced — keep up the "
            "healthy routine!"
        )

    return recs


# --------------- pattern detection ---------------

def detect_patterns(user_input: dict) -> list[str]:
    """
    Identify notable behavioural patterns from the user's input.
    """
    patterns = []

    st = user_input.get("screen_time", 0)
    if st > 8:
        patterns.append("Very high daily screen time")
    elif st > 5:
        patterns.append("Above-average daily screen time")

    if user_input.get("social_media", 0) > 4:
        patterns.append("High social media usage")

    if user_input.get("gaming", 0) > 3:
        patterns.append("Significant gaming hours")

    if user_input.get("short_video", 0) > 3:
        patterns.append("Considerable short-video consumption")

    pu = user_input.get("phone_unlocks", 0)
    if pu > 100:
        patterns.append("Frequent phone checking")
    elif pu > 60:
        patterns.append("Moderately frequent phone checking")

    if user_input.get("notifications", 0) > 200:
        patterns.append("Very high notification volume")

    if user_input.get("night_usage", 0) > 2:
        patterns.append("High evening/night screen usage")

    if user_input.get("sleep", 0) < 6:
        patterns.append("Below-recommended sleep duration")

    if user_input.get("focus_time", 0) < 2:
        patterns.append("Low daily focus/study time")

    if user_input.get("exercise", 0) < 15:
        patterns.append("Minimal physical activity")

    if user_input.get("stress", 0) > 7:
        patterns.append("Elevated self-reported stress")

    if user_input.get("productivity", 0) < 4:
        patterns.append("Low self-reported productivity")

    if not patterns:
        patterns.append("No major concerning patterns detected")

    return patterns


# --------------- template explanation ---------------

def generate_explanation(prediction: dict, user_input: dict) -> str:
    """
    Produce a concise natural-language summary of the user's analysis.
    This is the default explanation; Bedrock can optionally enhance it.
    """
    risk = prediction.get("risk_level", "Unknown")
    cluster = prediction.get("cluster_label", "Unknown")
    wb = prediction.get("wellbeing_score")
    patterns = prediction.get("patterns", [])

    lines = []
    lines.append(f"Your digital habits have been classified as "
                 f"**{risk} risk**.")

    if wb is not None:
        lines.append(f"Your estimated wellbeing score is "
                     f"**{wb}/100**.")

    lines.append(f"Based on K-Means clustering, your behavioural profile "
                 f"aligns with the **{cluster}** group.")

    if patterns:
        notable = [p for p in patterns
                   if "No major" not in p]
        if notable:
            lines.append("Notable patterns detected: " +
                         "; ".join(notable[:4]) + ".")

    if prediction.get("anomaly", {}).get("is_anomaly"):
        lines.append("⚠️ Your usage pattern is statistically unusual "
                     "compared with the dataset distribution.")

    lines.append("Please see the recommendations below for general "
                 "wellbeing suggestions.")

    return " ".join(lines)


# --------------- optional Bedrock & explanation coordinator ---------------

def get_explanation(prediction: dict, user_input: dict,
                    recommendations: list[str]) -> str:
    """
    Produce natural-language explanation for the user.
    Tries Bedrock first (if configured), and falls back seamlessly to
    the local rule-based template summary.
    """
    try:
        from aws.bedrock_utils import generate_bedrock_explanation
        bedrock_text = generate_bedrock_explanation(prediction, user_input)
        if bedrock_text:
            return bedrock_text
    except Exception:
        pass

    # Guaranteed offline local fallback
    return generate_explanation(prediction, user_input)
