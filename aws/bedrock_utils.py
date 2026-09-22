"""
MindfulTech - Amazon Bedrock Explanation Layer (Optional)
=========================================================
Provides natural-language explanation of already-computed Scikit-learn
predictions using Amazon Bedrock foundation models.

CRITICAL ACADEMIC & SAFETY RULES
--------------------------------
1. Bedrock does NOT classify risk or perform machine learning predictions.
   The classification is performed exclusively by the local Scikit-learn model.
2. Bedrock is only an explanation layer invoked AFTER the ML model has produced
   the risk level (Low / Moderate / High), wellbeing score, and cluster persona.
3. Bedrock MUST NOT:
   - Diagnose addiction, ADHD, depression, anxiety, or any mental health condition.
   - Claim to measure neurotransmitters or dopamine.
   - Override or alter the ML model's risk category.
   - Generate its own clinical conclusions.
4. If Bedrock is unavailable, unconfigured, or errors out, the application
   MUST continue functioning flawlessly with the local rule-based template.
"""

import os
import json
import logging

logger = logging.getLogger("mindfultech.bedrock")

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def is_bedrock_configured() -> bool:
    """Check if Bedrock model ID and AWS region/credentials are set."""
    if not BOTO3_AVAILABLE:
        return False
    model_id = os.environ.get("BEDROCK_MODEL_ID")
    return bool(model_id)


def get_bedrock_client():
    """Initialize a boto3 bedrock-runtime client."""
    if not BOTO3_AVAILABLE:
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    try:
        if access_key and secret_key:
            return boto3.client(
                "bedrock-runtime",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        else:
            return boto3.client("bedrock-runtime", region_name=region)
    except Exception as e:
        logger.warning(f"Could not initialize Bedrock client: {e}")
        return None


def generate_bedrock_explanation(prediction: dict, user_input: dict) -> str | None:
    """
    Generate a simple, supportive explanation of the ML prediction using Bedrock.

    Parameters
    ----------
    prediction : dict
        Output from Scikit-learn prediction pipeline:
        risk_level, wellbeing_score, cluster_label, patterns, recommendations
    user_input : dict
        13 primary habit metrics submitted by the user.

    Returns
    -------
    str or None
        Generated explanation text, or None if Bedrock is unavailable/fails.
    """
    if not is_bedrock_configured():
        return None

    client = get_bedrock_client()
    if not client:
        return None

    model_id = os.environ.get(
        "BEDROCK_MODEL_ID", "anthropic.claude-3-haiku-20240307-v1:0"
    )

    # Extract already-predicted outputs (NEVER ask Bedrock to classify)
    risk_level = prediction.get("risk_level", "Moderate")
    score = prediction.get("wellbeing_score", "N/A")
    cluster = prediction.get("cluster_label", "General Profile")
    patterns = prediction.get("patterns", [])
    recommendations = prediction.get("recommendations", [])

    system_prompt = (
        "You are an educational AI assistant for the MindfulTech digital habit platform. "
        "Your task is to provide a brief, empathetic, 2-3 sentence summary explaining why "
        "the user's habit profile was classified by our Machine Learning model.\n\n"
        "STRICT SAFETY & ETHICAL RULES:\n"
        "- Do NOT diagnose addiction, ADHD, depression, anxiety, or any medical or mental health condition.\n"
        "- Do NOT mention dopamine, neurotransmitters, or biochemical mechanisms.\n"
        "- Do NOT change or contradict the ML model's predicted risk level.\n"
        "- Emphasize that this is an educational behavioural observation, not medical advice.\n"
        "- Tone must be constructive, friendly, and non-judgmental."
    )

    user_prompt = (
        f"The user's habits were evaluated by our Scikit-learn Random Forest model:\n"
        f"- Predicted Risk Level: {risk_level}\n"
        f"- Wellbeing Score: {score}/100\n"
        f"- Behavioural Cluster: {cluster}\n"
        f"- Key Usage Stats: Screen Time={user_input.get('screen_time')}h, "
        f"Night Usage={user_input.get('night_usage')}h, "
        f"Sleep={user_input.get('sleep')}h, "
        f"Focus Time={user_input.get('focus_time')}h\n"
        f"- Detected Patterns: {', '.join(patterns) if patterns else 'Balanced patterns'}\n"
        f"- Top Recommendations: {', '.join(recommendations[:2]) if recommendations else 'Maintain habits'}\n\n"
        f"Please write a concise 2-3 sentence summary explaining this assessment to the user."
    )

    try:
        if "claude" in model_id.lower():
            # Anthropic Claude request format
            body = json.dumps({
                "anthropic_version": "bedrock-2023-05-31",
                "max_tokens": 250,
                "system": system_prompt,
                "messages": [
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": 0.4,
            })
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            content_blocks = response_body.get("content", [])
            if content_blocks and "text" in content_blocks[0]:
                return content_blocks[0]["text"].strip()

        elif "titan" in model_id.lower():
            # Amazon Titan request format
            body = json.dumps({
                "inputText": f"{system_prompt}\n\nUser Information:\n{user_prompt}\n\nExplanation:",
                "textGenerationConfig": {
                    "maxTokenCount": 250,
                    "temperature": 0.4,
                    "topP": 0.9,
                }
            })
            response = client.invoke_model(
                modelId=model_id,
                body=body,
                contentType="application/json",
                accept="application/json",
            )
            response_body = json.loads(response["body"].read().decode("utf-8"))
            results = response_body.get("results", [])
            if results and "outputText" in results[0]:
                return results[0]["outputText"].strip()

        else:
            logger.info(f"Unsupported Bedrock model ID: {model_id}")
            return None

    except Exception as e:
        logger.warning(f"Bedrock invocation failed (falling back to template): {e}")
        return None

    return None
