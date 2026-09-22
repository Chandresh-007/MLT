"""
MindfulTech – AWS Utilities Package
===================================
Provides modular, safe utilities for Amazon S3 (model & dataset storage)
and Amazon Bedrock (optional post-prediction explanation layer).
"""

from aws.s3_utils import (
    is_s3_configured,
    get_s3_client,
    upload_dataset_to_s3,
    upload_models_to_s3,
    download_models_from_s3,
    download_dataset_from_s3,
)

from aws.bedrock_utils import (
    is_bedrock_configured,
    generate_bedrock_explanation,
)

__all__ = [
    "is_s3_configured",
    "get_s3_client",
    "upload_dataset_to_s3",
    "upload_models_to_s3",
    "download_models_from_s3",
    "download_dataset_from_s3",
    "is_bedrock_configured",
    "generate_bedrock_explanation",
]
