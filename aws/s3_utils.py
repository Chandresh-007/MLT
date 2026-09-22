"""
MindfulTech - Amazon S3 Storage Utilities
=========================================
Handles uploading and downloading datasets, trained Scikit-learn models,
and evaluation artifacts to/from Amazon S3.

Architecture
------------
- Used on EC2 or local dev environment to backup and pull artifacts.
- Credentials loaded safely from environment variables (or EC2 IAM role).
- Never hardcodes AWS credentials.
- Operates safely without crashing if AWS credentials/bucket are not set.
"""

import os
import glob
import logging

logger = logging.getLogger("mindfultech.s3")

# Try importing boto3
try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False


def get_bucket_name() -> str | None:
    """Retrieve the S3 bucket name from environment variables."""
    return os.environ.get("S3_BUCKET_NAME")


def get_s3_client():
    """
    Initialize an S3 client using environment variables or instance IAM profile.
    Returns None if boto3 is unavailable or client initialization fails.
    """
    if not BOTO3_AVAILABLE:
        logger.warning("boto3 is not installed. S3 features are unavailable.")
        return None

    region = os.environ.get("AWS_REGION", "us-east-1")
    access_key = os.environ.get("AWS_ACCESS_KEY_ID")
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")

    try:
        if access_key and secret_key:
            return boto3.client(
                "s3",
                region_name=region,
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
            )
        else:
            # Fallback to default credentials (e.g., EC2 IAM Role, ~/.aws/credentials)
            return boto3.client("s3", region_name=region)
    except Exception as e:
        logger.warning(f"Could not initialize S3 client: {e}")
        return None


def is_s3_configured() -> bool:
    """Check whether S3 credentials and bucket name are configured."""
    bucket = get_bucket_name()
    if not bucket or not BOTO3_AVAILABLE:
        return False
    client = get_s3_client()
    return client is not None


def upload_dataset_to_s3(local_dir: str = "data", s3_prefix: str = "data/") -> dict:
    """
    Upload local dataset files (dataset.csv, processed_data.csv) to S3.

    Returns
    -------
    dict with keys: success (bool), uploaded (list), error (str or None)
    """
    client = get_s3_client()
    bucket = get_bucket_name()

    if not client or not bucket:
        return {
            "success": False,
            "uploaded": [],
            "error": "S3 is not configured. Set AWS credentials and S3_BUCKET_NAME.",
        }

    dataset_files = ["dataset.csv", "processed_data.csv"]
    uploaded = []

    for fname in dataset_files:
        fpath = os.path.join(local_dir, fname)
        if os.path.exists(fpath):
            s3_key = f"{s3_prefix.rstrip('/')}/{fname}"
            try:
                client.upload_file(fpath, bucket, s3_key)
                uploaded.append(s3_key)
                logger.info(f"Uploaded {fpath} to s3://{bucket}/{s3_key}")
            except Exception as e:
                return {
                    "success": False,
                    "uploaded": uploaded,
                    "error": f"Failed uploading {fname}: {e}",
                }

    return {
        "success": True,
        "uploaded": uploaded,
        "error": None,
    }


def upload_models_to_s3(local_dir: str = "models", s3_prefix: str = "models/") -> dict:
    """
    Upload all trained Scikit-learn model files (.pkl) and metadata (.json) to S3.

    Returns
    -------
    dict with keys: success (bool), uploaded (list), error (str or None)
    """
    client = get_s3_client()
    bucket = get_bucket_name()

    if not client or not bucket:
        return {
            "success": False,
            "uploaded": [],
            "error": "S3 is not configured. Set AWS credentials and S3_BUCKET_NAME.",
        }

    files = glob.glob(os.path.join(local_dir, "*.pkl")) + glob.glob(
        os.path.join(local_dir, "*.json")
    )

    if not files:
        return {
            "success": False,
            "uploaded": [],
            "error": f"No model files found in {local_dir} to upload.",
        }

    uploaded = []
    for fpath in files:
        fname = os.path.basename(fpath)
        s3_key = f"{s3_prefix.rstrip('/')}/{fname}"
        try:
            client.upload_file(fpath, bucket, s3_key)
            uploaded.append(s3_key)
            logger.info(f"Uploaded {fname} to s3://{bucket}/{s3_key}")
        except Exception as e:
            return {
                "success": False,
                "uploaded": uploaded,
                "error": f"Failed uploading {fname}: {e}",
            }

    return {
        "success": True,
        "uploaded": uploaded,
        "error": None,
    }


def download_models_from_s3(s3_prefix: str = "models/", local_dir: str = "models") -> dict:
    """
    Download trained models (.pkl) and metadata (.json) from S3.
    Useful when deploying to a fresh EC2 instance.

    Returns
    -------
    dict with keys: success (bool), downloaded (list), error (str or None)
    """
    client = get_s3_client()
    bucket = get_bucket_name()

    if not client or not bucket:
        return {
            "success": False,
            "downloaded": [],
            "error": "S3 is not configured. Set AWS credentials and S3_BUCKET_NAME.",
        }

    os.makedirs(local_dir, exist_ok=True)
    downloaded = []

    try:
        paginator = client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=s3_prefix)

        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                fname = os.path.basename(key)
                if fname and (fname.endswith(".pkl") or fname.endswith(".json")):
                    dest_path = os.path.join(local_dir, fname)
                    client.download_file(bucket, key, dest_path)
                    downloaded.append(dest_path)
                    logger.info(f"Downloaded s3://{bucket}/{key} -> {dest_path}")

        if not downloaded:
            return {
                "success": False,
                "downloaded": [],
                "error": f"No models found in s3://{bucket}/{s3_prefix}",
            }

        return {
            "success": True,
            "downloaded": downloaded,
            "error": None,
        }
    except Exception as e:
        return {
            "success": False,
            "downloaded": downloaded,
            "error": f"S3 download failed: {e}",
        }


def download_dataset_from_s3(s3_prefix: str = "data/", local_dir: str = "data") -> dict:
    """
    Download dataset files from S3 into local data directory.

    Returns
    -------
    dict with keys: success (bool), downloaded (list), error (str or None)
    """
    client = get_s3_client()
    bucket = get_bucket_name()

    if not client or not bucket:
        return {
            "success": False,
            "downloaded": [],
            "error": "S3 is not configured. Set AWS credentials and S3_BUCKET_NAME.",
        }

    os.makedirs(local_dir, exist_ok=True)
    downloaded = []

    try:
        for fname in ["dataset.csv", "processed_data.csv"]:
            key = f"{s3_prefix.rstrip('/')}/{fname}"
            dest_path = os.path.join(local_dir, fname)
            try:
                client.download_file(bucket, key, dest_path)
                downloaded.append(dest_path)
                logger.info(f"Downloaded s3://{bucket}/{key} -> {dest_path}")
            except ClientError as ce:
                # File may not exist in S3, ignore if optional
                if ce.response.get("Error", {}).get("Code") != "404":
                    logger.warning(f"Could not download {key}: {ce}")

        return {
            "success": len(downloaded) > 0,
            "downloaded": downloaded,
            "error": None if downloaded else "No dataset files found in S3.",
        }
    except Exception as e:
        return {
            "success": False,
            "downloaded": downloaded,
            "error": str(e),
        }
