"""
MindfulTech - Amazon S3 Synchronization CLI
============================================
Utility script for syncing datasets and trained machine learning models
between local storage and an Amazon S3 bucket.

Usage:
------
# 1. Check S3 connection and bucket status:
python scripts/sync_s3.py --status

# 2. Upload local dataset and trained models to S3:
python scripts/sync_s3.py --upload

# 3. Download trained models and dataset from S3 (e.g. on a new EC2 instance):
python scripts/sync_s3.py --download

Environment variables (read from .env or shell):
- AWS_ACCESS_KEY_ID
- AWS_SECRET_ACCESS_KEY
- AWS_REGION
- S3_BUCKET_NAME
"""

import os
import sys
import argparse

# Add project root to sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(PROJECT_ROOT, ".env"))
except ImportError:
    pass

from aws.s3_utils import (
    is_s3_configured,
    get_s3_client,
    get_bucket_name,
    upload_dataset_to_s3,
    upload_models_to_s3,
    download_models_from_s3,
    download_dataset_from_s3,
)


def cmd_status(bucket: str | None):
    print("=" * 60)
    print("MindfulTech - S3 Configuration Status")
    print("=" * 60)
    client = get_s3_client()
    target_bucket = bucket or get_bucket_name()

    print(f"Boto3 installed : Yes")
    print(f"AWS Region      : {os.environ.get('AWS_REGION', 'default')}")
    print(f"Configured Bucket: {target_bucket or 'NOT SET (check .env)'}")

    if not client:
        print("\n[!] Could not connect to AWS S3. Please verify your credentials.")
        return

    if not target_bucket:
        print("\n[!] Please specify S3_BUCKET_NAME in your .env file or with --bucket.")
        return

    try:
        # Check bucket existence
        client.head_bucket(Bucket=target_bucket)
        print(f"\n[OK] Successfully connected to bucket: {target_bucket}")

        # List contents
        resp = client.list_objects_v2(Bucket=target_bucket, MaxKeys=20)
        contents = resp.get("Contents", [])
        print(f"\nFound {len(contents)} object(s) in bucket:")
        for obj in contents:
            print(f"  - {obj['Key']} ({obj['Size']:,} bytes)")
    except Exception as e:
        print(f"\n[!] Error accessing bucket '{target_bucket}': {e}")


def cmd_upload(data_dir: str, models_dir: str):
    print("=" * 60)
    print("MindfulTech - Uploading Artifacts to Amazon S3")
    print("=" * 60)

    if not is_s3_configured():
        print("[!] S3 is not configured. Please set AWS credentials and S3_BUCKET_NAME in .env")
        sys.exit(1)

    print("\n[1/2] Uploading dataset files ...")
    res_data = upload_dataset_to_s3(local_dir=data_dir)
    if res_data["success"]:
        print(f"  Uploaded {len(res_data['uploaded'])} dataset file(s):")
        for f in res_data["uploaded"]:
            print(f"    + {f}")
    else:
        print(f"  [!] Dataset upload failed: {res_data['error']}")

    print("\n[2/2] Uploading trained Scikit-learn models & metadata ...")
    res_models = upload_models_to_s3(local_dir=models_dir)
    if res_models["success"]:
        print(f"  Uploaded {len(res_models['uploaded'])} model file(s):")
        for f in res_models["uploaded"]:
            print(f"    + {f}")
    else:
        print(f"  [!] Model upload failed: {res_models['error']}")

    print("\nUpload operation complete.")


def cmd_download(data_dir: str, models_dir: str):
    print("=" * 60)
    print("MindfulTech - Downloading Artifacts from Amazon S3")
    print("=" * 60)

    if not is_s3_configured():
        print("[!] S3 is not configured. Please set AWS credentials and S3_BUCKET_NAME in .env")
        sys.exit(1)

    print("\n[1/2] Downloading trained models & metadata ...")
    res_models = download_models_from_s3(local_dir=models_dir)
    if res_models["success"]:
        print(f"  Downloaded {len(res_models['downloaded'])} model file(s):")
        for f in res_models["downloaded"]:
            print(f"    + {f}")
    else:
        print(f"  [!] Model download failed: {res_models['error']}")

    print("\n[2/2] Downloading dataset ...")
    res_data = download_dataset_from_s3(local_dir=data_dir)
    if res_data["success"]:
        print(f"  Downloaded {len(res_data['downloaded'])} dataset file(s):")
        for f in res_data["downloaded"]:
            print(f"    + {f}")
    else:
        print(f"  [!] Dataset download notice: {res_data['error']}")

    print("\nDownload operation complete.")


def main():
    parser = argparse.ArgumentParser(description="MindfulTech S3 Sync Utility")
    parser.add_argument("--upload", action="store_true", help="Upload local dataset and models to S3")
    parser.add_argument("--download", action="store_true", help="Download models and dataset from S3")
    parser.add_argument("--status", action="store_true", help="Check S3 credentials and bucket status")
    parser.add_argument("--bucket", type=str, default=None, help="Override S3 bucket name")
    parser.add_argument("--data-dir", type=str, default=os.path.join(PROJECT_ROOT, "data"), help="Path to data directory")
    parser.add_argument("--models-dir", type=str, default=os.path.join(PROJECT_ROOT, "models"), help="Path to models directory")

    args = parser.parse_args()

    if args.upload:
        cmd_upload(args.data_dir, args.models_dir)
    elif args.download:
        cmd_download(args.data_dir, args.models_dir)
    elif args.status or len(sys.argv) == 1:
        cmd_status(args.bucket)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
