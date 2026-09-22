#!/bin/bash
# ==============================================================================
# MindfulTech – Amazon EC2 Deployment Script
# ==============================================================================
# Target: Amazon Linux 2023 or Ubuntu 22.04 LTS (t2.micro / t3.micro Free Tier)
#
# Prerequisites on AWS Console:
# 1. Launch a Free Tier EC2 instance (Amazon Linux 2023 or Ubuntu 22.04).
# 2. Configure Security Group:
#    - Inbound Rule: Allow TCP port 22 (SSH) from your IP.
#    - Inbound Rule: Allow TCP port 5000 (Custom TCP) from 0.0.0.0/0 (or your IP).
#    - Inbound Rule (Optional): Allow TCP port 80 (HTTP) from 0.0.0.0/0.
# 3. Attach an IAM Role to EC2 with "AmazonS3ReadOnlyAccess" (or full S3 access)
#    so the instance can download models without needing hardcoded credentials!
# ==============================================================================

set -e

echo "============================================================"
echo "MindfulTech – EC2 Setup & Deployment"
echo "============================================================"

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
else
    OS="unknown"
fi

echo "--> Detected OS: $OS"

# 1. Update and install packages
echo "--> Updating system packages..."
if [ "$OS" = "amzn" ] || [ "$OS" = "al2023" ]; then
    sudo dnf update -y
    sudo dnf install -y python3 python3-pip git
elif [ "$OS" = "ubuntu" ] || [ "$OS" = "debian" ]; then
    sudo apt-get update -y
    sudo apt-get install -y python3 python3-pip python3-venv git
else
    echo "Unsupported OS. Please install Python 3.9+, pip, and git manually."
fi

# 2. Verify repository directory
APP_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$APP_DIR"
echo "--> Working in: $APP_DIR"

# 3. Create and activate virtual environment
if [ ! -d "venv" ]; then
    echo "--> Creating Python virtual environment..."
    python3 -m venv venv
fi

echo "--> Activating virtual environment..."
source venv/bin/activate

# 4. Install dependencies
echo "--> Installing Python dependencies from requirements.txt..."
pip install --upgrade pip
pip install -r requirements.txt

# 5. Model artifact check
echo "--> Checking ML model files..."
if [ ! -f "models/random_forest_model.pkl" ]; then
    echo "--> Models not found locally. Attempting S3 download or local training..."
    if [ -n "$S3_BUCKET_NAME" ]; then
        python scripts/sync_s3.py --download || true
    fi

    # Fallback to local training if models are still missing
    if [ ! -f "models/random_forest_model.pkl" ]; then
        echo "--> Generating dataset and training models locally on EC2..."
        python data/generate_dataset.py
        python ml/train_models.py
        python ml/evaluate_models.py
    fi
fi

echo "============================================================"
echo "Setup complete!"
echo "============================================================"
echo "You can now run MindfulTech directly with:"
echo "    source venv/bin/activate"
echo "    python main.py"
echo ""
echo "Or create a background systemd service:"
echo "    sudo tee /etc/systemd/system/mindfultech.service > /dev/null <<EOF"
echo "[Unit]"
echo "Description=MindfulTech ML Web App"
echo "After=network.target"
echo ""
echo "[Service]"
echo "User=$(whoami)"
echo "WorkingDirectory=$APP_DIR"
echo "ExecStart=$APP_DIR/venv/bin/python $APP_DIR/main.py"
echo "Restart=always"
echo ""
echo "[Install]"
echo "WantedBy=multi-user.target"
echo "EOF"
echo ""
echo "    sudo systemctl daemon-reload"
echo "    sudo systemctl start mindfultech"
echo "    sudo systemctl enable mindfultech"
echo "============================================================"
