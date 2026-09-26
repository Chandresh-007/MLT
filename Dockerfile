# MindfulTech – Production Docker Image for AWS Amplify
# =====================================================
# Uses a multi-stage build for smaller image size.

# --- Stage 1: Builder ---
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies for scikit-learn, numpy, etc.
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt
RUN pip install --no-cache-dir --prefix=/install gunicorn

# --- Stage 2: Production ---
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . .

# Create writable directory for SQLite database
RUN mkdir -p /app/database && chmod 777 /app/database

# Copy and set up entrypoint (remaps MLT_* -> AWS_* env vars for Amplify)
COPY entrypoint.sh /app/entrypoint.sh
RUN chmod +x /app/entrypoint.sh

# Expose port 8080 (Amplify default for containers)
EXPOSE 8080

# Health-check endpoint (Amplify uses this)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/')" || exit 1

# Run with Gunicorn (production WSGI server)
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["gunicorn", "--bind", "0.0.0.0:8080", "--workers", "2", "--threads", "4", "--timeout", "120", "app:app"]
