"""
Vercel Serverless Function Entry Point for MindfulTech (MLT)
============================================================
Exposes the WSGI Flask `app` callable for Vercel's Python runtime.
"""

import os
import sys

# Ensure the repository root directory is in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Ensure database tables exist in the writable environment
from database.db import init_db
try:
    init_db()
except Exception as e:
    print(f"Warning: database init error: {e}", file=sys.stderr)

# Import Flask app callable
from app import app
