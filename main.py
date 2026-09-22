"""
MindfulTech – Main Application Entry Point
==========================================
Allows running the application via `python main.py` as an alias for `python app.py`.
"""

from app import app

if __name__ == "__main__":
    print("Starting MindfulTech Web Application...")
    print("Open http://127.0.0.1:5000 in your browser.")
    app.run(debug=True, port=5000)
