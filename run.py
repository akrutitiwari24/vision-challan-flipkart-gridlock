#!/usr/bin/env python3
"""
VisionChallan AI - Launcher
Starts FastAPI backend and Streamlit frontend simultaneously.

Usage:
    python run.py
    python run.py --api-only
    python run.py --ui-only
"""

import sys
import os
import subprocess
import threading
import time
import argparse

ROOT = os.path.dirname(os.path.abspath(__file__))


def get_python_executable():
    if sys.platform == "win32":
        venv_py = os.path.join(ROOT, "venv", "Scripts", "python.exe")
    else:
        venv_py = os.path.join(ROOT, "venv", "bin", "python")
    if os.path.exists(venv_py):
        return venv_py
    return sys.executable


def start_api():
    print("[INFO] Starting FastAPI backend on http://localhost:8000 ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    py_exec = get_python_executable()
    subprocess.run(
        [py_exec, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT, env=env,
    )


def start_ui():
    print("[INFO] Starting Streamlit UI on http://localhost:8501 ...")
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    py_exec = get_python_executable()
    subprocess.run(
        [py_exec, "-m", "streamlit", "run", "streamlit_app.py",
         "--server.port", "8501",
         "--server.address", "0.0.0.0",
         "--browser.gatherUsageStats", "false"],
        cwd=ROOT, env=env,
    )


def main():
    parser = argparse.ArgumentParser(description="VisionChallan AI Launcher")
    parser.add_argument("--api-only", action="store_true")
    parser.add_argument("--ui-only",  action="store_true")
    args = parser.parse_args()

    # Load .env if present
    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)
        print("[SUCCESS] Loaded .env configuration")
    else:
        print("[WARNING] No .env file found. Copy .env.example to .env and add your GROQ_API_KEY.")

    if args.api_only:
        start_api()
    elif args.ui_only:
        start_ui()
    else:
        # Run both concurrently
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        time.sleep(2)   # give API a moment to bind
        start_ui()      # blocks (Streamlit runs in main thread)


if __name__ == "__main__":
    main()
