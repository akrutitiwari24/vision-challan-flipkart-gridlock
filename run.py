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


def wait_for_api(url="http://localhost:8000/health", timeout=30):
    print("  Waiting for API to be ready...", end="", flush=True)
    start = time.time()
    while time.time() - start < timeout:
        try:
            import requests
            r = requests.get(url, timeout=1)
            if r.status_code == 200:
                print(" ✓")
                return True
        except Exception:
            pass
        print(".", end="", flush=True)
        time.sleep(0.5)
    print(" ✗ (timeout)")
    return False


def start_api():
    env = os.environ.copy()
    env["PYTHONPATH"] = ROOT
    py_exec = get_python_executable()
    subprocess.run(
        [py_exec, "-m", "uvicorn", "app.main:app",
         "--host", "0.0.0.0", "--port", "8000", "--reload"],
        cwd=ROOT, env=env,
    )


def start_ui():
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

    env_file = os.path.join(ROOT, ".env")
    if os.path.exists(env_file):
        from dotenv import load_dotenv
        load_dotenv(env_file)

    print("━" * 50)
    print("  VisionChallan AI — Starting up")
    print("━" * 50)
    print("  API:  http://localhost:8000")
    print("  UI:   http://localhost:8501")
    print("  Docs: http://localhost:8000/docs")
    print("━" * 50)

    if args.api_only:
        start_api()
    elif args.ui_only:
        start_ui()
    else:
        api_thread = threading.Thread(target=start_api, daemon=True)
        api_thread.start()
        wait_for_api()
        start_ui()


if __name__ == "__main__":
    main()
