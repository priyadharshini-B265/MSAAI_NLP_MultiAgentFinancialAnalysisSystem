#!/usr/bin/env python3
"""
Launcher script for the Gemini ADK Multi-Agent Financial Orchestrator Web Interface.

Usage:
    python run_web.py
Then open http://127.0.0.1:8080 in your browser.
"""

import os
import sys
import subprocess
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
SERVER_SCRIPT = WEB_DIR / "server.py"

def main():
    print("=" * 70)
    print("  Gemini ADK Multi-Agent Financial Orchestrator Web Application")
    print("  Course: Natural Language Processing (MSAAI) - Final Team Project")
    print("  Team: MS_AAI_520_GRP3 (Moinuddin K, Ravisha Krishnamurthy, Priya)")
    print("=" * 70)
    print(f"Launching web server from: {SERVER_SCRIPT}")
    print("Target Local URL: http://127.0.0.1:8080")
    print("Press CTRL+C at any time to stop the server.\n")

    sys.path.insert(0, str(BASE_DIR))
    sys.path.insert(0, str(BASE_DIR / "src"))
    sys.path.insert(0, str(WEB_DIR))
    os.chdir(str(WEB_DIR))

    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")

    try:
        import uvicorn
        uvicorn.run("server:app", host=host, port=port, app_dir=str(WEB_DIR), reload=False)
    except KeyboardInterrupt:
        print("\nWeb server shut down successfully.")
    except Exception as e:
        print(f"\nError launching server: {e}")
        print("Fallback command: cd web && python server.py")

if __name__ == "__main__":
    main()
