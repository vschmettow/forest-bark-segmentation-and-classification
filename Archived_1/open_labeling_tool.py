#!/usr/bin/env python3
"""
Launch the labeling tool (AnyLabeling) on this machine.
On Windows, the project includes AutoLabeling-GPU.exe; on macOS/Linux
we run AnyLabeling via pip.
"""
import sys
import subprocess
import shutil
from pathlib import Path

def main():
    # On Windows, prefer the bundled exe if present
    if sys.platform == "win32":
        exe = Path(__file__).resolve().parent / "Labeling Tool" / "AutoLabeling-GPU" / "AutoLabeling-GPU.exe"
        if exe.exists():
            subprocess.Popen([str(exe)], cwd=exe.parent)
            print("Launched AutoLabeling-GPU (Windows).")
            return

    # macOS / Linux: run via entry point (anylabeling has no __main__)
    try:
        subprocess.run([
            sys.executable, "-c",
            "from anylabeling.app import main; main()"
        ])
        print("Launched AnyLabeling.")
        return
    except Exception as e:
        print("Could not start AnyLabeling:", e, file=sys.stderr)
        print("Install:  pip3 install anylabeling PyQt5", file=sys.stderr)
        print("Then run:  python3 open_labeling_tool.py", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
