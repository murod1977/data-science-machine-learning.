from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parent
for folder in sorted(root.glob("0*")):
    subprocess.run([sys.executable, "project.py"], cwd=folder, check=True)
