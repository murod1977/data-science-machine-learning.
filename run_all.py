from pathlib import Path
import subprocess, sys
for script in sorted(Path(__file__).resolve().parent.glob("0*/project.py")):
    subprocess.run([sys.executable, str(script)], check=True)
