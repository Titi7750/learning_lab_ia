""" Main entry point to launch the Streamlit diagnostic interface """

import sys
import subprocess
from pathlib import Path

# -----

def main() -> None:
    """ Launch Streamlit UI from the project root """

    app_path = Path(__file__).resolve().parent / "streamlit_app" / "app.py"
    command = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    subprocess.run(command, check=True)

# -----

if __name__ == "__main__":
    main()
