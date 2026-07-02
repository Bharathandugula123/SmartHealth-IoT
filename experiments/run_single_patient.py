from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from main import run_pipeline

if __name__ == "__main__":
    run_pipeline()
