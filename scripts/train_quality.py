"""Train overall quality and optional verified defect heads."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from qsart.training.cli import training_main

if __name__ == "__main__":
    training_main("quality")
