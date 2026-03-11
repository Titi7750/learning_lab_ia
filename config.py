""" Configuration file for the PCB detection project using YOLO """

from pathlib import Path

# -----

PROJECT_DIR = "runs_pcb"
EXPERIMENT_NAME = "yolo_pcb_detection"
DATA_YAML = "data/pcb_dataset/data.yaml"
MODEL_PATH = Path(PROJECT_DIR) / EXPERIMENT_NAME / "weights" / "best.pt"
