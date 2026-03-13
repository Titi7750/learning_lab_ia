""" File for validating a YOLO model on the PCB dataset """

from ultralytics import YOLO
from config import DATA_YAML, MODEL_PATH

# -----

def validate_model() -> None:
    """ Validate the trained YOLO model on the PCB dataset """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    # Validate the model on the PCB dataset
    model = YOLO(str(MODEL_PATH))
    metrics = model.val(data=DATA_YAML)

    print("Validation completed.")
    print(metrics)

    return None
