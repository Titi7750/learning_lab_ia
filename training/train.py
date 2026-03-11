""" File for training a YOLO model on the PCB dataset """

from ultralytics import YOLO
from config import DATA_YAML, PROJECT_DIR, EXPERIMENT_NAME

# -----

def train_model() -> None:
    """ Train a YOLO model on the PCB dataset """

    # Load a pre-trained YOLO model (version nano)
    model = YOLO("yolov8n.pt")

    # Train the model on the PCB dataset
    model.train(
        data=DATA_YAML,       # path to dataset config file
        epochs=30,            # number of epochs
        imgsz=640,            # image size
        batch=16,             # batch size
        device="cpu",         # "cpu" or "0" for GPU
        project=PROJECT_DIR,  # output folder
        name=EXPERIMENT_NAME, # experiment name
        pretrained=True,      # use pre-trained weights
        verbose=True          # print training progress and results
    )

    return None
