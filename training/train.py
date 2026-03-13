""" File for training a YOLO model on the PCB dataset """

import torch
from ultralytics import YOLO
from config import DATA_YAML, PROJECT_DIR, EXPERIMENT_NAME

# -----

def train_model() -> None:
    """ Train a YOLO model on the PCB dataset """

    # Load a pre-trained YOLO model (version nano)
    model = YOLO("yolo26n.pt")
    device = 0 if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    # Train the model on the PCB dataset
    model.train(
        data=DATA_YAML,       # path to dataset config file
        epochs=50,            # number of epochs
        imgsz=1024,           # image size
        batch=4,              # batch size
        device=device,        # "cpu" or "0" for GPU
        workers=0,            # number of data loading workers
        cache=False,          # cache images for faster training (set to True if you have enough RAM)
        project=PROJECT_DIR,  # output folder
        name=EXPERIMENT_NAME, # experiment name
        pretrained=True,      # use pre-trained weights
        verbose=True          # print training progress and results
    )

    return None
