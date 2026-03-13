""" Inference script to run predictions on new PCB images using the trained YOLO model """

from pathlib import Path
from ultralytics import YOLO
from config import MODEL_PATH

# -----

def predict_image(param_image_path: str) -> None:
    """ Run inference on a new PCB image using the trained YOLO model """

    # Load the trained model
    image_path_obj = Path(param_image_path)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    if not image_path_obj.exists():
        raise FileNotFoundError(f"Input image not found: {image_path_obj}")

    # Run inference on the input image
    model = YOLO(str(MODEL_PATH))
    results = model.predict(
        source=str(image_path_obj),
        conf=0.4,
        save=True
    )

    # Display results
    for result in results:
        result.show()

    print("Prediction completed.")

    return None
