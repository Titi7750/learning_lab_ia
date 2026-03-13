""" Main entry point for training and validating the YOLO model - Python 3.11.9 """

from training.train import train_model
from inference.predict import predict_image
from training.validate import validate_model

# -----

def main() -> None:
    """ Main function to train and validate the YOLO model on the PCB dataset """

    print("\n--- Training model ---")
    train_model()

    print("\n--- Validating model ---")
    validate_model()

    print("\n--- Running inference ---")
    predict_image("data/test/images/")

    return None

# -----

if __name__ == "__main__":
    main()
