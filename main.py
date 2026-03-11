""" Main entry point for training and validating the YOLO model """

from training.train import train_model
from inference.predict import predict_image
from training.validate import validate_model

# -----

def main() -> None:
    """ Main function to train and validate the YOLO model on the PCB dataset """

    # Train the model
    print("Training model...")
    train_model()

    # Validate the model
    print("Validating model...")
    validate_model()

    # Example prediction on a new image (uncomment and provide path to use)
    # print("Running inference...")
    # predict_image("path/to/new/image.jpg")

    return None

# -----

if __name__ == "__main__":
    main()
