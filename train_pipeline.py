""" Pipeline entry point for training, validation and sample inference """

from training.train import train_model
from inference.predict import predict_image
from training.validate import validate_model

# -----

def main() -> None:
    """ Train, validate, and run sample inference on the PCB dataset """

    print("\n--- Training model ---")
    train_model()

    print("\n--- Validating model ---")
    validate_model()

    print("\n--- Running inference ---")
    predict_image("data/test/images/")

# -----

if __name__ == "__main__":
    main()
