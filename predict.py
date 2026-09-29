from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torch.nn as nn

from PIL import Image


BASE_DIR = Path(__file__).resolve().parent

OUTPUT_DIR = BASE_DIR / "outputs"

MODEL_PATH = (
    OUTPUT_DIR / "mnist_logistic_model.pth"
)

CONFIG_PATH = (
    OUTPUT_DIR / "model_config.json"
)


class MNISTLogisticRegression(nn.Module):

    def __init__(
        self,
        input_size,
        num_classes
    ):
        super().__init__()

        self.linear = nn.Linear(
            input_size,
            num_classes
        )

    def forward(self, x):
        return self.linear(x)


def load_model():

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}\n"
            "Run train.py first."
        )

    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(
            f"Model configuration not found: {CONFIG_PATH}\n"
            "Run train.py first."
        )

    with open(
        CONFIG_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        config = json.load(file)

    model = MNISTLogisticRegression(
        config["input_size"],
        config["num_classes"]
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location="cpu"
        )
    )

    model.eval()

    return model, config


def preprocess_image(image_path):

    image = Image.open(
        image_path
    ).convert("L")

    image = image.resize(
        (28, 28)
    )

    image_array = np.array(
        image,
        dtype=np.float32
    )

    image_array = image_array / 255.0

    image_array = image_array.reshape(
        1,
        784
    )

    return torch.tensor(
        image_array,
        dtype=torch.float32
    )


def predict(image_path):

    model, config = load_model()

    image_tensor = preprocess_image(
        image_path
    )

    with torch.no_grad():

        logits = model(
            image_tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predicted_digit = probabilities.argmax(
            dim=1
        ).item()

    probabilities = probabilities[0].numpy()

    print("\nPrediction")
    print("----------")

    print(
        f"Image: {image_path}"
    )

    print(
        f"Predicted digit: {predicted_digit}"
    )

    print(
        f"Confidence: "
        f"{probabilities[predicted_digit]:.2%}"
    )

    print("\nAll class probabilities:")

    for digit, probability in enumerate(
        probabilities
    ):

        print(
            f"Digit {digit}: "
            f"{probability:.2%}"
        )

    return predicted_digit


def main():

    parser = argparse.ArgumentParser(
        description=(
            "Predict an MNIST digit using "
            "the saved logistic regression model."
        )
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to a digit image."
    )

    args = parser.parse_args()

    image_path = Path(
        args.image
    )

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    predict(image_path)


if __name__ == "__main__":
    main()