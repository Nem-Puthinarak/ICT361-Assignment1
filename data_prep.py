from pathlib import Path

import kagglehub
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = Path(
    kagglehub.dataset_download("oddrationale/mnist-in-csv")
)

CLASS_NAMES = [str(i) for i in range(10)]

def find_csv(filename):
    file_path = DATA_DIR / filename

    if file_path.is_file():
        return file_path

    matches = list(DATA_DIR.rglob(filename))

    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find {filename} inside {DATA_DIR}"
    )

def load_csv(filename):
    file_path = find_csv(filename)

    dataframe = pd.read_csv(file_path)

    if "label" not in dataframe.columns:
        raise ValueError(
            f"{filename} must contain a 'label' column."
        )

    if dataframe.shape[1] != 785:
        raise ValueError(
            f"{filename} must contain 1 label column and "
            f"784 pixel columns. Found {dataframe.shape[1]} columns."
        )

    if dataframe.empty:
        raise ValueError(f"{filename} contains no data.")

    if not dataframe["label"].isin(range(10)).all():
        raise ValueError(
            f"{filename} contains invalid class labels."
        )

    pixel_columns = dataframe.drop(columns="label")

    X = pixel_columns.to_numpy(dtype=np.float32)

    y = dataframe["label"].astype("int64")

    if not np.isfinite(X).all():
        raise ValueError(
            f"{filename} contains missing or infinite pixel values."
        )

    if X.min() < 0 or X.max() > 255:
        raise ValueError(
            f"{filename} must contain pixel values between 0 and 255."
        )

    X = X / 255.0

    return X, y, pixel_columns.columns

def prepare_data():
    X_train, y_train, train_columns = load_csv(
        "mnist_train.csv"
    )

    X_test, y_test, test_columns = load_csv(
        "mnist_test.csv"
    )

    if not train_columns.equals(test_columns):
        raise ValueError(
            "Training and test pixel columns must have "
            "the same names and order."
        )

    return X_train, y_train, X_test, y_test

def save_dataset_figures(
    X_train,
    y_train,
    X_test,
    y_test
):
    output_dir = BASE_DIR / "outputs"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.axis("off")

    text = (
        "MNIST Dataset Information\n\n"
        f"Training images : {len(X_train):,}\n"
        f"Training shape  : {X_train.shape}\n"
        f"Test images     : {len(X_test):,}\n"
        f"Test shape      : {X_test.shape}\n"
        f"Image size      : 28 × 28 pixels\n"
        f"Features/image  : 784\n"
        f"Classes         : 10 (digits 0–9)\n"
        f"Pixel range     : {X_train.min():.1f} – {X_train.max():.1f}"
    )

    ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        fontsize=14
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "dataset_shapes.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    fig, axes = plt.subplots(
        2,
        5,
        figsize=(10, 5)
    )

    for digit in range(10):
        indices = np.where(
            y_train.to_numpy() == digit
        )[0]

        index = indices[0]

        ax = axes.flat[digit]

        ax.imshow(
            X_train[index].reshape(28, 28),
            cmap="gray"
        )

        ax.set_title(f"Digit: {digit}")
        ax.axis("off")

    fig.suptitle(
        "One Training Sample for Each MNIST Digit",
        fontsize=15
    )

    fig.tight_layout()

    fig.savefig(
        output_dir / "sample_digits.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

if __name__ == "__main__":

    print("Dataset folder:")
    print(DATA_DIR)

    X_train, y_train, X_test, y_test = prepare_data()

    print("\nDataset Shapes")
    print("X_train:", X_train.shape)
    print("y_train:", y_train.shape)
    print("X_test :", X_test.shape)
    print("y_test :", y_test.shape)

    print("\nPixel Range After Normalization")
    print("Minimum:", X_train.min())
    print("Maximum:", X_train.max())

    print("\nTraining Class Distribution")

    counts = y_train.value_counts().sort_index()

    for digit, count in counts.items():
        print(
            f"Digit {digit}: {count:,} images"
        )

    save_dataset_figures(
        X_train,
        y_train,
        X_test,
        y_test
    )

    print("\nDataset preparation complete.")
    print("Figures saved inside outputs/")