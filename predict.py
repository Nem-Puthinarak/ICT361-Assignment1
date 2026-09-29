from pathlib import Path
import argparse
import json

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageOps

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_PATH = OUTPUT_DIR / "mnist_logistic_model.pth"
CONFIG_PATH = OUTPUT_DIR / "model_config.json"

class MNISTLogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}\nRun train.py first.")
    if not CONFIG_PATH.is_file():
        raise FileNotFoundError(f"Config not found: {CONFIG_PATH}\nRun train.py first.")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        config = json.load(file)

    model = MNISTLogisticRegression(config["input_size"], config["num_classes"])
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()
    return model, config

def _crop_center(path):
    """Camera photo -> MNIST-style 28x28: subtract paper background, keep the
    main stroke blob, crop, pad to square, resize to 20x20, centre in 28x28."""
    from scipy import ndimage as ndi
    g = np.array(Image.open(path).convert("L"), dtype=np.float32)
    bg = ndi.gaussian_filter(ndi.median_filter(g[::4, ::4], size=15), 1)
    bg = np.array(Image.fromarray(bg).resize(g.shape[::-1], Image.BILINEAR))
    d = ndi.gaussian_filter(np.clip(bg - g, 0, None), 1.5)   # dark ink on lighter paper
    d[:8], d[-8:], d[:, :8], d[:, -8:] = 0, 0, 0, 0
    if d.max() < 1e-6:
        return None
    mask = d > 0.35 * d.max()
    lab, n = ndi.label(mask, structure=np.ones((3, 3)))
    sizes = ndi.sum(mask, lab, range(1, n + 1))
    edge = set(np.unique(np.concatenate([lab[:15].ravel(), lab[-15:].ravel(),
                                         lab[:, :15].ravel(), lab[:, -15:].ravel()])))
    ok = [i for i in range(1, n + 1) if i not in edge] or list(range(1, n + 1))
    mask = lab == max(ok, key=lambda i: sizes[i - 1])         # drop stray marks
    ys, xs = np.where(mask)
    crop = (d * mask)[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = crop.shape
    side = max(h, w)
    sq = np.zeros((side, side), dtype=np.float32)
    sq[(side - h) // 2:(side - h) // 2 + h, (side - w) // 2:(side - w) // 2 + w] = crop
    small = Image.fromarray((sq / sq.max() * 255).astype(np.uint8)).resize((20, 20), Image.LANCZOS)
    out = np.zeros((28, 28), dtype=np.float32)
    out[4:24, 4:24] = np.array(small, dtype=np.float32) / 255.0
    cy, cx = ndi.center_of_mass(out)
    return np.clip(ndi.shift(out, (14 - cy, 14 - cx), order=1), 0, 1)

def preprocess_image(image_path, invert=None, raw=False):
    array = None if raw else _crop_center(image_path)
    if array is None:  # --raw, or nothing found: original behaviour
        image = ImageOps.autocontrast(Image.open(image_path).convert("L")).resize((28, 28))
        array = np.array(image, dtype=np.float32) / 255.0
        if invert is None:
            invert = array.mean() > 0.5
        if invert:
            array = 1.0 - array
    return torch.tensor(array.reshape(1, 784), dtype=torch.float32)

def predict_tensor(model, x):
    with torch.no_grad():
        probabilities = torch.softmax(model(x), dim=1)
    return probabilities.argmax(dim=1), probabilities

def predict(image_path, invert=None, raw=False):
    model, _ = load_model()
    x = preprocess_image(image_path, invert, raw)
    labels, probabilities = predict_tensor(model, x)
    digit = int(labels[0])
    probabilities = probabilities[0].numpy()

    print("\nPrediction\n----------")
    print(f"Image: {image_path}")
    print(f"Predicted digit: {digit}")
    print(f"Confidence: {probabilities[digit]:.2%}")
    print("\nAll class probabilities:")
    for d, p in enumerate(probabilities):
        print(f"Digit {d}: {p:.2%}")
    return digit

def demo():
    import matplotlib.pyplot as plt
    from data_prep import prepare_data

    _, _, X_test, y_test = prepare_data()
    model, _ = load_model()
    idx = np.random.default_rng().choice(len(X_test), size=9, replace=False)
    labels, probs = predict_tensor(model, torch.tensor(X_test[idx], dtype=torch.float32))

    fig, axes = plt.subplots(3, 3, figsize=(7, 7))
    for ax, k, pred, pr in zip(axes.flat, idx, labels, probs):
        true, pred = int(y_test.to_numpy()[k]), int(pred)
        ax.imshow(X_test[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"True {true} | Pred {pred} ({pr[pred]:.0%})",
                     color="green" if true == pred else "red")
        ax.axis("off")
        print(f"Test #{k}: true={true} predicted={pred} confidence={pr[pred]:.1%}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "predict_demo.png", dpi=150)
    plt.close()
    print("Saved outputs/predict_demo.png")

def main():
    parser = argparse.ArgumentParser(
        description="Predict an MNIST digit using the saved logistic regression model."
    )
    parser.add_argument("--image", help="Path to a digit image (omit for a demo).")
    parser.add_argument("--invert", action="store_true",
                        help="Force color inversion (default: auto-detect).")
    parser.add_argument("--raw", action="store_true",
                        help="Skip crop/centre; plain resize to 28x28 (original behaviour).")
    args = parser.parse_args()

    if args.image is None:
        demo()
        return

    image_path = Path(args.image)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    predict(image_path, invert=True if args.invert else None, raw=args.raw)

if __name__ == "__main__":
    main()