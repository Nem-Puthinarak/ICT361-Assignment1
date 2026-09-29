from pathlib import Path
import json

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import TensorDataset, DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    classification_report
)

from sklearn.dummy import DummyClassifier

from data_prep import (
    prepare_data,
    save_dataset_figures,
    CLASS_NAMES,
    BASE_DIR
)


# ============================================================
# SETTINGS
# ============================================================

SEED = 42

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 40
WEIGHT_DECAY = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

torch.manual_seed(SEED)
np.random.seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


print("Device:", DEVICE)


# ============================================================
# DATA
# ============================================================

X_train, y_train, X_test, y_test = prepare_data()

save_dataset_figures(
    X_train,
    y_train,
    X_test,
    y_test
)


train_dataset = TensorDataset(
    torch.tensor(
        X_train,
        dtype=torch.float32
    ),
    torch.tensor(
        y_train.to_numpy(),
        dtype=torch.long
    )
)


test_dataset = TensorDataset(
    torch.tensor(
        X_test,
        dtype=torch.float32
    ),
    torch.tensor(
        y_test.to_numpy(),
        dtype=torch.long
    )
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)


test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

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


INPUT_SIZE = X_train.shape[1]
NUM_CLASSES = len(CLASS_NAMES)


model = MNISTLogisticRegression(
    INPUT_SIZE,
    NUM_CLASSES
).to(DEVICE)


print("\nModel:")
print(model)


# ============================================================
# LOSS AND OPTIMIZER
# ============================================================

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# SAVE MODEL ARCHITECTURE FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))

ax.axis("off")

architecture = (
    "Multiclass Logistic Regression\n\n"
    "Input\n"
    "784 pixel values\n"
    "(28 × 28)\n\n"
    "↓\n\n"
    "Linear Layer\n"
    "nn.Linear(784, 10)\n\n"
    "↓\n\n"
    "10 Output Logits\n"
    "Digits 0–9"
)

ax.text(
    0.5,
    0.5,
    architecture,
    ha="center",
    va="center",
    fontsize=16
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "model_architecture.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# SAVE TRAINING SETTINGS FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.axis("off")

settings_text = (
    "Training Settings\n\n"
    f"Batch size       : {BATCH_SIZE}\n"
    f"Learning rate    : {LEARNING_RATE}\n"
    f"Epochs           : {EPOCHS}\n"
    f"Weight decay     : {WEIGHT_DECAY}\n"
    f"Optimizer        : Adam\n"
    f"Loss function    : CrossEntropyLoss\n"
    f"Random seed      : {SEED}\n"
    f"Device           : {DEVICE}"
)

ax.text(
    0.5,
    0.5,
    settings_text,
    ha="center",
    va="center",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "training_settings.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# TRAINING
# ============================================================

training_losses = []


print("\nStarting training...")


for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        optimizer.zero_grad()

        logits = model(X_batch)

        loss = loss_fn(
            logits,
            y_batch
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item()
            * X_batch.size(0)
        )

    epoch_loss = (
        running_loss
        / len(train_dataset)
    )

    training_losses.append(
        epoch_loss
    )

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} "
        f"- Loss: {epoch_loss:.6f}"
    )


# ============================================================
# TRAINING LOSS FIGURE
# ============================================================

fig, ax = plt.subplots(figsize=(9, 5))

ax.plot(
    range(1, EPOCHS + 1),
    training_losses
)

ax.set_title(
    "Training Loss Over Epochs"
)

ax.set_xlabel(
    "Epoch"
)

ax.set_ylabel(
    "Training Loss"
)

ax.grid(True)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "training_loss.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# EVALUATION
# ============================================================

model.eval()

y_true = []
y_pred = []
y_prob = []


with torch.no_grad():

    for X_batch, y_batch in test_loader:

        X_batch = X_batch.to(DEVICE)

        logits = model(X_batch)

        probabilities = torch.softmax(
            logits,
            dim=1
        )

        predictions = probabilities.argmax(
            dim=1
        )

        y_true.extend(
            y_batch.numpy()
        )

        y_pred.extend(
            predictions.cpu().numpy()
        )

        y_prob.extend(
            probabilities.cpu().numpy()
        )


y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)


# ============================================================
# ACCURACY
# ============================================================

test_accuracy = accuracy_score(
    y_true,
    y_pred
)


print("\nTest Accuracy:")
print(
    f"{test_accuracy:.2%}"
)


# ============================================================
# MOST-FREQUENT-CLASS BASELINE
# ============================================================

baseline = DummyClassifier(
    strategy="most_frequent"
)

baseline.fit(
    X_train,
    y_train
)

baseline_predictions = baseline.predict(
    X_test
)

baseline_accuracy = accuracy_score(
    y_test,
    baseline_predictions
)


print("\nMost-Frequent-Class Baseline:")
print(
    f"{baseline_accuracy:.2%}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print("\nClassification Report:")
print(report)


report_dict = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)


# ============================================================
# FIND HIGHEST / LOWEST RECALL
# ============================================================

recalls = {
    digit: report_dict[str(digit)]["recall"]
    for digit in range(10)
}

highest_recall_digit = max(
    recalls,
    key=recalls.get
)

lowest_recall_digit = min(
    recalls,
    key=recalls.get
)


print(
    f"Highest recall: digit "
    f"{highest_recall_digit} "
    f"({recalls[highest_recall_digit]:.2%})"
)

print(
    f"Lowest recall: digit "
    f"{lowest_recall_digit} "
    f"({recalls[lowest_recall_digit]:.2%})"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred
)


plt.figure(
    figsize=(9, 7)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)

plt.title(
    "MNIST Confusion Matrix"
)

plt.xlabel(
    "Predicted Digit"
)

plt.ylabel(
    "Actual Digit"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "confusion_matrix.png",
    dpi=200
)

plt.close()


# ============================================================
# MOST COMMON CONFUSIONS
# ============================================================

confusions = []

for actual in range(10):

    for predicted in range(10):

        if actual != predicted:

            confusions.append(
                (
                    cm[actual, predicted],
                    actual,
                    predicted
                )
            )


confusions.sort(
    reverse=True
)


top_confusions = confusions[:10]


print("\nMost Common Confusions:")

for count, actual, predicted in top_confusions:

    print(
        f"Actual {actual} → Predicted {predicted}: "
        f"{count} images"
    )


# ============================================================
# ACCURACY VS BASELINE FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(8, 5)
)

labels = [
    "Most-Frequent Baseline",
    "Logistic Regression"
]

values = [
    baseline_accuracy * 100,
    test_accuracy * 100
]

ax.bar(
    labels,
    values
)

ax.set_ylabel(
    "Accuracy (%)"
)

ax.set_title(
    "Test Accuracy vs Most-Frequent-Class Baseline"
)

ax.set_ylim(
    0,
    100
)

for i, value in enumerate(values):

    ax.text(
        i,
        value + 1,
        f"{value:.2f}%",
        ha="center"
    )

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "accuracy_baseline.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# CLASSIFICATION REPORT FIGURE
# ============================================================

fig, ax = plt.subplots(
    figsize=(10, 7)
)

ax.axis("off")

ax.text(
    0.01,
    0.99,
    report,
    ha="left",
    va="top",
    family="monospace",
    fontsize=11
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "classification_report.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# NINE RANDOM TEST PREDICTIONS
# ============================================================

rng = np.random.default_rng(SEED)

indices = rng.choice(
    len(X_test),
    size=min(9, len(X_test)),
    replace=False
)


fig, axes = plt.subplots(
    3,
    3,
    figsize=(9, 9)
)


for ax, index in zip(
    axes.flat,
    indices
):

    image = X_test[index].reshape(
        28,
        28
    )

    true_label = y_true[index]
    predicted_label = y_pred[index]

    probability = y_prob[
        index,
        predicted_label
    ]

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        f"True: {true_label} | "
        f"Pred: {predicted_label}\n"
        f"Confidence: {probability:.2%}"
    )

    ax.axis("off")


fig.suptitle(
    "Nine Random MNIST Test Predictions",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "predictions_9.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# THREE INCORRECT PREDICTIONS
# ============================================================

incorrect_indices = np.where(
    y_true != y_pred
)[0]


num_incorrect_to_show = min(
    3,
    len(incorrect_indices)
)


fig, axes = plt.subplots(
    1,
    num_incorrect_to_show,
    figsize=(12, 4)
)


if num_incorrect_to_show == 1:
    axes = [axes]


for ax, index in zip(
    axes,
    incorrect_indices[:num_incorrect_to_show]
):

    image = X_test[index].reshape(
        28,
        28
    )

    true_label = y_true[index]
    predicted_label = y_pred[index]

    confidence = y_prob[
        index,
        predicted_label
    ]

    ax.imshow(
        image,
        cmap="gray"
    )

    ax.set_title(
        f"True: {true_label}\n"
        f"Predicted: {predicted_label}\n"
        f"Confidence: {confidence:.2%}"
    )

    ax.axis("off")


fig.suptitle(
    "Examples of Incorrect Predictions",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    OUTPUT_DIR / "incorrect_predictions.png",
    dpi=200,
    bbox_inches="tight"
)

plt.close(fig)


# ============================================================
# SAVE EVALUATION TEXT
# ============================================================

evaluation_text = f"""
MNIST MULTICLASS LOGISTIC REGRESSION
====================================

Dataset
-------
Training images: {len(X_train)}
Test images: {len(X_test)}
Image dimensions: 28 x 28
Input features: {INPUT_SIZE}
Number of classes: {NUM_CLASSES}
Classes: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

Training Settings
-----------------
Batch size: {BATCH_SIZE}
Learning rate: {LEARNING_RATE}
Epochs: {EPOCHS}
Weight decay: {WEIGHT_DECAY}
Optimizer: Adam
Loss function: CrossEntropyLoss
Random seed: {SEED}
Device: {DEVICE}

Results
-------
Test accuracy: {test_accuracy:.4%}
Most-frequent-class baseline: {baseline_accuracy:.4%}

Highest recall:
Digit {highest_recall_digit}: {recalls[highest_recall_digit]:.4%}

Lowest recall:
Digit {lowest_recall_digit}: {recalls[lowest_recall_digit]:.4%}

Classification Report
---------------------
{report}

Most Common Confusions
----------------------
"""


for count, actual, predicted in top_confusions:

    evaluation_text += (
        f"Actual {actual} -> Predicted {predicted}: "
        f"{count} images\n"
    )


with open(
    OUTPUT_DIR / "evaluation.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write(
        evaluation_text
    )


# ============================================================
# SAVE MODEL
# ============================================================

model_path = (
    OUTPUT_DIR
    / "mnist_logistic_model.pth"
)

torch.save(
    model.state_dict(),
    model_path
)


# ============================================================
# SAVE MODEL CONFIGURATION
# ============================================================

config = {
    "input_size": INPUT_SIZE,
    "num_classes": NUM_CLASSES,
    "class_names": CLASS_NAMES,
    "batch_size": BATCH_SIZE,
    "learning_rate": LEARNING_RATE,
    "epochs": EPOCHS,
    "weight_decay": WEIGHT_DECAY,
    "seed": SEED
}


with open(
    OUTPUT_DIR / "model_config.json",
    "w",
    encoding="utf-8"
) as file:

    json.dump(
        config,
        file,
        indent=4
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n========================================")
print("TRAINING AND EVALUATION COMPLETE")
print("========================================")

print(
    f"Test accuracy : {test_accuracy:.2%}"
)

print(
    f"Baseline      : {baseline_accuracy:.2%}"
)

print(
    f"Model saved   : {model_path}"
)

print(
    f"Outputs folder: {OUTPUT_DIR}"
)