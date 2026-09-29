from pathlib import Path
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.dummy import DummyClassifier
from data_prep import prepare_data, save_dataset_figures, CLASS_NAMES

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 64
LEARNING_RATE = 0.0005
EPOCHS = 40
WEIGHT_DECAY = 0.0001
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def save_text_figure(text, filename, title, figsize=(8, 4)):
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")
    ax.set_title(title, fontsize=14)
    ax.text(0.02, 0.5, text, family="monospace", fontsize=11, va="center")
    fig.savefig(OUTPUT_DIR / filename, dpi=200, bbox_inches="tight")
    plt.close(fig)

X_train, y_train, X_test, y_test = prepare_data()
save_dataset_figures(X_train, y_train, X_test, y_test)

train_dataset = TensorDataset(
    torch.tensor(X_train, dtype=torch.float32),
    torch.tensor(y_train.to_numpy(), dtype=torch.long)
)
test_dataset = TensorDataset(
    torch.tensor(X_test, dtype=torch.float32),
    torch.tensor(y_test.to_numpy(), dtype=torch.long)
)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

class MNISTLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

model = MNISTLogisticRegression().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

n_params = sum(p.numel() for p in model.parameters())
architecture_text = (
    f"{model}\n\nInput size : 784\nOutput size: 10\n"
    f"Trainable parameters: {n_params} (784 x 10 weights + 10 biases)"
)
settings_text = (
    f"Batch size    : {BATCH_SIZE}\nLearning rate : {LEARNING_RATE}\n"
    f"Epochs        : {EPOCHS}\nOptimizer     : Adam (weight_decay={WEIGHT_DECAY})\n"
    f"Loss function : CrossEntropyLoss\nSeed          : {SEED}\nDevice        : {device}"
)
print(architecture_text)
print(settings_text)
(OUTPUT_DIR / "model_architecture.txt").write_text(architecture_text, encoding="utf-8")
(OUTPUT_DIR / "training_settings.txt").write_text(settings_text, encoding="utf-8")
save_text_figure(architecture_text, "model_architecture.png", "Model Architecture")
save_text_figure(settings_text, "training_settings.png", "Training Settings")

losses = []
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X.size(0)
    epoch_loss = total_loss / len(train_dataset)
    losses.append(epoch_loss)
    print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss:.4f}")

model.eval()
y_true, y_pred, y_prob = [], [], []
with torch.no_grad():
    for X, y in test_loader:
        probability = torch.softmax(model(X.to(device)), dim=1)
        y_true.extend(y.numpy())
        y_pred.extend(probability.argmax(dim=1).cpu().numpy())
        y_prob.extend(probability.cpu().numpy())
y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(y_true, y_pred)
baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_accuracy = accuracy_score(y_test, baseline.predict(X_test))
print(f"Test Accuracy: {accuracy:.2%}")
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

report = classification_report(
    y_true, y_pred, target_names=CLASS_NAMES, zero_division=0, digits=4
)
print(report)

save_text_figure(report, "classification_report.png", "Classification Report", figsize=(8, 5))

cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))

recall = cm.diagonal() / cm.sum(axis=1)
best, worst = int(recall.argmax()), int(recall.argmin())
pairs = sorted(
    [(int(cm[i, j] + cm[j, i]), i, j, int(cm[i, j]), int(cm[j, i]))
     for i in range(10) for j in range(i + 1, 10)],
    reverse=True,
)
pair_lines = [
    f"{i} and {j}: {t} errors ({i} predicted as {j}: {a}; {j} predicted as {i}: {b})"
    for t, i, j, a, b in pairs[:5]
]
summary = (
    f"Highest recall: digit {best} ({recall[best]:.2%})\n"
    f"Lowest recall : digit {worst} ({recall[worst]:.2%})\n\n"
    "Most confused digit pairs:\n" + "\n".join(pair_lines)
)
print("\n" + summary)

save_text_figure(
    f"Test accuracy     : {accuracy:.2%}\nBaseline accuracy : {baseline_accuracy:.2%}\n"
    f"Correct           : {int((y_true == y_pred).sum())} / {len(y_true)}",
    "accuracy_baseline.png", "Test vs Baseline Accuracy", figsize=(7, 2.5),
)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), losses, marker="o")
plt.xlabel("Epoch")
plt.ylabel("Training loss")
plt.title("MNIST: Multiclass Logistic Regression")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
plt.close()

rng = np.random.default_rng(SEED)
idx = rng.choice(len(X_test), size=9, replace=False)
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, k in zip(axes.flat, idx):
    t, p = y_true[k], y_pred[k]
    ax.imshow(X_test[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"True: {t}  Predicted: {p}\nConfidence: {y_prob[k, p]:.1%}",
                 color="green" if t == p else "red", fontsize=10)
    ax.axis("off")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "predictions_9.png", dpi=150)
plt.close()

wrong = np.where(y_true != y_pred)[0]
incorrect_lines = []
if len(wrong) > 0:
    chosen = rng.choice(wrong, size=min(3, len(wrong)), replace=False)
    fig, axes = plt.subplots(1, len(chosen), figsize=(4 * len(chosen), 4.5))
    axes = np.atleast_1d(axes)
    for ax, k in zip(axes, chosen):
        t, p = y_true[k], y_pred[k]
        ax.imshow(X_test[k].reshape(28, 28), cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"Test #{k}\nTrue: {t}  Predicted: {p}\n"
                     f"P(pred)={y_prob[k, p]:.1%}  P(true)={y_prob[k, t]:.1%}",
                     color="red", fontsize=10)
        ax.axis("off")
        incorrect_lines.append(f"#{k}: true {t}, predicted {p}")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "incorrect_predictions.png", dpi=150)
    plt.close()
    print("\nIncorrect examples shown:\n" + "\n".join(incorrect_lines))

torch.save(model.state_dict(), OUTPUT_DIR / "mnist_logistic_model.pth")

with open(OUTPUT_DIR / "evaluation.txt", "w", encoding="utf-8") as f:
    f.write(f"Test Accuracy: {accuracy:.4%}\n")
    f.write(f"Baseline Accuracy: {baseline_accuracy:.4%}\n\n")
    f.write(summary + "\n\n")
    if incorrect_lines:
        f.write("Incorrect examples shown:\n" + "\n".join(incorrect_lines) + "\n\n")
    f.write(report)

with open(OUTPUT_DIR / "model_config.json", "w") as f:
    json.dump({"input_size": 784, "num_classes": 10}, f, indent=4)

print("\nResults saved inside outputs/")