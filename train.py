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

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE
)

class MNISTLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)

model = MNISTLogisticRegression().to(device)

loss_fn = nn.CrossEntropyLoss()

optimizer = optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

losses = []

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    for X, y in train_loader:
        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        output = model(X)
        loss = loss_fn(output, y)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * X.size(0)

    epoch_loss = total_loss / len(train_dataset)
    losses.append(epoch_loss)

    print(f"Epoch {epoch + 1}/{EPOCHS} Loss: {epoch_loss:.4f}")

model.eval()

y_true = []
y_pred = []
y_prob = []

with torch.no_grad():
    for X, y in test_loader:
        output = model(X.to(device))
        probability = torch.softmax(output, dim=1)
        prediction = probability.argmax(dim=1)

        y_true.extend(y.numpy())
        y_pred.extend(prediction.cpu().numpy())
        y_prob.extend(probability.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_prob = np.array(y_prob)

accuracy = accuracy_score(y_true, y_pred)

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
baseline_prediction = baseline.predict(X_test)
baseline_accuracy = accuracy_score(y_test, baseline_prediction)

print(f"Test Accuracy: {accuracy:.2%}")
print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

report = classification_report(
    y_true,
    y_pred,
    target_names=CLASS_NAMES,
    zero_division=0
)

print(report)

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png")
plt.close()

plt.figure(figsize=(8, 5))
plt.plot(range(1, EPOCHS + 1), losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss.png")
plt.close()

torch.save(
    model.state_dict(),
    OUTPUT_DIR / "mnist_logistic_model.pth"
)

with open(OUTPUT_DIR / "evaluation.txt", "w") as f:
    f.write(f"Test Accuracy: {accuracy:.4%}\n")
    f.write(f"Baseline Accuracy: {baseline_accuracy:.4%}\n\n")
    f.write(report)

with open(OUTPUT_DIR / "model_config.json", "w") as f:
    json.dump(
        {
            "input_size": 784,
            "num_classes": 10
        },
        f,
        indent=4
    )