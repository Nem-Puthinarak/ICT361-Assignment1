# Handwritten Digit Recognition
# Project Files
| File | Description |
|---|---|
| `data_prep.py` | Download, validate, and normalise the data; save dataset figures |
| `train.py` | Train, evaluate, visualise, and save the model |
| `predict.py` | Load the saved model and predict without retraining |
| `requirements.txt` | Required packages |
| `outputs/` | Saved model, evaluation results, and figures |

## 3. Installation and Running

```bash
python -m pip install -r requirements.txt
python data_prep.py
python train.py
python predict.py
```
## 4. Dataset and Preprocessing

**Dataset:** `kagglehub.dataset_download("oddrationale/mnist-in-csv")`

| Item | Value |
|---|---|
| Training images | 60,000 |
| Test images | 10,000 |
| Image size | 28 × 28 pixels, 1 grayscale channel |
| Input features | 784 |
| Classes | 10 (digits 0–9) |
| Columns per CSV | 785: one `label` + 784 pixels |
| Files | `mnist_train.csv`, `mnist_test.csv` |
<<<<<<< HEAD
=======

60,000 training images and 10,000 test images.

1. **Image dimensions and class labels:**  
   Each image is 28 × 28 pixels with 1 grayscale channel. The class labels are integers 0–9, representing the ten digit classes.

2. **Why convert each image into 784 values?**  
   28 × 28 = 784, so each image is converted into a flat vector of 784 pixel values. The linear layer expects a flat input vector, and the CSV dataset already stores the images in this format.

3. **Why divide pixel values by 255?**  
   Pixel values range from 0 to 255. Dividing by 255 scales them to the range 0–1, which provides a smaller and consistent input scale, helping produce more stable gradients and faster convergence.

4. **Why must training and test data remain separate?**  
   The training set is used to update the model weights, while the test set must remain unseen during training so it can measure how well the model generalises to unseen data. Mixing the two can causes data leakage (data contamination). It may cheats by memorizing the answers it is supposed to be tested on.

# Training and test shapes.
![dataset shapes](outputs/dataset_shapes.png)


# Sample image for each digit.
![dataset shapes](outputs/sample_digits.png)

## 5. Workflow

1. Download the MNIST CSV files using KaggleHub.
2. Separate the pixel values (`X`) from the labels (`y`).
3. Validate the data and scale pixel values to the range 0–1.
4. Convert the data to tensors and create mini-batches using `DataLoader`.
5. Train a single linear layer using cross-entropy loss and the Adam optimizer.
6. Evaluate the model on the test set and compare its accuracy with a most-frequent-class baseline.
7. Generate a loss plot, confusion matrix, and sample prediction visualisations.
8. Save the trained model and evaluation results to `outputs/`.


## 6. Model and Training

```python
class MNISTLogisticRegression(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

| Tensor | Shape |
|---|---|
| Input pixels | `(64, 784)` |
| Output logits | `(64, 10)` |
| Labels | `(64,)` |

The model uses a single linear layer that maps 784 input pixel values to 10 output logits, one for each digit class (0–9). A batch size of 64 means 64 images are processed at once.

Training uses **cross-entropy loss** with the **Adam optimiser**.

### Model Architecture

The model uses **784 inputs** because each 28 × 28 grayscale image contains 784 pixel values. It uses **10 outputs** because MNIST has 10 digit classes (0–9). Each output is a score (logit) for one digit class.

The model has **7,850 trainable parameters**:

`784 × 10 + 10 = 7,850`

The predicted digit is the class with the highest output score.

### Why CrossEntropyLoss?

`CrossEntropyLoss` is used for multiclass classification. It compares the model's output logits with the correct digit label and penalises incorrect predictions.

It applies `log-softmax` internally, so the model outputs raw logits without needing a separate softmax layer. Labels are stored as integer class indices using `torch.long`.

### Training Settings

| Setting | Value |
|---|---|
| Batch size | 64 |
| Learning rate | 0.0005 |
| Epochs | 40 |
| Optimizer | Adam, `weight_decay=0.0001` |
| Loss | `CrossEntropyLoss` |
| Seed | 42 |

### Training Loss

The training loss decreased from **0.7037** in epoch 1 to **0.3756** in epoch 2 and **0.2500** in epoch 40, which is a **64.5% overall reduction**.

Most of the improvement occurred during the first few epochs, with the loss decreasing from **0.7037 to 0.2962** by epoch 5. After that, the curve gradually flattened. For example, the loss was **0.2719** at epoch 10 and **0.2530** at epoch 30.

This shows that the model learned quickly and then gradually converged. The small improvement during the later epochs suggests that additional training would provide limited benefit for this single linear layer. Training loss is not the same as error percentage, and the training-loss curve alone cannot determine whether the model is overfitting. The test results are used to evaluate performance on unseen data.

![dataset shapes](outputs/model_architecture.png)
![dataset shapes](outputs/training_settings.png)
![dataset shapes](outputs/training_loss.png)
