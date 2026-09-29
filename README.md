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


`load_csv` also validates that each file exists and is non-empty, contains one `label` column plus 784 pixel columns, has labels in the range 0–9, and has finite pixel values within 0–255.
>>>>>>> d0bf46b193cd2d451fb43ab29a2c50bae5be1c97
