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
