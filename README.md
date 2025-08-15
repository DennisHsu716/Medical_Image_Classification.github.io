# Deep Learning for Medical Image Classification

## 📌 Project Overview
This project demonstrates how to train and evaluate a deep learning model (ResNet18 by default) for medical image classification using PyTorch. The dataset is provided in CSV format, where each row contains an image path and its label. The workflow supports training, validation, and model saving for later inference.

**Data Source**: User-provided medical imaging dataset (CSV format with image_path,label).
**Run in Colab**: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/16av-dWn5nWB6snEu6fk907DwBcgieYQR?usp=sharing)

## 📂 Project Structure
```
med-imaging/  
├── src/  
│   ├── dataset.py        # Custom PyTorch Dataset for medical images  
│   ├── train.py          # Training script  
│   └── evaluate.py       # Evaluation script  
├── config/  
│   └── default.yaml      # Model and training configuration  
├── data/  
│   ├── processed/  
│   │   ├── train.csv     # Training data paths & labels  
│   │   └── val.csv       # Validation data paths & labels  
├── runs/                 # Saved models and logs  
└── requirements.txt      # Dependencies  
```

## 📊 Training Pipeline

1. Prepare Data

* Store medical images in a folder.

* Create ```train.csv``` and ```val.csv``` with columns: ```image_path,label```.

2. Model Training

* ```train.py``` loads data, applies transforms (Resize, Normalize), and trains ResNet18.

* Tracks validation accuracy and saves the best model to ```runs/best_model.pth```.

3. Model Evaluation

* ```evaluate.py``` loads the trained model and computes classification metrics (accuracy, confusion matrix, classification report).

## 🧮 Example CSV Format

| image_path              | label |
|-------------------------|-------|
| data/images/img001.png  | 0     |
| data/images/img002.png  | 1     |

## 🚀 How to Run (Google Colab)
```
# Switch to the project directory
%cd "/content/drive/MyDrive/med-imaging"

# 1) Train model
!python3 src/train.py --config config/default.yaml \
    --train_csv data/processed/train.csv \
    --val_csv   data/processed/val.csv

# 2) Evaluate model
!python3 src/evaluate.py --model runs/best_model.pth \
    --val_csv data/processed/val.csv
```

## 📈 Example Output
```
epoch 1/8 val_acc=1.0000 | sample_preds=[([1], [1])]
epoch 8/8 val_acc=1.0000 | sample_preds=[([1], [1])]
saved best: runs/best_model.pth (acc=1.0000)

Confusion matrix [[TN FP],[FN TP]]:
 [[5 0],
  [0 7]]

Classification report:
               precision    recall  f1-score   support
           0     1.0000    1.0000    1.0000         5
           1     1.0000    1.0000    1.0000         7

```
## 🔧 Future Improvements

* Add more advanced architectures (DenseNet, EfficientNet, Vision Transformer).

* Implement data augmentation (rotation, brightness adjustment, flipping).

* Support multi-label classification.

* Integrate Grad-CAM for model explainability in medical diagnostics.
