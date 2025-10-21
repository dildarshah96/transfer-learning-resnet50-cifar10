
# CIFAR-10/100 Image Classification with ResNet-50

This project demonstrates how to fine-tune a **ResNet-50** model pretrained on **ImageNet** for **CIFAR-10** image classification.  
It includes a full training and evaluation pipeline with data augmentation, dynamic learning rate scheduling, and performance visualization.
```
## 🚀 Features

- ✅ **Transfer Learning** using pretrained ResNet-50 (`torchvision.models.resnet50`)
- 🧠 **Fine-tuning** on CIFAR-10 (10 classes)
- 🎨 **Data Augmentation** (random crop, resize, horizontal flip, normalization)
- 📉 **Dynamic Learning Rate Scheduler** with staged decay
- ⚙️ **Gradient Clipping** for stable training
- 📊 **Training Diagnostics** (loss and accuracy plots)
- 🧾 **Classification Report** (Precision, Recall, F1-score)
- 💾 **Model Saving** and reproducibility-friendly setup

```

## 🧩 **Project Structure**
```
resnet50-cifar10-finetune/
│
├── cifar10_train.py # Main training and evaluation script
├── data/ # CIFAR-10 dataset (auto-downloaded)
├── results/ # Folder for plots and saved model (optional)
├── requirements.txt # Python dependencies
└── README.md
```

## ⚙️ Setup Instructions

1️⃣ **Clone the repository**
```
bash
git clone https://github.com/yourusername/resnet50-cifar10-finetune.git
cd resnet50-cifar10-finetune
```
2️⃣ **Create a virtual environment**
```
python -m venv venv
source venv/bin/activate    # On macOS/Linux
venv\Scripts\activate       # On Windows
```
3️⃣ **Install dependencies**
```
pip install -r requirements.txt
```
**Example requirements.txt:**
```
torch
torchvision
numpy
matplotlib
seaborn
scikit-learn
```
🏋️‍♂️ **Training the Model**
```
Run the training pipeline:

python cifar10_train.py
```
This will:

1. Download and augment CIFAR-10

2. Train for 90 epochs with staged learning rate decay

3. Save the trained model as cifar10_model.pt

4. Plot loss and accuracy curves

5 Print classification metrics
```
📈 **Example Output**
```
Classification Report
              precision    recall  f1-score   support

           0       0.91      0.89      0.90      1000
           1       0.95      0.94      0.95      1000
Model Summary
> Final Test Accuracy: 93.472%
Total parameters: 23500000
Trainable parameters: 23500000
Total run time: 1520.33 seconds
```
💾 **Saving & Loading the Model**
```
To load your saved model later:
from torchvision.models import resnet50
import torch
import torch.nn as nn

model = resnet50(weights=None)
model.fc = nn.Linear(2048, 10)
model.load_state_dict(torch.load('cifar10_model.pt'))
model.eval()
```
## 🔗 Links
```
- [View Notebook on Kaggle](https://www.kaggle.com/code/syeddildarshah/baseline-model-resnet-50-implentation)
