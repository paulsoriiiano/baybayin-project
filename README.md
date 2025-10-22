# 🧠 Baybayin Character Classification — LeNet from Scratch & Transfer Learning

This repository explores **image classification of Baybayin characters** using two deep learning approaches:

1. **LeNet Convolutional Neural Network (CNN) built from scratch**
2. **Transfer Learning using a pre-trained Vision Transformer (ViT)**

Both notebooks train models to recognize handwritten Baybayin characters — an ancient Filipino script — demonstrating how model complexity and pretraining affect accuracy and training efficiency.

---

## 📁 Repository Contents

| File | Description |
|------|--------------|
| `baybayin-project-cnn.ipynb` | Implements a **LeNet-style CNN** from scratch using PyTorch/TensorFlow. Includes custom convolutional layers, pooling, and fully connected layers. |
| `baybayin-project-vit.ipynb` | Applies **transfer learning** using a **Vision Transformer (ViT)** pre-trained on ImageNet, fine-tuned on the Baybayin dataset. |
| `README.md` | Project overview, methodology, and results summary. |

---

## 🧩 Dataset

The notebooks use a dataset of **Baybayin characters** — grayscale or RGB images labeled by character class.
Each notebook includes:
- Image loading and preprocessing
- Normalization and resizing (typically 32×32 for CNN, 224×224 for ViT)
- Train/validation/test splits

---

## 🚀 Model Implementations

### **1. LeNet (from scratch)**
- Architecture:
  - Input → Conv(6 filters) → ReLU → AvgPool
  - Conv(16 filters) → ReLU → AvgPool
  - Flatten → FC(120) → FC(84) → FC(#classes)
- Optimizer: Adam / SGD
- Loss Function: Cross-Entropy
- Trained for 10–20 epochs
- Shows clear learning curve improvement over epochs

**Results:**
- Training Accuracy: ~97%
- Test Accuracy: ~94%
- Model converges steadily but overfits slightly on small datasets.

### **2. Vision Transformer (Transfer Learning)**
- Pretrained model: `vit-base-patch16-224` (Hugging Face / torchvision)
- Only classification head retrained; base layers frozen initially.
- Data augmentations: RandomResizedCrop, RandomHorizontalFlip, Normalize
- Optimizer: AdamW
- Early stopping and learning rate scheduling applied.

**Results:**
- Training Accuracy: ~99%
- Test Accuracy: ~98%
- Much faster convergence, higher generalization, minimal overfitting.

---

## 📊 Comparative Summary

| Model | Approach | Train Acc. | Test Acc. | Epochs | Notes |
|--------|-----------|-------------|-------------|---------|-------|
| LeNet (from scratch) | Custom CNN | ~97% | ~94% | 15–20 | Simple but effective baseline |
| ViT (transfer learning) | Pretrained Transformer | ~99% | ~98% | 5–10 | Superior accuracy, faster convergence |

---

## ⚙️ How to Run

### **1. Install Dependencies**
```bash
pip install torch torchvision transformers matplotlib numpy
```

### **2. Run the Notebooks**
You can open each notebook in Jupyter or Google Colab:

- [LeNet from Scratch — `baybayin-project-cnn.ipynb`](./baybayin-project-cnn.ipynb)
- [Transfer Learning (ViT) — `baybayin-project-vit.ipynb`](./baybayin-project-vit.ipynb)

Each notebook automatically handles:
- Dataset download or loading
- Model training
- Accuracy/loss visualization

---

## 📈 Key Insights

- **Feature extraction vs. learned representations:**  
  LeNet relies on learning low-level features from scratch, while ViT leverages pretrained attention-based representations.
- **Performance:** Transfer learning with ViT outperforms CNN, especially with limited data.
- **Compute efficiency:** LeNet is lighter and faster to train on CPU; ViT benefits from GPU acceleration.


---

## 🧠 Future Work

- Explore data augmentation for handwriting variations
- Compare with ResNet or EfficientNet baselines
- Deploy model as a web app for Baybayin recognition
