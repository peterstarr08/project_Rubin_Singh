# 🎵 Music Genre Classification (Rubin Singh Project)

This project focuses on **classifying music genres** using a **Convolutional Neural Network (CNN)** trained on **spectrogram images** of songs.

The goal is to predict one of **10 possible genres** given a spectrogram input.

---

## 📂 Project Structure

```
/project_Rubin_Singh/
│
├── checkpoints/            # Model checkpoints (saved weights)
├── data/                   # Spectrogram images, organized by genre
│     ├── blues
│     │      ├── blue00000.png
│     │      ├── blue00001.png
│     ├── classical
│     │      ├── classical00000.png
│     │      ├── classical00001.png
├── config.py                # Configuration file (all settings)
├── dataset.py               # Dataset loader
├── model.py                 # CNN model definition
├── train.py                 # Training script
├── predict.py               # Prediction script
├── main.py                  # Runs a full training session, gives validation after each epoch
├── interface.py             # Simple interface
├── README.md                # 📄 (this file)
└── environment.yml          # Dependencies
```

---

## 🎯 Target Genres

The model is trained to classify the following **10 genres**:

- Blues
- Classical
- Country
- Disco
- HipHop
- Jazz
- Metal
- Pop
- Reggae
- Rock

---

## ⚙️ Configuration (`config.py`)

All project settings are centralized inside `config.py`:

| Setting | Value |
|:--------|:------|
| Dataset Path | `data/` |
| Image Size | `(128, 128)` |
| Batch Size | `16` |
| Shuffle Data | `True` |
| Epochs | `200` |
| Learning Rate | `0.001` |
| Momentum | `0.9` |
| Weight Decay | `1e-5` |
| Log Progress | `True` |
| Train/Test Split | `90% / 10%` |
| Checkpoint Path | `checkpoints/final_weights.pth` |
| CNN Hidden Channels | 64 → 128 → 256 |
| FC Layers | 256 → 128 |

---

## 🧠 Model Architecture (`model.py`)

A clean and simple **Convolutional Neural Network**:

### CNN Blocks:
- **Block 1**:
  - Conv2D (Input channels: 1 → 64)
  - ReLU Activation
  - Dropout (30%)
  - MaxPooling (2x2)

- **Block 2**:
  - Conv2D (64 → 128)
  - ReLU Activation
  - Dropout (30%)
  - MaxPooling (2x2)

- **Block 3**:
  - Conv2D (128 → 256)
  - ReLU Activation
  - Dropout (30%)
  - MaxPooling (2x2)

<!-- (Blocks 4 and 5 were experimented with but commented out for better simplicity and faster convergence.) -->

### Fully Connected Layers:
- Flatten after final conv block
- FC1: Linear → 256 → ReLU → Dropout (50%)
- FC2: Linear → 128 → ReLU → Dropout (30%)
- Output Layer: Linear → `10` (genres)

---

## 🚀 Training

- Trains using the settings in `config.py`.
- Trained for **100 epochs**.
- Uses **Cross-Entropy Loss** and **Adam Optimizer**.
- Periodically saves the best model weights to `/checkpoints/`.

Training can be started with:

```bash
python main.py
```

---

## 📜 Extra Information

- **Spectrograms** are used instead of raw audio (no `librosa`, `scikit`, etc.).
- Data split is **manual** (handled inside training script).
- Minimal external packages: only `torch`, `torchvision`, `torchaudio`, `PIL`, `numpy`.
- Images are resized to `128x128` during loading.

---
