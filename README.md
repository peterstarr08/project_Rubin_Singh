# 🎵 Music Genre Classification (Rubin Singh Project)

---

## 📜 Project Proposal: CNN-Based Music Genre Classification

**Rubin Singh**  
ID: **20221227**

### Problem Statement
Music genre classification is essential for streaming services and recommendation systems. Manual tagging is subjective and inconsistent given the broadness of subgenres. Some songs have a unique sound that defies genre, making classification challenging.  
This project aims to **automate genre classification** using a **CNN model** on **spectrograms** extracted from 30-second audio clips. This project is a **first step** towards making accurate *“feel”* based classification of songs.

---

### Input/Output:

- **Input**: 30-second audio file converted into a spectrogram image (432×288 grayscale or RGB, includes white padding from dataset).
- **Output**: Predicted genre label (Blues, Classical, Country, Disco, Hip-hop, Jazz, Metal, Pop, Reggae, Rock).

---

### 📊 Data Source

- **Dataset**: [GTZAN Dataset (Kaggle)](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification) – 1,000 tracks across 10 genres.

---

### 🏛️ Model Architecture

Our CNN consists of **three convolutional blocks** (Conv2D → ReLU → Dropout → MaxPool), followed by **two fully connected layers** and a final **output layer**.

- **Channels Progression**: 1 → 64 → 128 → 256
- **Dropout**: 30% after conv layers, 50%/30% after FC layers
- **Output**: 10 genres predicted using Softmax activation

This lightweight structure is optimized for fast convergence on spectrogram images while maintaining high accuracy. More details later at bottom.

---

### 🎯 Justification

CNNs are ideal for spectrograms, automatically recognizing frequency patterns without manual feature engineering.  
Beyond genre classification, this model is a **stepping stone** toward discovering songs with similar *sound characteristics*, regardless of predefined genres.

---

## 📂 Project Structure

```
/project_Rubin_Singh/
│
├── checkpoints/            # Model checkpoints (saved weights)
├── data/                   # Spectrogram images, organized by genre
│     ├── blues
│     │      ├── blues00000.png
│     │      ├── blues00001.png
│     ├── classical
│     │      ├── classical00000.png
│     │      ├── classical00001.png
│     ├── ... (other genres)
│
├── config.py                # Configuration file (all settings)
├── dataset.py               # Dataset loader and dataloader utilities
├── model.py                 # CNN model definition
├── train.py                 # Training script
├── predict.py               # Prediction script
├── main.py                  # Full training session with validation after each epoch
├── interface.py             # Simple interface to interact with training/prediction
├── README.md                # 📄 (this file)
└── environment.yml          # 📦 Conda environment file (dependencies)
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
  - MaxPooling (2×2)
- **Block 2**:
  - Conv2D (64 → 128)
  - ReLU Activation
  - Dropout (30%)
  - MaxPooling (2×2)
- **Block 3**:
  - Conv2D (128 → 256)
  - ReLU Activation
  - Dropout (30%)
  - MaxPooling (2×2)

### Fully Connected Layers:
- Flatten after final conv block
- FC1: Linear → 256 → ReLU → Dropout (50%)
- FC2: Linear → 128 → ReLU → Dropout (30%)
- Output Layer: Linear → `10` (genres)

---

## 🚀 Training

- Trains using the settings in `config.py`.
- Trained for **200 epochs**.
- Uses **Cross-Entropy Loss** and **Adam Optimizer**.
- Periodically saves the best model weights to `/checkpoints/` under `final_weights.pth`.

Training can be started with:

```bash
python main.py
```

---

## 📌 Additional Notes

- ✅ **Trained weights** are stored as `final_weights.pth`.  
  Running a new train will **overwrite** it with the newly trained weights.

- ✅ **Dataset loader** includes a `genre_dataloader` function that generates a dataloader for a given dataset.  
  Since the expected format was ambiguous, for now a **prepared full dataset loader** is provided for evaluation use.

- ✅ **predict.py** takes an image path for prediction.  
  It **explicitly** builds a model and loads the checkpoint weights inside the script, so re-training (`main.py`) will automatically update future predictions.

- ✅ **main.py** is used to **retrain** the model, validating on a validation split after every epoch.

- ✅ The `environment.yml` file contains the **full Conda environment** setup required to run the project.

- ✅ **Comments** are provided wherever necessary for clarity.

---

## 🧪 Sample Run of `predict.py`

```
Path: data\rock\rock00079.png | Expected: rock | Predicted: rock
Path: data\pop\pop00061.png | Expected: pop | Predicted: pop
Path: data\classical\classical00010.png | Expected: classical | Predicted: classical
Path: data\reggae\reggae00051.png | Expected: reggae | Predicted: reggae
Path: data\country\country00041.png | Expected: country | Predicted: country
...
Path: data\metal\metal00082.png | Expected: metal | Predicted: metal
Path: data\reggae\reggae00057.png | Expected: reggae | Predicted: reggae
Path: data\country\country00039.png | Expected: country | Predicted: country
Path: data\disco\disco00016.png | Expected: disco | Predicted: disco
Path: data\pop\pop00018.png | Expected: pop | Predicted: pop

Accuracy on 200 random samples: **95.80%**
```

---
