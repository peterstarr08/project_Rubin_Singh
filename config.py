# config.py

# Dataset Path and Genres
dataset_path = "data/images_original"
genres = [
    "blues",
    "classical",
    "country",
    "disco",
    "hiphop",
    "jazz",
    "metal",
    "pop",
    "reggae",
    "rock"
]

# Image size (used for resizing images during data loading)
image_size = (432, 288)

# Training Hyperparameters
batch_size = 10
shuffle = True
epochs = 20
learning_rate = 0.001
checkpoint_path = "checkpoints/final_weights.pth"
momentum = 0.9
weight_decay = 1e-4  # L2 regularization (optional, if desired)
optimizer = "Adam"  # Could be Adam, SGD, etc.
loss_function = "CrossEntropy"  # CrossEntropyLoss for multi-class classification

# Model Hyperparameters
conv1_out_channels = 32
conv2_out_channels = 64
fc1_out_features = 128
training_split = 0.8  # 80% training, 20% validation