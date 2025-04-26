# config.py

# Dataset Path and Genres
dataset_path = "data"
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
image_size = (128, 128)

# Training Hyperparameters
batch_size = 16
shuffle = True
epochs = 200
learning_rate = 0.001
checkpoint_path = "checkpoints/final_weights.pth"
momentum = 0.9
weight_decay = 1e-5  # L2 regularization (optional, if desired)


log=True

conv1_out_channels = 64
conv2_out_channels = 128
conv3_out_channels = 256
fc1_out_features = 256
fc2_out_features = 128


training_split = {
    "train": 0.9,
    "test": 0.1
}