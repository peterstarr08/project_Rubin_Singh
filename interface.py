from model import GenreCNNModel as TheModel
from dataset import MusicGenreDataset as TheDataset, default_transform, genre_to_label
from dataset import get_dataloader as the_dataloader
from train import train_model as the_trainer
from predict import predict_image as the_predictor

import torch
import config
from torch.utils.data import Subset, DataLoader
import random

# Manual stratified split (simulated without sklearn)
def split_dataset(dataset, split_ratio):
    label_to_indices = {}
    for idx, (_, label, _) in enumerate(dataset.samples):  # FIXED: only 3 elements
        label_to_indices.setdefault(label, []).append(idx)

    train_indices = []
    val_indices = []

    for label, indices in label_to_indices.items():
        random.shuffle(indices)
        split = int(len(indices) * split_ratio)
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])

    return Subset(dataset, train_indices), Subset(dataset, val_indices)

if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    # Initialize dataset and model
    full_dataset = TheDataset(config.dataset_path, config.genres, default_transform)
    train_dataset, val_dataset = split_dataset(full_dataset, config.training_split)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = TheModel(num_classes=len(config.genres))

    # Train the model
    print("\n--- Training Model ---")
    the_trainer(model, train_loader, val_loader)

    # Predict on a few samples from validation set
    print("\n--- Running Predictions on Validation Set ---")
    for images, labels, paths, literals in val_loader:
        prediction = the_predictor(model, paths[0])
        print(f"Predicted: {prediction} | Actual: {literals[0]} | File: {paths[0]}")
        break  # Show only one prediction
