import os
import torch
import random
import torch.optim as optim
from model import GenreCNNModel as TheModel
from dataset import MusicGenreDataset as TheDataset, default_transform, genre_to_label
from dataset import get_dataloader as the_dataloader
from train import train_model as the_trainer
from predict import predict_image as the_predictor
import config
from torch.utils.data import Subset, DataLoader

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


def load_checkpoint(model, checkpoint_dir):
    if not os.path.exists(checkpoint_dir) or len(os.listdir(checkpoint_dir)) == 0:
        print(f"No checkpoints found in {checkpoint_dir}. Starting training from scratch.")
        return
    
    # List all files in the checkpoint directory
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]

    if checkpoint_files:
        # Sort the files by the epoch number extracted from the filename
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Sort by epoch number
        latest_checkpoint = checkpoint_files[-1]  # Get the most recent checkpoint
        checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)

        print(f"Checkpoint found at {checkpoint_path}. Do you want to continue from this checkpoint? (y/n)")
        response = input().lower()

        if response == 'y':
            model.load_state_dict(torch.load(checkpoint_path))
            print(f"Checkpoint loaded. Continuing training from epoch {latest_checkpoint.split('_')[-1].split('.')[0]}.")
        else:
            print("Starting training from scratch.")
    else:
        print("No checkpoint found in the directory. Starting training from scratch.")


if __name__ == '__main__':
    torch.manual_seed(42)
    random.seed(42)

    # Initialize dataset and model
    full_dataset = TheDataset(config.dataset_path, config.genres, default_transform)
    train_dataset, val_dataset = split_dataset(full_dataset, config.training_split)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = TheModel(num_classes=len(config.genres))

    # Ensure that `config.checkpoint_path` is the directory, not the full file path
    checkpoint_dir = os.path.dirname(config.checkpoint_path)  # Get directory path only

    # Load the latest checkpoint if exists
    load_checkpoint(model, checkpoint_dir)  # Check for the latest checkpoint

    # Choose the optimizer based on config
    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:  # If SGD is chosen
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    # Loss function
    if config.loss_function == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()

    # Train the model
    print("\n--- Training Model ---")
    the_trainer(model, config.epochs, train_loader, loss_fn, optimizer)

    # Validate on all images in the validation set
    print("\n--- Running Predictions on Validation Set ---")
    correct = 0
    total = 0
    for images, labels, paths, literals in val_loader:
        predictions = the_predictor(model, paths)  # Inference for batch
        for i, prediction in enumerate(predictions):
            if prediction == literals[i]:
                correct += 1
            total += 1
        if total >= len(val_dataset):  # Stop after processing all validation data
            break

    accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")
