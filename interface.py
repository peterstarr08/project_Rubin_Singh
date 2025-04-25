import os
import torch
import random
import torch.optim as optim
from model import GenreCNNModel as TheModel
from dataset import MusicGenreDataset as TheDataset, default_transform
from dataset import get_dataloader as the_dataloader
from train import train_model as the_trainer
from predict import predict_image as the_predictor
import config
from torch.utils.data import Subset, DataLoader
from tqdm import tqdm  # Add this import at the top

# Manual stratified split
def split_dataset(dataset, split_ratio):
    label_to_indices = {}
    for idx, (_, label, _) in enumerate(dataset.samples):
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
    
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
        latest_checkpoint = checkpoint_files[-1]
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

    full_dataset = TheDataset(config.dataset_path, config.genres, default_transform)
    train_dataset, val_dataset = split_dataset(full_dataset, config.training_split)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)

    model = TheModel(num_classes=len(config.genres))

    checkpoint_dir = os.path.dirname(config.checkpoint_path)
    load_checkpoint(model, checkpoint_dir)

    if config.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=config.learning_rate, momentum=config.momentum, weight_decay=config.weight_decay)

    if config.loss_function == "CrossEntropy":
        loss_fn = torch.nn.CrossEntropyLoss()

    print("\n--- Training Model ---")
    the_trainer(model, config.epochs, train_loader, loss_fn, optimizer)

    print("\n--- Running Predictions on Validation Set ---")
    correct = 0
    total = 0

    # Flatten all paths and literals from the val_loader
    all_paths = []
    all_labels = []

    for _, _, paths, literals in val_loader:
        all_paths.extend(paths)
        all_labels.extend(literals)

    # Progress bar
    for i in tqdm(range(len(all_paths)), desc="Validating", ncols=100):  # Adding ncols for better display
        path = all_paths[i]

        # Check if the file exists before proceeding
        if not os.path.exists(path):
            print(f"Skipping invalid path: {path}")
            continue
        
        # Get the prediction for the current image
        prediction = the_predictor(model, path)  # Use predict_image directly
        
        # Compare the prediction with the actual label
        if prediction == all_labels[i]:
            correct += 1
        total += 1

    accuracy = (correct / total) * 100
    print(f"Validation Accuracy: {accuracy:.2f}%")