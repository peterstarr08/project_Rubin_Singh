import torch
from torch.utils.data import random_split
from torch import optim
import matplotlib.pyplot as plt

from model import GenreCNNModel as TheModel
from train import train_model as the_trainer
from train import evaluate
from dataset import MusicGenreDataset as TheDataset
from dataset import genre_dataloader as the_dataloader
from config import batch_size as the_batch_size
from config import epochs as total_epochs
import config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def main():
    # Track accuracies
    train_accuracies = []
    val_accuracies = []

    # Load dataset
    dataset = TheDataset()
    train_size = int(config.training_split['train'] * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    train_loader = the_dataloader(train_dataset, batch_size=the_batch_size, shuffle=True)
    test_loader = the_dataloader(test_dataset, batch_size=the_batch_size, shuffle=False)

    model = TheModel(num_classes=len(dataset.genres)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Starting training...")

    for epoch in range(total_epochs):
        print(f"\nEpoch {epoch + 1}/{total_epochs}:")
        model = the_trainer(model, 1, train_loader, loss_fn, optimizer, device)
        evaluate(model, test_loader, device)


    # Save model
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Final model saved to {config.checkpoint_path}")

    # Plot accuracy
    plt.figure(figsize=(8, 5))
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(val_accuracies, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("accuracy_plot.png")  # Save plot to file
    plt.show()  # Also display it

if __name__ == "__main__":
    main()
