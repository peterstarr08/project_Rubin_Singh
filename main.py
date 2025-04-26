import torch
from torch import optim
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
    # Load dataset
    dataset = TheDataset()

    # Stratified split
    train_dataset, test_dataset = dataset.stratified_split(train_ratio=config.training_split['train'])

    train_loader = the_dataloader(train_dataset, batch_size=the_batch_size, shuffle=True)
    test_loader = the_dataloader(test_dataset, batch_size=the_batch_size, shuffle=False)

    model = TheModel(num_classes=len(dataset.genres)).to(device)

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)

    print("Starting training...")

    if config.log:
        with open("training_log.txt", "w") as log_file:
            log_file.write("Epoch, Loss, Accuracy, Test Loss\n")

            for epoch in range(total_epochs):
                print(f"\nEpoch {epoch + 1}/{total_epochs}:")
                model, epoch_loss, epoch_accuracy = the_trainer(model, 1, train_loader, loss_fn, optimizer, device)
                epoch_test_loss = evaluate(model, test_loader, device)
                log_file.write(f"{epoch + 1}, {epoch_loss:.4f}, {epoch_accuracy:.4f}, {epoch_test_loss:.4f}\n")
                print(f"Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}, Test Loss: {epoch_test_loss:.4f}")
    else:
        for epoch in range(total_epochs):
            print(f"\nEpoch {epoch + 1}/{total_epochs}:")
            model, epoch_loss, epoch_accuracy = the_trainer(model, 1, train_loader, loss_fn, optimizer, device)
            epoch_test_loss = evaluate(model, test_loader, device)
            print(f"Epoch Loss: {epoch_loss:.4f}, Epoch Accuracy: {epoch_accuracy:.4f}, Test Loss: {epoch_test_loss:.4f}")

    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Final model saved to {config.checkpoint_path}")

if __name__ == "__main__":
    main()
