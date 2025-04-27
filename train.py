import torch
import torch.nn as nn
import torch.optim as optim
import config
import os
from tqdm import tqdm 

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device=None):
    """
        Train the model for a given number of epochs and return the trained model.

        Args
            model (torch.nn.Module): The model to train.
            num_epochs (int): Number of epochs to train the model.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            loss_fn (torch.nn.Module): Loss function to use for training.
            optimizer (torch.optim.Optimizer): Optimizer to use for training.
            device (str, optional): Device to use for training ('cuda' or 'cpu'). If None, will use the default device.


        Returns
            model (torch.nn.Module): The trained model.
            epoch_loss (float): The average loss over the training dataset.
            epoch_accuracy (float): The accuracy of the model on the training dataset.
    """

    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_predictions = 0

        # tqdm progress bar
        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{num_epochs}]", leave=False)
        
        for inputs, labels, _, _ in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

            # Calculate accuracy
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            correct_predictions += (predicted == labels).sum().item()
            total_predictions += labels.size(0)

            # Update tqdm bar description
            loop.set_postfix(loss=running_loss / (loop.n + 1), accuracy=100 * correct_predictions / total_predictions)

        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = 100 * correct_predictions / total_predictions
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_accuracy:.2f}%")

    # Save only the final model checkpoint after all epochs
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Final model saved to {config.checkpoint_path}")

    return model, epoch_loss, epoch_accuracy

def evaluate(model, dataloader, device=None):
    """
        Evaluate the model on a given dataloader and return the accuracy.

        Args:
            model (torch.nn.Module): The model to evaluate.
            dataloader (torch.utils.data.DataLoader): DataLoader for the evaluation dataset.
            device (str, optional): Device to use for evaluation ('cuda' or 'cpu'). If None, will use the default device.

        Returns:
            acc (float): The accuracy of the model on the evaluation dataset. 
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc="Evaluating", leave=False)
        for inputs, labels, _, _ in loop:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loop.set_postfix(acc=100 * correct / total)

    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc