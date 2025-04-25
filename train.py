import torch
import torch.nn as nn
import torch.optim as optim
import config
import os

def train_model(model, num_epochs, train_loader, loss_fn, optimizer, device="cpu"):
    model = model.to(device)

    # Ensure the checkpoints directory exists
    os.makedirs(os.path.dirname(config.checkpoint_path), exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        for inputs, labels, _, _ in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        # Print loss for the current epoch
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {running_loss/len(train_loader):.4f}")
        
        # Save the model after every epoch
        epoch_checkpoint_path = f"{config.checkpoint_path.split('.')[0]}_epoch_{epoch+1}.pth"  # Modify path to include epoch number
        torch.save(model.state_dict(), epoch_checkpoint_path)
        print(f"Model saved to {epoch_checkpoint_path}")

    # Final save after training
    torch.save(model.state_dict(), config.checkpoint_path)
    print(f"Final model saved to {config.checkpoint_path}")

    return model

def evaluate(model, dataloader, device="cpu"):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels, _, _ in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    acc = 100 * correct / total
    print(f"Validation Accuracy: {acc:.2f}%")
    return acc  # Changed from ACC to acc, assuming it was a typo
