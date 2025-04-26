import torch
from PIL import Image
from torchvision import transforms
import config
from tqdm import tqdm  # Import tqdm for progress bar
from model import GenreCNNModel  # Make sure to import your model class
import os

# Same transform as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(config.image_size),
    transforms.ToTensor()
])

def inferloader(list_of_img_paths):
    """
    Converts a list of image paths to a batch of images suitable for the model.
    """
    images = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path).convert("L")
        image = transform(image).unsqueeze(0)  # Add batch dimension
        images.append(image)
    # Stack all images into a single batch tensor
    return torch.cat(images, dim=0)

def load_model(model_path, num_classes, device):
    """
    Load the trained model weights from the checkpoint file.
    """
    model = GenreCNNModel(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using untrained model.")
    model.eval()  # Set the model to evaluation mode
    model.to(device)  # Move the model to the correct device
    return model

def classify_genres(list_of_img_paths, model=None, model_path='checkpoints/final_weights.pth', device=None):
    """
    Given a list of image paths, predicts the genre of each image.
    """
    # Set the device to GPU if available, else fall back to CPU
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model from checkpoint if not provided
    if model is None:
        print("Loading model from checkpoint...")
        model = load_model(model_path, num_classes=len(config.genres), device=device)
    else:
        print("Using provided model...")

    # Convert image paths to batch of tensors
    image_batch = inferloader(list_of_img_paths).to(device)

    with torch.no_grad():
        output = model(image_batch)  # Get model predictions
        _, predicted = torch.max(output, 1)  # Get the class with max probability

    # Map predicted indices to genre names
    predicted_genres = [config.genres[idx] for idx in predicted.cpu().numpy()]
    return predicted_genres
