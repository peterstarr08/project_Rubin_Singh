import torch
from PIL import Image
from torchvision import transforms
import config
from tqdm import tqdm  # Import tqdm for progress bar
from model import GenreCNNModel  # Import your model
from dataset import MusicGenreDataset  # <<== NEW: Import your dataset class
import os

# Function to load the model
def load_model(model_path, num_classes, device):
    model = GenreCNNModel(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using untrained model.")
    model.eval()
    model.to(device)
    return model

# Function for image transformation (same as in Dataset)
def inferloader(list_of_img_paths, transform):
    images = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0)
        images.append(image)
    return torch.cat(images, dim=0)

# Main classification function
def classify_genres(list_of_img_paths, model=None, model_path='checkpoints/final_weights.pth', device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate dataset to get transform and sorted genre names
    dataset = MusicGenreDataset(dataset_path=config.dataset_path)
    genres = dataset.genres
    transform = dataset.transform  # Use the same transform as the Dataset class

    # Load model from checkpoint if not provided
    if model is None:
        print("Loading model from checkpoint...")
        model = load_model(model_path, num_classes=len(genres), device=device)
    else:
        print("Using provided model...")

    # Convert image paths to batch of tensors
    image_batch = inferloader(list_of_img_paths, transform).to(device)

    with torch.no_grad():
        output = model(image_batch)
        _, predicted = torch.max(output, 1)

    # Use dataset's genre list to map indices
    predicted_genres = [genres[idx] for idx in predicted.cpu().numpy()]
    return predicted_genres
