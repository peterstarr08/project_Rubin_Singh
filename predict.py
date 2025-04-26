import torch
from PIL import Image
import config
from model import GenreCNNModel
from dataset import MusicGenreDataset
import os


def load_model(model_path, num_classes, device):
    """
        Load the model weights from the specified path/checkpoints.
        If no weights are found, it will use the untrained model.
        This function is used to load the model for inference.

        Args:
            model_path (str): Path to the model checkpoint. Default is 'checkpoints/final_weights.pth'.
            num_classes (int): Number of classes in the dataset.
            device (str): Device to run the model on ('cuda' or 'cpu').
    """
    model = GenreCNNModel(num_classes=num_classes)
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print(f"Model weights loaded from {model_path}")
    else:
        print(f"Warning: No model found at {model_path}, using untrained model.")
    model.eval()
    model.to(device)
    return model

def inferloader(list_of_img_paths, transform):
    """
        Prepares a batch of images for inference.
        This function takes a list of image file paths and applies the necessary transformations to prepare them for the model.

        Args:
            list_of_img_paths (list): List of image file paths to classify. E.g. ['data\rock\rock00096.png', 'Path: data\hiphop\hiphop00058.png']
            transform (torchvision.transforms): Transformations to apply to the images.
    """

    images = []
    for img_path in list_of_img_paths:
        image = Image.open(img_path)
        image = transform(image).unsqueeze(0)
        images.append(image)
    return torch.cat(images, dim=0)

def classify_genres(list_of_img_paths, model=None, model_path=config.checkpoint_path , device=None):
    """
        Classify genres of a list of images using a pre-trained model.

        This funtion epexts a model to be passed in, or it will load a fresh model. It also looks for checkpoint weights in the
        specified path. If no weights are found, it will use the untrained model.

        Returns a list of predicted genres for the input images. Note: The output is a list of genre names, not labels.
        The order of the genres corresponds to the order of the input images.

        Args:
            list_of_img_paths (list): List of image file paths to classify. E.g. ['data\rock\rock00096.png', 'Path: data\hiphop\hiphop00058.png']
            model (torch.nn.Module, optional): Pre-trained model. If None, a new model will be loaded.
            model_path (str, optional): Path to the model checkpoint. Default is 'checkpoints/final_weights.pth'.
            device (str, optional): Device to run the model on ('cuda' or 'cpu'). If None, it will use 'cuda' if available.

        Outputs:
            list: List of predicted genres for the input images. E.g. ['rock', 'hiphop']
    
    """
    
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    dataset = MusicGenreDataset(dataset_path=config.dataset_path) # This is required to get correct sequence of genres used in the model. Also needed to get the transform for the images.
    genres = dataset.genres # The correct genre sequence is needed to get the correct genre names.
    transform = dataset.transform # The transform is needed to get the correct input size and normalization for the model.

    if model is None:
        print("Loading model from checkpoint...")
        model = load_model(model_path, num_classes=len(genres), device=device)
    else:
        print("Using provided model...")

    image_batch = inferloader(list_of_img_paths, transform).to(device) #Since creating input requires correct transformation, we use the same transform as in the dataset class.

    with torch.no_grad():
        output = model(image_batch)
        _, predicted = torch.max(output, 1)

    predicted_genres = [genres[idx] for idx in predicted.cpu().numpy()]
    return predicted_genres
