import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config  # Uses config.py

def genre_to_label(genre, genre_list):
    return genre_list.index(genre)

class MusicGenreDataset(Dataset):
    def __init__(self, dataset_path=config.dataset_path, genres=config.genres):
        self.dataset_path = dataset_path
        self.genres = genres
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),           # Convert to grayscale
            transforms.Resize(config.image_size),
            transforms.ToTensor(),                                 # Scale to [0, 1]
            transforms.Normalize(mean=[0.5], std=[0.5])            # Normalize to [-1, 1]
        ])
        self.samples = []

        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            if not os.path.exists(genre_path):
                continue
            for filename in os.listdir(genre_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(genre_path, filename)
                    label = genre_to_label(genre, genres)
                    self.samples.append((full_path, label, genre))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, literal_label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, label, img_path, literal_label


def genre_dataloader(dataset, batch_size=config.batch_size, shuffle=config.shuffle):
    """
    Create data loaders without splitting the dataset. 
    It assumes the dataset passed is already split into train, validation, and test sets.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


# Example usage
test_dataset = MusicGenreDataset(dataset_path=config.dataset_path, genres=config.genres)
test_loader = genre_dataloader(test_dataset, batch_size=config.batch_size, shuffle=False)
