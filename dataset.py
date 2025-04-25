import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config  # Uses config.py

def genre_to_label(genre, genre_list):
    return genre_list.index(genre)

class MusicGenreDataset(Dataset):
    def __init__(self, dataset_path, genres, transform=None):
        self.dataset_path = dataset_path
        self.genres = genres
        self.transform = transform
        self.samples = []

        # Collect image paths and labels
        for genre in genres:
            genre_path = os.path.join(dataset_path, genre)
            if not os.path.exists(genre_path):
                continue  # Skip if genre folder doesn't exist
            for filename in os.listdir(genre_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((os.path.join(genre_path, filename), genre_to_label(genre, genres), genre))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, literal_label = self.samples[idx]
        image = Image.open(img_path).convert("L")  # Force grayscale

        if self.transform:
            image = self.transform(image)

        return image, label, img_path, literal_label

# Define standard transform
default_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),         # Force grayscale
    transforms.Resize(config.image_size),                # Resize
    transforms.ToTensor()                                # To tensor
])

# Function to get DataLoader
def get_dataloader(batch_size=config.batch_size, shuffle=config.shuffle):
    dataset = MusicGenreDataset(config.dataset_path, config.genres, default_transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader
