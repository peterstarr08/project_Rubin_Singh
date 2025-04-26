import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import config

class MusicGenreDataset(Dataset):
    def __init__(self, dataset_path=config.dataset_path):
        self.dataset_path = dataset_path
        
        self.genres = sorted([
            folder for folder in os.listdir(dataset_path)
            if os.path.isdir(os.path.join(dataset_path, folder))
        ])

        self.genre_to_label = {genre: idx for idx, genre in enumerate(self.genres)}

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize(config.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        self.samples = []
        for genre in self.genres:
            genre_path = os.path.join(dataset_path, genre)
            for filename in os.listdir(genre_path):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    full_path = os.path.join(genre_path, filename)
                    label = self.genre_to_label[genre]
                    self.samples.append((full_path, label, genre))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, literal_label = self.samples[idx]
        image = Image.open(img_path).convert("L")
        image = self.transform(image)
        return image, label, img_path, literal_label

    def stratified_split(self, train_ratio=0.8, seed=42):
        random.seed(seed)

        genre_to_samples = {genre: [] for genre in self.genres}
        for sample in self.samples:
            _, _, genre = sample
            genre_to_samples[genre].append(sample)

        train_samples = []
        test_samples = []

        for genre, samples in genre_to_samples.items():
            random.shuffle(samples)
            split_idx = int(train_ratio * len(samples))
            train_samples.extend(samples[:split_idx])
            test_samples.extend(samples[split_idx:])

        train_dataset = MusicGenreDataset(self.dataset_path)
        test_dataset = MusicGenreDataset(self.dataset_path)

        train_dataset.samples = train_samples
        test_dataset.samples = test_samples

        return train_dataset, test_dataset

def genre_dataloader(dataset, batch_size=config.batch_size, shuffle=config.shuffle):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)



#This full dataset is used in interface.py to load the dataset for evaluation
full_dataset = MusicGenreDataset(config.dataset_path)
eval_dataloader = genre_dataloader(full_dataset, batch_size=config.batch_size, shuffle=True)