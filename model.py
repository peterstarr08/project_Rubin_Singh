import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import os

# Load configuration from config.json
config_path = os.path.join(os.path.dirname(__file__), "..", "config.py")
import config  # Import the config.py file directly

class GenreCNNModel(nn.Module):
    def __init__(self, num_classes):
        super(GenreCNNModel, self).__init__()

        # Simple CNN architecture
        self.conv1 = nn.Conv2d(1, config.conv1_out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(config.conv1_out_channels, config.conv2_out_channels, kernel_size=3, stride=1, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)

        # Calculate the flattened size dynamically
        self._flattened_size = self._get_flattened_size()

        self.fc1 = nn.Linear(self._flattened_size, config.fc1_out_features)
        self.fc2 = nn.Linear(config.fc1_out_features, num_classes)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, *config.image_size)
            x = self.pool(F.relu(self.conv1(x)))
            x = self.pool(F.relu(self.conv2(x)))
            return x.view(1, -1).size(1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
