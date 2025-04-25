import torch
from PIL import Image
from torchvision import transforms
import config
from tqdm import tqdm  # Import tqdm for progress bar

# Same transform as training
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize(config.image_size),
    transforms.ToTensor()
])

def predict_image(model, image_path, device="cpu"):
    model.eval()
    image = Image.open(image_path).convert("L")
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output.data, 1)

    predicted_label_idx = predicted.item()
    predicted_genre = config.genres[predicted_label_idx]
    return predicted_genre
