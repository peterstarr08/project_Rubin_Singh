from model import GenreCNNModel as TheModel  # The model class from model.py

from train import train_model as the_trainer  # Training function from train.py

from predict import classify_genres as the_predictor

from dataset import MusicGenreDataset as TheDataset
from dataset import idk_why_this_exists as the_dataloader

from config import batch_size as the_batch_size 
from config import epochs as total_epochs 
