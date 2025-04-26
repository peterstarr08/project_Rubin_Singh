from model import GenreCNNModel as TheModel 

from train import train_model as the_trainer

from predict import classify_genres as the_predictor

from dataset import MusicGenreDataset as TheDataset

#If a dataloader function is needed, check datraset.py for the dataloader function.
from dataset import eval_dataloader as the_dataloader

from config import batch_size as the_batch_size 
from config import epochs as total_epochs 
