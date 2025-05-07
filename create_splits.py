# data_processing/create_splits.py

import torch
from torch.utils.data import random_split

def create_data_splits(dataset, train_fraction=0.8, val_fraction=0.1):
    """
    Create training, validation, and test splits for the dataset.
    """
    total_size = len(dataset)
    train_size = int(train_fraction * total_size)
    val_size = int(val_fraction * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])
    
    return train_dataset, val_dataset, test_dataset
