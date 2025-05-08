# generate_synthetic_samples.py

import torch
import random
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def generate_synthetic_samples(dataset, num_samples=1000):
    """
    Generate synthetic samples by applying random transformations to existing samples.
    """
    synthetic_samples = []
    for _ in range(num_samples):
        idx = random.randint(0, len(dataset) - 1)
        img, target = dataset[idx]
      
        transform = transforms.RandomHorizontalFlip(p=1)
        img = transform(img)
        synthetic_samples.append((img, target))
    return synthetic_samples

def create_synthetic_loader(dataset, num_samples=1000, batch_size=16):
    """
    Create a DataLoader for synthetic data.
    """
    synthetic_samples = generate_synthetic_samples(dataset, num_samples)
    synthetic_loader = DataLoader(synthetic_samples, batch_size=batch_size, shuffle=True)
    return synthetic_loader
