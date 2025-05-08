# distillation.py

import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def distill_data(train_dataset, distilled_size_ratio=0.5):
    """
    Split the dataset into full and distilled datasets. 
    The distilled dataset will be a subset (distilled_size_ratio) of the full dataset.
    """
    total_size = len(train_dataset)
    distilled_size = int(distilled_size_ratio * total_size)
    remaining_size = total_size - distilled_size

    distilled_dataset, _ = random_split(train_dataset, [distilled_size, remaining_size])
    
    distilled_loader = DataLoader(distilled_dataset, batch_size=16, shuffle=True)
    return distilled_loader

def train_model(model, dataloader, epochs=10, device='cuda'):
    """
    Train the model using the given dataloader and compute the loss.
    """
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        running_loss = 0.0
        for inputs, boxes, targets in dataloader:
            inputs, boxes, targets = inputs.to(device), boxes.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs, boxes)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(dataloader)}")

    return model
