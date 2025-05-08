# distillation_evaluation.py

from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def evaluate_model(model, dataloader, device='cuda'):
    """
    Evaluate the model on a given dataset and compute performance metrics.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, boxes, targets in dataloader:
            inputs, boxes, targets = inputs.to(device), boxes.to(device), targets.to(device)
            outputs = model(inputs, boxes)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    return accuracy, precision, recall, f1
