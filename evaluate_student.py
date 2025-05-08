# knowledge_distillation/evaluate_student.py

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from student_model import StudentModel
from sklearn.metrics import accuracy_score

# Load the trained student model
student_model = StudentModel(num_classes=80)
student_model.load_state_dict(torch.load("student_model.pth"))
student_model.eval()

# Prepare the dataset and DataLoader for evaluation
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
eval_dataset = datasets.CocoDetection(root="path/to/val/images", annFile="path/to/val/annotations.json", transform=transform)
eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)

# Evaluate the model
all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, targets in eval_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass through student model
        outputs = student_model(inputs)

        # Get the predicted labels
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(targets.cpu().numpy())

# Calculate accuracy
accuracy = accuracy_score(all_labels, all_preds)
print(f"Student Model Accuracy: {accuracy * 100:.2f}%")
