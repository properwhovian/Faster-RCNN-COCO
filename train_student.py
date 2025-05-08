# knowledge_distillation/train_student.py

import torch
import torch.optim as optim
from student_model import StudentModel
from distillation import distillation_loss
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Initialize student model and optimizer
student_model = StudentModel(num_classes=80)
optimizer = optim.Adam(student_model.parameters(), lr=0.001)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student_model.to(device)

# Prepare the dataset and DataLoader
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                     std=[0.229, 0.224, 0.225])])
train_dataset = datasets.CocoDetection(root="path/to/train/images", annFile="path/to/train/annotations.json", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

# Define the teacher model 
teacher_model = models.resnet50(pretrained=True)
teacher_model.eval()  # Set teacher model to evaluation mode
teacher_model.to(device)

# Training loop for the student model
for epoch in range(10):
    student_model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)

        # Forward pass through teacher model
        with torch.no_grad():
            teacher_outputs = teacher_model(inputs)

        # Forward pass through student model
        student_outputs = student_model(inputs)

        # Compute the distillation loss
        loss = distillation_loss(student_outputs, teacher_outputs)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader)}")

# Save the trained student model
torch.save(student_model.state_dict(), "student_model.pth")
