# knowledge_distillation/student_model.py

import torch
import torch.nn as nn
import torchvision.models as models

class StudentModel(nn.Module):
    def __init__(self, num_classes=80):
        super(StudentModel, self).__init__()
        # Example: Use a smaller ResNet-18 as the student model compared to ResNet-50 for teacher
        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)
