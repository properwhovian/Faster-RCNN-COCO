# knowledge_distillation/report_3_metrics.py

import torch
from student_model import StudentModel
from evaluate_student import evaluate_student

# Assuming student model is already trained and saved
student_model = StudentModel(num_classes=80)
student_model.load_state_dict(torch.load("student_model.pth"))

# Evaluate the student model
accuracy, precision, recall, f1 = evaluate_student(student_model)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1}")
