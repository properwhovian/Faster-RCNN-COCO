# knowledge_distillation/distillation.py

import torch
import torch.nn as nn

def distillation_loss(y_student, y_teacher, T=2.0, alpha=0.5):
    """
    Calculate the distillation loss.

    Args:
        y_student (Tensor): Output from the student model
        y_teacher (Tensor): Output from the teacher model
        T (float): Temperature for softening the logits
        alpha (float): Weight for the distillation loss

    Returns:
        loss (Tensor): Distillation loss
    """
    # Softened probabilities for teacher and student
    p_teacher = torch.nn.functional.softmax(y_teacher / T, dim=1)
    p_student = torch.nn.functional.softmax(y_student / T, dim=1)

    # Cross-entropy loss between student and teacher
    loss_distill = nn.KLDivLoss()(torch.log(p_student), p_teacher) * (T * T)

    # Cross-entropy loss for the student's hard targets (from ground truth)
    loss_ce = nn.CrossEntropyLoss()(y_student, torch.argmax(p_teacher, dim=1))

    # Final loss: Weighted sum of both losses
    loss = alpha * loss_ce + (1 - alpha) * loss_distill
    return loss
