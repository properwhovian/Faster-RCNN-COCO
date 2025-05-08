# base_model/model.py

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.ops as ops

class SimpleRCNN(nn.Module):
    def __init__(self, backbone, num_classes):
        super(SimpleRCNN, self).__init__()
        self.backbone = backbone
        self.roi_pool = ops.MultiScaleRo
