import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models

class DRModel(nn.Module):
    def __init__(self):
        super(DRModel, self).__init__()
        
        self.num_classes = 5

    def get_model(num_classes):
        model = models.inception_v3()
        # for parameter in self.network.parameters():
        #     parameter.requires_grad = False

        num_ftrs = model.AuxLogits.fc.in_features
        model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        return model