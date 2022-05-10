import torch
import torch.nn as nn
import numpy as np
import torchvision.models as models
import sys
package_path = './efficientnet_pytorch-0.7.1'
sys.path.append(package_path)
from efficientnet_pytorch import EfficientNet

class DRModel(nn.Module):
    def __init__(self):
        super(DRModel, self).__init__()
        
        self.num_classes = 5

    def get_model(num_classes):
        # model = models.inception_v3()
        # # for parameter in self.network.parameters():
        # #     parameter.requires_grad = False

        # num_ftrs = model.AuxLogits.fc.in_features
        # model.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # # Handle the primary net
        # num_ftrs = model.fc.in_features
        # model.fc = nn.Linear(num_ftrs, num_classes)
        model = EfficientNet.from_name('efficientnet-b4')
        model._fc = nn.Linear(model._fc.in_features, 5)
        # model.load_state_dict(torch.load('.Udacity_Blindness_Detection-master/models/model_{}.bin'.format(model_name, 1)))   

        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        return model