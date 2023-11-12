import torch
import numpy as np

from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large

def get_model(model_name, num_classes=35, pretrained=True, freeze=True):
    if model_name == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
    elif model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1024, out_features=num_classes, bias=True)
        )
    elif model_name == 'mobilenet_v3_large':
        model = mobilenet_v3_large(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=1280, out_features=num_classes, bias=True)
        )
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        #unfreeze the fc layer
        for param in model.classifier.parameters():
            param.requires_grad = True
    return model