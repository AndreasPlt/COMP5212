import torch
import numpy as np

from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large#,resnet50,ResNet50_Weights


default_num_features = {    
    'mobilenet_v2': 1280,
    'mobilenet_v3_small': 576,
    'mobilenet_v3_large': 960,
}

'''
Function to get the model for pretraining
Source: https://youtu.be/dQw4w9WgXcQ?si=DWrEaB61AxnhRmZu
For mobilenet_v3 we noticed, that the .classifier replaces the last two convolutions, thereby we had to adjust the input shape of the linear layer. Needs further investigation.
'''
def get_model(model_name, num_in_features = 0, num_classes=35, pretrained=True, freeze=True, unfreeze_last_n=0):
    if num_in_features == 0:
        num_in_features = default_num_features[model_name]
    if model_name == 'mobilenet_v2':
        model = mobilenet_v2(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=num_in_features, out_features=num_classes, bias=True)
        )
    elif model_name == 'mobilenet_v3_small':
        model = mobilenet_v3_small(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=num_in_features, out_features=num_classes, bias=True)
        )
    elif model_name == 'mobilenet_v3_large':
        model = mobilenet_v3_large(pretrained=pretrained)
        model.classifier = torch.nn.Sequential(
            torch.nn.Flatten(1, -1),
            torch.nn.Dropout(p=0.2, inplace=False),
            torch.nn.Linear(in_features=num_in_features, out_features=num_classes, bias=True)
        )
    elif model_name == 'resnet50':
        model = resnet50(weights = ResNet50_Weights.DEFAULT)
        num_ftrs = model.fc.in_features  # Get the number of input features for the fully connected layer
        model.fc = torch.nn.Linear(in_features=num_ftrs,out_features= num_classes, bias=True)  # Replace the fc layer
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")
    if freeze:
        for param in model.parameters():
            param.requires_grad = False
        #unfreeze the fc layer
        for param in model.classifier.parameters():
            param.requires_grad = True
        # Unfreeze the last n layers
        if unfreeze_last_n <= 0:
            return model
        for param in model.features[-unfreeze_last_n:].parameters():
            param.requires_grad = True
    return model
