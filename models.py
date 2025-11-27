import timm
import torch.nn as nn
from configs import DATASET_PATH, CLASSES, NUM_CLASSES
# ========================
# Model Definition
# ========================
class GIClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=NUM_CLASSES, pretrained=True):
        super(GIClassifier, self).__init__()
        self.model_name = model_name
        
        if model_name == 'resnet50':
            self.model = timm.create_model('resnet50', pretrained=pretrained, num_classes=num_classes)
        elif model_name == 'efficientnet_b3':
            self.model = timm.create_model('efficientnet_b3', pretrained=pretrained, num_classes=num_classes)
        elif model_name == 'vit_base_patch16_224':
            self.model = timm.create_model('vit_base_patch16_224', pretrained=pretrained, num_classes=num_classes)
        elif model_name == 'swin_base_patch4_window7_224':
            self.model = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=num_classes)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
    def forward(self, x):
        return self.model(x)