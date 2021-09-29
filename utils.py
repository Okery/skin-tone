import torch
from torch import nn
from torchvision import transforms, models


class FCNResNet101(nn.Module):
    def __init__(self, score_threshold=0.5, pretrained=False, pretrained_backbone=False):
        super().__init__()
        self.score_threshold = score_threshold

        self.model = models.segmentation.fcn_resnet101(
            aux_loss=True, pretrained=pretrained, pretrained_backbone=pretrained_backbone)
        
        self.model.classifier[4] = nn.Conv2d(512, 1, 1)
        self.model.aux_classifier[4] = nn.Conv2d(256, 1, 1)

    def forward(self, x):
        out = self.model(x)["out"]
        out = out.sigmoid() > self.score_threshold
        return out
    
    