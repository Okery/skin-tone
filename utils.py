import math

import torch
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms, models


class SkinDetector(nn.Module):
    """
    The pretrained model is from https://github.com/WillBrennan/SemanticSegmentation
    """
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
    
    
def show(image, mask=None, skin_tone=0, save=""):
    """
    Arguments:
        image (tensor[3, H, W]): RGB channels, value range: [0.0, 1.0]
        mask: shape=[1, H, W], dtype=torch.float
        save (str): path where to save the figure
    """
    assert skin_tone > -1, "skin_tone should > -1"
    
    image = image.clone()
    
    if mask is not None:
        mask = mask.float().cpu()
        beta = (skin_tone + 1) ** 2
        if beta == 1:
            v1 = image * mask
        else:
            v1 = torch.log(image * mask * (beta - 1) + 1) / math.log(beta)
        v2 = image * (1 - mask)
        image = v1 + v2
            
    image = image.clamp(0, 1)
    H, W = image.shape[-2:]
    fig = plt.figure(figsize=(W / 72, H / 72))
    ax = fig.add_subplot(111)
    
    im = image.numpy()
    ax.imshow(im.transpose(1, 2, 0)) # RGB
    ax.set_title("W: {}   H: {}".format(W, H))
    ax.axis("off")

    if save:
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(save)
    plt.show()
    
    