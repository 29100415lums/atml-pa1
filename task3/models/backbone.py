import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

EMBED_DIM = 512

class PACSBackbone(nn.Module):
    """
    ResNet-18 backbone for PACS Task 3.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Remove ImageNet head, keep feature extractor
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # up to avgpool

    def forward(self, x):
        """Returns features."""
        f = self.feature_extractor(x)           # (B, 512, 1, 1)
        f = torch.flatten(f, 1)                 # (B, 512)
        return f

    def freeze_bn_running_stats(self):
        """
        Enforces the BatchNorm policy required by the assignment:
        Freeze running mean and variance at ImageNet pretrained values.
        Keep affine scale (weight) and bias trainable.
        Must be called after model.train() in every training iteration.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()
