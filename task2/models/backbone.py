"""
task2/models/backbone.py
ResNet-18 backbone with 7-class head for PACS Task 2.
Enforces the strict BatchNorm policy: running mean/variance frozen at ImageNet values;
only affine scale (gamma) and bias (beta) remain trainable.
"""
import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

NUM_CLASSES = 7
EMBED_DIM = 512  # ResNet-18 penultimate layer dimension


class PACSBackbone(nn.Module):
    """
    ResNet-18 backbone with a custom 7-class linear head.
    All BatchNorm running statistics are frozen at ImageNet pretrained values.
    Affine parameters (gamma, beta) remain trainable.
    """
    def __init__(self, num_classes=NUM_CLASSES, pretrained=True):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        base = resnet18(weights=weights)

        # Remove ImageNet head, keep feature extractor
        self.feature_extractor = nn.Sequential(*list(base.children())[:-1])  # up to avgpool
        self.classifier = nn.Linear(EMBED_DIM, num_classes)

        # Initialize classifier head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        """Returns (logits, features)."""
        f = self.feature_extractor(x)           # (B, 512, 1, 1)
        f = torch.flatten(f, 1)                 # (B, 512)
        logits = self.classifier(f)             # (B, 7)
        return logits, f

    def get_features(self, x):
        """Return only features without logits (used in adaptation losses)."""
        f = self.feature_extractor(x)
        return torch.flatten(f, 1)

    def freeze_bn_running_stats(self):
        """
        Enforces the BatchNorm policy required by the assignment:
        Freeze running mean and variance at ImageNet pretrained values.
        Keep affine scale (weight) and bias trainable.
        Must be called after model.train() in every training iteration.
        """
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.eval()  # Freezes running_mean and running_var; affine params remain grad-enabled
