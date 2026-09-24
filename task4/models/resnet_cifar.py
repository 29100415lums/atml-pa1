import torch
import torch.nn as nn
from torchvision.models.resnet import resnet18, ResNet18_Weights

def get_cifar_resnet18(num_classes=10):
    """
    Constructs a CIFAR-appropriate ResNet-18.
    - Replaces the 7x7 stride-2 first conv with a 3x3 stride-1 conv.
    - Removes the initial max-pooling layer.
    """
    model = resnet18(weights=None)
    
    # 1. Replace first conv
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    
    # 2. Remove max pooling (by replacing it with an Identity layer)
    model.maxpool = nn.Identity()
    
    # 3. Replace the final fully connected layer for the desired number of classes
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

class ResNet18Penultimate(nn.Module):
    """
    A wrapper around ResNet18 that returns both the penultimate features and the logits.
    """
    def __init__(self, base_model):
        super().__init__()
        # Extract features up to the global average pooling
        self.features = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )
        self.fc = base_model.fc
        
    def forward(self, x):
        f = self.features(x)
        f = torch.flatten(f, 1)
        logits = self.fc(f)
        return logits, f

class ResNet18PROSER(nn.Module):
    """
    PROSER model wrapper that supports Manifold Mixup at layer2 and 
    has 15 output classes (10 known + 5 dummy).
    """
    def __init__(self, base_model, num_known=10, num_dummy=5):
        super().__init__()
        # Up to layer 2
        self.pre_mixup = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1,
            base_model.layer2
        )
        # Layer 3 and 4 + pool
        self.post_mixup = nn.Sequential(
            base_model.layer3,
            base_model.layer4,
            base_model.avgpool
        )
        
        # Original FC weights for known classes, random init for dummy classes
        self.fc = nn.Linear(base_model.fc.in_features, num_known + num_dummy)
        
        # Copy known weights
        self.fc.weight.data[:num_known] = base_model.fc.weight.data
        if base_model.fc.bias is not None:
            self.fc.bias.data[:num_known] = base_model.fc.bias.data
            
    def forward(self, x, mixup=False, mixup_lambda=None, indices=None):
        if not mixup:
            f = self.pre_mixup(x)
            f = self.post_mixup(f)
            f = torch.flatten(f, 1)
            logits = self.fc(f)
            return logits, f
            
        else:
            # Manifold Mixup path
            h = self.pre_mixup(x)
            
            if indices is not None and mixup_lambda is not None:
                # h_tilde = lambda * h_i + (1 - lambda) * h_j
                # Usually indices permutations are passed to ensure y_i != y_j
                h_tilde = mixup_lambda * h + (1 - mixup_lambda) * h[indices]
                h = h_tilde
                
            f = self.post_mixup(h)
            f = torch.flatten(f, 1)
            logits = self.fc(f)
            return logits, f

