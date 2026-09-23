"""
task2/models/domain_discriminator.py
Gradient Reversal Layer (GRL) and Domain Discriminator for DANN and CDAN.
"""
import torch
import torch.nn as nn
from torch.autograd import Function
import numpy as np


class GradientReversalFunction(Function):
    """
    Custom autograd Function implementing the Gradient Reversal Layer.
    Forward pass: identity.
    Backward pass: multiply gradient by -alpha to reverse it.
    """
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.save_for_backward(torch.tensor(alpha))
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        alpha = ctx.saved_tensors[0].item()
        return -alpha * grad_output, None


class GradientReversal(nn.Module):
    """Gradient Reversal Layer wrapper with dynamic alpha."""
    def __init__(self):
        super().__init__()
        self.alpha = 0.0

    def set_alpha(self, alpha):
        self.alpha = alpha

    def forward(self, x):
        return GradientReversalFunction.apply(x, self.alpha)


def grl_schedule(current_epoch, max_epochs):
    """
    Computes GRL alpha using the schedule from the DANN paper:
        p = current_epoch / max_epochs
        alpha(p) = 2 / (1 + exp(-10*p)) - 1
    Increases from ~0 at start to ~1 at end.
    """
    p = current_epoch / max(max_epochs, 1)
    return 2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0


class DomainDiscriminator(nn.Module):
    """
    Domain discriminator for DANN:
      Linear(in_dim -> 256) -> ReLU -> Dropout(0.5) -> Linear(256 -> 2)
    Attached to feature space via a Gradient Reversal Layer.

    Args:
        in_dim: input feature dimension (512 for DANN, 512*7=3584 for CDAN).
    """
    def __init__(self, in_dim=512):
        super().__init__()
        self.grl = GradientReversal()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, 2)
        )
        # Initialize weights
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.grl(x)
        return self.net(x)

    def set_alpha(self, alpha):
        self.grl.set_alpha(alpha)
