import torch
import torch.nn as nn

NUM_CLASSES = 7
EMBED_DIM = 512

class PACSClassifierHead(nn.Module):
    """
    7-class linear head for PACS Task 3.
    """
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.classifier = nn.Linear(EMBED_DIM, num_classes)
        # Initialize classifier head
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, f):
        """Returns logits."""
        return self.classifier(f)
