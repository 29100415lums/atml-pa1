"""
models/backbones.py
"""
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
import open_clip

class FrozenBackbone(nn.Module):
    def __init__(self, model_type='resnet50', device='cuda'):
        super().__init__()
        self.model_type = model_type
        self.device = device

        if model_type == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V2
            backbone = models.resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            self.embed_dim = 2048
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        elif model_type == 'vit_b_16':
            weights = ViT_B_16_Weights.IMAGENET1K_V1
            self.backbone = models.vit_b_16(weights=weights)
            self.backbone.heads = nn.Identity()
            self.embed_dim = 768
            self.mean = [0.485, 0.456, 0.406]
            self.std = [0.229, 0.224, 0.225]

        elif model_type == 'clip_vit_b32':
            clip_model, _, _ = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
            self.backbone = clip_model.visual
            self.clip_model = clip_model
            self.embed_dim = 512
            self.mean = [0.48145466, 0.4578275, 0.40821073]
            self.std = [0.26862954, 0.26130258, 0.27577711]
            self.tokenizer = open_clip.get_tokenizer('ViT-B-32')

        for param in self.parameters():
            param.requires_grad = False
        self.eval()

    def forward(self, x):
    # Automatically resize 96x96 STL-10 images to 224x224 for ViT & CLIP
        if x.shape[-2:] != (224, 224):
            x = F.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)

        with torch.no_grad():
            if self.model_type == 'resnet50':
                out = torch.flatten(self.backbone(x), 1)
            elif self.model_type == 'vit_b_16':
                out = self.backbone(x)
            elif self.model_type == 'clip_vit_b32':
                out = F.normalize(self.backbone(x), dim=-1)
        return out


class ClassifierHead(nn.Module):
    def __init__(self, embed_dim, num_classes=10):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


def train_classifier_head(backbone, train_loader, val_loader, num_classes=10, epochs=50, lr=1e-3, weight_decay=1e-4, patience=5, device='cuda', save_path=None):
    backbone.eval()
    head = ClassifierHead(backbone.embed_dim, num_classes).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    patience_counter = 0
    best_weights = None

    for epoch in range(epochs):
        head.train()
        for imgs, labels, _ in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            with torch.no_grad():
                feats = backbone(imgs)

            optimizer.zero_grad()
            logits = head(feats)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

        head.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels, _ in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                feats = backbone(imgs)
                logits = head(feats)
                preds = logits.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        val_acc = val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = head.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            break

    if best_weights is not None:
        head.load_state_dict(best_weights)
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(best_weights, save_path)

    return head, best_val_acc


def zero_shot_clip_predict(clip_backbone, imgs, class_names, device='cuda'):
    clip_model = clip_backbone.clip_model.to(device)
    clip_model.eval()

    prompts = [f"a photo of a {c}" for c in class_names]
    tokens = clip_backbone.tokenizer(prompts).to(device)

    with torch.no_grad():
        text_features = F.normalize(clip_model.encode_text(tokens), dim=-1)
        image_features = clip_backbone(imgs.to(device))
        logit_scale = clip_model.logit_scale.exp()
        logits = logit_scale * (image_features @ text_features.T)
        probs = F.softmax(logits, dim=-1)

    return logits, probs
