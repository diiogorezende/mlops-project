import torch.nn as nn

class SimpleImageClassifier(nn.Module):
    def __init__(self, backbone: nn.Module, head: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x):
        features = self.backbone(x)
        logits = self.head(features)
        return logits