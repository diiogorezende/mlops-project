import torch.nn as nn

class LinearHead(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.layers(x)