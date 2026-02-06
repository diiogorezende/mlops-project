import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18(nn.Module):
    def __init__(self, in_channels: int = 3, pretrained: bool = False):
        super().__init__()

        # Carrega os pesos pretreinados se especificado
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.net = resnet18(weights=weights)

        # Ajusta a primeira camada convolucional para o numero de canais de entrada
        if in_channels != 3:
            self.net.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            )

        # Remove o classificador original
        self.net.fc = nn.Identity()

    def forward(self, x):
        return self.net(x)
    