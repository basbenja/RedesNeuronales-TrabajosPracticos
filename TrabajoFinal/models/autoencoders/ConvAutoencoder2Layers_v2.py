import torch.nn as nn

from utils.post_transform_sizes import *

print(post_conv_shape(28, 28, 16, 5))

# Sizes:
# (1, 28, 28)
# (16, 12, 12)
# (32, 5, 5)
class ConvAutoencoder2Layers_v2(nn.Module):
    def __init__(self, add_linear, p):
        super().__init__()
        self._add_linear = add_linear
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5),    # (1,28,28) -> (8,24,24)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2),    # (8,24,24) -> (8,12,12)

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),   # (8,12,12) -> (16,10,10)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2)    # (16,10,10) -> (16,5,5)
        )
        self.linear = nn.Sequential(
            nn.Flatten(),    # (16,5,5) -> 16*5*5
            nn.Linear(in_features=32*5*5, out_features=32*5*5),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Unflatten(dim=1, unflattened_size=(32,5,5)),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=2, output_padding=1
            ),    # (16,5,5) -> (8,12,12)
            nn.ConvTranspose2d(
                in_channels=16, out_channels=1, kernel_size=5, stride=2, output_padding=1
            ),    # (8,12,12) -> (1,28,28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if self._add_linear:
            x = self.linear(x)
        x = self.decoder(x)
        return x