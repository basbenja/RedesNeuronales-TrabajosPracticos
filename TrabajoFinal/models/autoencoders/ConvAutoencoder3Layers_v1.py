import torch.nn as nn

# Sizes:
# (1, 28, 28)
# (8, 13, 13)
# (16, 5, 5)
# (32, 1, 1)
class ConvAutoencoder3Layers_v1(nn.Module):
    def __init__(self, add_linear, n, p):
        super().__init__()
        self._add_linear = add_linear
        self._dropout = p
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, padding=1),    # (1,28,28) -> (8,28,28)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2),    # (8,28,28) -> (8,14,14)

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),   # (8,14,14) -> (16,14,14)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2),    # (16,14,14) -> (16,7,7)
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),   # (16,7,7) -> (32,7,7)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2)    # (32,7,7) -> (32,3,3)
        )
        self.linear_encoder = nn.Sequential(
            nn.Flatten(),    # (32,4,4) -> 32*4*4
            nn.Linear(in_features=32*3*3, out_features=n),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(in_features=n, out_features=32*3*3),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3))
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=32, out_channels=16, kernel_size=3, stride=2
            ),    # (32,3,3) -> (16,7,7)
            nn.ConvTranspose2d(
                in_channels=16, out_channels=8, kernel_size=3, stride=2, padding=1, output_padding=1
            ),    # (16,7,7) -> (8,14,14)
            nn.ConvTranspose2d(
                in_channels=8, out_channels=1, kernel_size=3, stride=2, padding=1, output_padding=1
            ),    # (8,14,14) -> (1,28,28)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if self._add_linear:
            x = self.linear_encoder(x)
            x = self.linear_decoder(x)
        x = self.decoder(x)
        return x