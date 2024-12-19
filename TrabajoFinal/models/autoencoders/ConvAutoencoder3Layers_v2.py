import torch.nn as nn

# Sizes:
# (1, 28, 28)
# (4, 13, 13)
# (8, 13, 13)
# (16, 2, 2)
class ConvAutoencoder3Layers_v2(nn.Module):
    def __init__(self, add_linear, n, p):
        super().__init__()
        self._add_linear = add_linear
        self._dropout = p
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3),    # (1,28,28) -> (4,26,26)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2),    # (4,26,26) -> (4,13,13)

            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=1),   # (4,13,13) -> (8,13,13)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2),    # (8,13,13) -> (8,6,6)
            
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3),   # (8,6,6) -> (16,4,4)
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.MaxPool2d(kernel_size=2)    # (16,4,4) -> (16,2,2)
        )
        self.linear_encoder = nn.Sequential(
            nn.Flatten(),    # (16,2,2) -> 16*2*2
            nn.Linear(in_features=16*2*2, out_features=n),
            nn.ReLU(),
            nn.Dropout(p=p)
        )
        self.linear_decoder = nn.Sequential(
            nn.Linear(in_features=n, out_features=16*2*2),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Unflatten(dim=1, unflattened_size=(16, 2, 2)),
        )
        self.decoder = nn.Sequential(
            
            nn.ConvTranspose2d(
                in_channels=16, out_channels=8, kernel_size=5
            ),    # (16, 2, 2) -> (8, 6, 6)
            
            nn.ConvTranspose2d(
                in_channels=8, out_channels=4, kernel_size=3, stride=2
            ),    # (8, 6, 6) -> (4, 13, 13)
            
            nn.ConvTranspose2d(
                in_channels=4, out_channels=1, kernel_size=4, stride=2
            ),    # (4, 13, 13) -> (1, 28, 28)
            
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        if self._add_linear:
            x = self.linear_encoder(x)
            x = self.linear_decoder(x)
        x = self.decoder(x)
        return x