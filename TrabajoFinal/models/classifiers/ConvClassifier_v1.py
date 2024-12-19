import torch.nn as nn

class ConvClassifier_v1(nn.Module):
    def __init__(self, p, n1, n2, encoder=None):
        super().__init__()
        self._dropout = p
        if not encoder:
            print(f"Creating new encoder!")
            self.encoder = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1),    # (1,28,28) -> (16,28,28)
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.MaxPool2d(kernel_size=2),    # (16,28,28) -> (16,14,14)

                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),   # (16,14,14) -> (32,14,14)
                nn.ReLU(),
                nn.Dropout(p=p),
                nn.MaxPool2d(kernel_size=2)    # (32,14,14) -> (32,7,7)
            )
        else:
            print(f"Using provided encoder!")
            self.encoder = encoder
            for param in self.encoder.parameters():
                param.requires_grad = False

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=32*7*7, out_features=n1),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(in_features=n1, out_features=n2),
            nn.ReLU(),
            nn.Dropout(p),
            nn.Linear(in_features=n2, out_features=10)
        )
    
    def forward(self, x):
        x = self.encoder(x)
        logits = self.classifier(x)
        return logits