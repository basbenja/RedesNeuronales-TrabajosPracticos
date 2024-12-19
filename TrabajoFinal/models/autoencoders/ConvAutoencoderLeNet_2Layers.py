import torch.nn as nn

# Inspirado en LeNet-5 (libro Hands-On ML with Scikit-Learn, Keras, and TensorFlow)
class ConvAutoencoderLeNet_2Layers(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1),    # (1,32,32) -> (6,28,28)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),    # (6,28,28) -> (6,14,14)
            
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),    # (6,14,14) -> (16,10,10)
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),    # (16,10,10) -> (16,5,5)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=16, out_channels=6, kernel_size=5, stride=2, output_padding=1
            ),    # (16,5,5) -> (6,14,14)
            nn.ReLU(),
            
            nn.ConvTranspose2d(
                in_channels=6, out_channels=1, kernel_size=5, stride=2, output_padding=1
            ),
            
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x