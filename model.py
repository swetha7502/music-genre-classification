import torch
import torch.nn as nn

class GenreCNN(nn.Module):
    def __init__(self, num_classes=10, input_size=None):
        super(GenreCNN, self).__init__()
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Pool to a manageable fixed size regardless of time dimension
            nn.AdaptiveAvgPool2d((8, 8)),
        )
        
        # Use provided input_size or default
        # If input_size provided, honor it (e.g., for backward compatibility),
        # otherwise assume adaptive pooled to 8x8.
        default_fc = 128 * 8 * 8
        fc_input = input_size if input_size else default_fc
        
        self.fc_layers = nn.Sequential(
            nn.Linear(fc_input, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layers(x)
        return x
