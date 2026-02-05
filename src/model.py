# model.py
import torch.nn as nn
import src.config as config

class DSCBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size= 3, padding= 1, groups= in_channels, bias= False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels, kernel_size= 1, bias= False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size= 2)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.maxpool(x)
        return x

class HomeAudioCNN(nn.Module):
    def __init__(self, num_classes, base_filters=config.BASE_FILTERS, dropout_rate=config.DROPOUT_RATE):
        super().__init__()
        
        f1, f2, f3 = base_filters, base_filters * 2, base_filters * 4
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, f1, kernel_size=3, stride=1, padding=1, bias= False),
            nn.BatchNorm2d(f1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv2 = DSCBlock(f1, f2)
        self.conv3 = DSCBlock(f2, f3)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((5, 5))
        self.flatten = nn.Flatten()
        
        self.classifier = nn.Sequential(
            nn.Linear(f3 * 5 * 5, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.adaptive_pool(x)
        x = self.flatten(x)
        return self.classifier(x)