import torch
from torch import nn

class NILMCNN(nn.Module):
    def __init__(self, window_size=99):
        super(NILMCNN, self).__init__()
        # 1D CNN for time series
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        # Calculate flattened size
        conv_output_size = window_size // 4  # After 2 pooling layers
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * conv_output_size, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(128, window_size)  # Output same size as input window
    
    def forward(self, x):
        # x shape: (batch_size, 1, window_size)
        x = self.relu1(self.conv1(x))
        x = self.pool1(x)
        x = self.relu2(self.conv2(x))
        x = self.pool2(x)
        x = self.flatten(x)
        x = self.relu3(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
