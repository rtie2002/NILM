import torch
from torch import nn
from torchsummary import summary

class LeNet5(nn.Module):
    """
    Original LeNet5 for image classification (28x28 -> 10 classes)
    """
    def __init__(self):
       super(LeNet5, self).__init__()
       self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5, stride=1, padding=2)
       self.sig = nn.Sigmoid()
       self.s1 = nn.AvgPool2d(kernel_size=2, stride=2)
       self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
       self.s2 = nn.AvgPool2d(kernel_size=2, stride=2)

       self.flattern = nn.Flatten()
       self.f3 = nn.Linear(in_features=5*5*16, out_features=120)
       self.f4 = nn.Linear(in_features=120, out_features=84)
       self.f5 = nn.Linear(in_features=84, out_features=10)
    
    def forward(self, x):
        x = self.sig(self.conv1(x))
        x = self.s1(x)
        x = self.sig(self.conv2(x))
        x = self.s2(x)
        x = self.flattern(x)
        x = self.f3(x)
        x = self.f4(x)
        x = self.f5(x)
        return x


class NILM_CNN(nn.Module):
    """
    CNN model specifically designed for NILM (Non-Intrusive Load Monitoring)
    
    Input: 1D time series of mains power (batch_size, window_size)
    Output: 1D time series of appliance power (batch_size, window_size)
    
    Architecture:
    - 1D Convolutional layers for time series feature extraction
    - Fully connected layers for regression
    - Output layer predicts appliance power for each time step
    """
    def __init__(self, window_size=99, num_appliances=1):
        super(NILM_CNN, self).__init__()
        
        self.window_size = window_size
        self.num_appliances = num_appliances
        
        # 1D Convolutional layers for time series feature extraction
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        
        # Activation and pooling
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Global average pooling to reduce dimensions
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Fully connected layers for regression
        self.fc1 = nn.Linear(128, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, window_size * num_appliances)
        
    def forward(self, x):
        # Input shape: (batch_size, window_size)
        # Reshape to (batch_size, 1, window_size) for 1D conv
        x = x.unsqueeze(1)
        
        # 1D Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Global average pooling
        x = self.global_avg_pool(x)  # (batch_size, 128, 1)
        x = x.squeeze(-1)  # (batch_size, 128)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output layer - predict power for each time step
        x = self.fc3(x)  # (batch_size, window_size * num_appliances)
        
        # Reshape to (batch_size, window_size) for single appliance
        if self.num_appliances == 1:
            x = x.view(-1, self.window_size)
        else:
            x = x.view(-1, self.window_size, self.num_appliances)
        
        return x


class NILM_CNN_Simple(nn.Module):
    """
    Simplified CNN model for NILM - easier to train and debug
    """
    def __init__(self, window_size=99):
        super(NILM_CNN_Simple, self).__init__()
        
        self.window_size = window_size
        
        # Simple 1D CNN architecture
        self.conv1 = nn.Conv1d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
        # Calculate the size after convolutions
        # After 3 conv layers with padding, size remains window_size
        conv_output_size = 64 * window_size
        
        # Fully connected layers
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, window_size)
        
    def forward(self, x):
        # Input: (batch_size, window_size)
        # Reshape for 1D conv: (batch_size, 1, window_size)
        x = x.unsqueeze(1)
        
        # Convolutional layers
        x = self.relu(self.conv1(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv2(x))
        x = self.dropout(x)
        
        x = self.relu(self.conv3(x))
        x = self.dropout(x)
        
        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)  # (batch_size, 64 * window_size)
        
        # Fully connected layers
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        
        # Output: (batch_size, window_size)
        x = self.fc3(x)
        
        return x

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Test original LeNet5
    print("\n=== Original LeNet5 (for images) ===")
    model = LeNet5().to(device)
    print(summary(model, (1, 28, 28)))
    
    # Test NILM CNN models
    print("\n=== NILM CNN (for time series) ===")
    window_size = 99
    nilm_model = NILM_CNN(window_size=window_size).to(device)
    
    # Create dummy input for testing
    dummy_input = torch.randn(2, window_size).to(device)  # batch_size=2, window_size=99
    output = nilm_model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    
    print("\n=== NILM CNN Simple (for time series) ===")
    nilm_simple = NILM_CNN_Simple(window_size=window_size).to(device)
    output_simple = nilm_simple(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {output_simple.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in nilm_model.parameters())
    trainable_params = sum(p.numel() for p in nilm_model.parameters() if p.requires_grad)
    print(f"\nNILM CNN Parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    total_params_simple = sum(p.numel() for p in nilm_simple.parameters())
    trainable_params_simple = sum(p.numel() for p in nilm_simple.parameters() if p.requires_grad)
    print(f"NILM CNN Simple Parameters: {total_params_simple:,} total, {trainable_params_simple:,} trainable")
    
       
       
       