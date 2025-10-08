import torch
import torch.nn as nn
import torch.nn.functional as F

class NILM_CNN(nn.Module):
    def __init__(self, input_length=600, n_appliances=1):
        super(NILM_CNN, self).__init__()
        
        # Input shape: (batch_size, 1, input_length)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv1d(16, 32, 5, padding=2)
        self.conv3 = nn.Conv1d(32, 64, 5, padding=2)
        
        # Calculate the size of the flattened features
        # After 3 max pooling layers with stride=2, the length becomes input_length / (2^3)
        self.flattened_size = 64 * (input_length // 8)
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, n_appliances)  # One output per appliance
        
        # Additional layers for better training
        self.dropout = nn.Dropout(0.5)
        self.bn1 = nn.BatchNorm1d(16)
        self.bn2 = nn.BatchNorm1d(32)
        self.bn3 = nn.BatchNorm1d(64)
        
    def forward(self, x):
        # Input shape: (batch_size, 1, input_length)
        x = x.unsqueeze(1)  # Add channel dimension if not present
        
        # Convolutional layers with ReLU and max pooling
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.max_pool1d(x, 2)
        
        # Flatten the output for the fully connected layers
        x = x.view(-1, self.flattened_size)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Example usage
if __name__ == "__main__":
    # Test the model
    model = NILM_CNN(input_length=600, n_appliances=5)
    print(model)
    
    # Create a random input tensor (batch_size=32, sequence_length=600)
    x = torch.randn(32, 1, 600)
    output = model(x)
    print("Output shape:", output.shape)  # Should be [32, 5] for 5 appliances
