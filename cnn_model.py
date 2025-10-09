import torch
from torch import nn
from torchsummary import summary

class LeNet5(nn.Module):
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

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = LeNet5().to(device)
    print(summary(model, (1, 28, 28)))
    
       
       
       