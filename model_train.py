from torchvision.datasets import FashionMNIST
from torchvision import transforms
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import LeNet5

def train_val_data_process():
    train_data = FashionMNIST(root='./data', 
                                train=True, 
                                transform=transforms.ToTensor(), 
                                download=True)
    
    train_data, val_data = Data.random_split(train_data, lengths=[round(len(train_data)*0.8), round(len(train_data)*0.2)])

    train_dataloader = Data.DataLoader(dataset=train_data, 
                                       batch_size=128, 
                                       shuffle=True,
                                       num_workers=8)
    
    train_dataloader = Data.DataLoader(dataset=train_data, 
                                       batch_size=128, 
                                       shuffle=True,
                                       num_workers=8)

                                       
