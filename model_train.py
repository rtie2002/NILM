import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as Data
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import pandas as pd
from cnn_model import NILMCNN
from preprocessing import NILMPreprocessor

class NILMDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        # Reshape for 1D CNN: (batch_size, 1, window_size)
        x = self.X[idx].unsqueeze(0)  # Add channel dimension
        y = self.y[idx]
        return x, y

def load_nilm_data():
    # Load preprocessed data
    DATASET_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5"
    TARGET_APPLIANCES = ['washer dryer']
    
    preprocessor = NILMPreprocessor(DATASET_PATH)
    train_start, train_end, test_start, test_end = preprocessor.set_time_windows()
    
    # Load data
    preprocessor.load_training_data(train_start, train_end, TARGET_APPLIANCES)
    preprocessor.load_testing_data(test_start, test_end, TARGET_APPLIANCES)
    
    # Create windows and normalize
    processed_data = preprocessor.create_windows_and_normalize(window_size=99, stride=1)
    
    # Create datasets
    train_dataset = NILMDataset(processed_data['X_train'], processed_data['y_train']['washer dryer'])
    val_dataset = NILMDataset(processed_data['X_val'], processed_data['y_val']['washer dryer'])
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    
    return train_dataloader, val_dataloader


def train_model_process(model, train_dataloader, val_dataloader, epochs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()  # Use MSE Loss for regression

    # Store the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = float('inf')

    # Training and validation losses
    train_losses_all = []
    val_losses_all = []

    since = time.time()
    

    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 10)

        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        # Training phase
        model.train()
        for step, (b_x, b_y) in enumerate(train_dataloader):
            b_x = b_x.to(device)
            b_y = b_y.to(device)

            optimizer.zero_grad()
            output = model(b_x)
            loss = criterion(output, b_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * b_x.size(0)
            train_num += b_x.size(0)
            
            # Show progress every 1000 batches
            if step % 1000 == 0:
                current_loss = train_loss / train_num
                print(f"  Batch {step}/{len(train_dataloader)} - Current Loss: {current_loss:.4f}")
        
        # Validation phase
        model.eval()
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)
                
                # Show validation progress every 500 batches
                if step % 500 == 0:
                    current_val_loss = val_loss / val_num
                    print(f"  Val Batch {step}/{len(val_dataloader)} - Current Val Loss: {current_val_loss:.4f}")
        
        # Calculate average losses
        avg_train_loss = train_loss / train_num
        avg_val_loss = val_loss / val_num
        
        train_losses_all.append(avg_train_loss)
        val_losses_all.append(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Val Loss: {avg_val_loss:.4f}")
        
        # Calculate time per epoch
        epoch_time = time.time() - since
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        print(f"Best Val Loss so far: {best_loss:.4f}")
        print("=" * 50)

        # Save best model
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'model_train.pth')
            print("🎉 New best model saved!")
        
        since = time.time()  # Reset timer for next epoch

    # Load best model
    model.load_state_dict(best_model_wts)
    
    train_process = pd.DataFrame(data = {"epoch": range(epochs), 
                                         "train_loss": train_losses_all, 
                                         "val_loss": val_losses_all})
    
    return train_process

def matplot_loss(train_process):
    plt.figure(figsize=(10, 6))
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], 'bo-', label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Create NILM CNN model
    model = NILMCNN(window_size=99)
    
    # Load NILM data
    train_dataloader, val_dataloader = load_nilm_data()
    
    # Train model
    train_process = train_model_process(model, train_dataloader, val_dataloader, epochs=10)
    
    # Plot results
    matplot_loss(train_process)




    





  

