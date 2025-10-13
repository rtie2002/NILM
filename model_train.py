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
from preprocessing import NILMDataLoader, TARGET_APPLIANCES
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score, recall_score, precision_score, matthews_corrcoef

# Standard NILM metrics (from toolkit)
on_threshold = {'washer dryer': 20, 'fridge': 50, 'kettle': 2000, 'dish washer': 20, 'washing machine': 20, 'drill': 10, 'light': 5}

def mae(app_name, app_gt, app_pred):
    """Mean Absolute Error"""
    return mean_absolute_error(app_gt, app_pred)

def rmse(app_name, app_gt, app_pred):
    """Root Mean Square Error"""
    return mean_squared_error(app_gt, app_pred)**0.5

def rmae(app_name, app_gt, app_pred):
    """Relative Mean Absolute Error"""
    numerator = np.sum(np.abs(app_gt - app_pred))
    denominator = np.sum(np.abs(app_gt))
    return numerator / denominator if denominator > 0 else 0

def nep(app_name, app_gt, app_pred):
    """Normalized Error in Total Power"""
    numerator = np.sum(np.abs(app_gt - app_pred))
    denominator = np.sum(np.abs(app_gt))
    return numerator/denominator if denominator > 0 else 0

def omae(app_name, app_gt, app_pred):
    """On-state Mean Absolute Error (only when appliance is ON)"""
    threshold = on_threshold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    idx = gt_temp > threshold
    if np.sum(idx) == 0:
        return 0
    gt_temp = gt_temp[idx]
    pred_temp = np.array(app_pred)[idx]
    return mae(app_name, gt_temp, pred_temp)

def f1score(app_name, app_gt, app_pred):
    """F1 Score for ON/OFF classification"""
    threshold = on_threshold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)
    return f1_score(gt_temp, pred_temp)

def recall(app_name, app_gt, app_pred):
    """Recall for ON/OFF classification"""
    threshold = on_threshold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)
    return recall_score(gt_temp, pred_temp)

def precision(app_name, app_gt, app_pred):
    """Precision for ON/OFF classification"""
    threshold = on_threshold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)
    return precision_score(gt_temp, pred_temp)

def mcc(app_name, app_gt, app_pred):
    """Matthews Correlation Coefficient"""
    threshold = on_threshold.get(app_name, 10)
    gt_temp = np.array(app_gt)
    gt_temp = np.where(gt_temp < threshold, 0, 1)
    pred_temp = np.array(app_pred)
    pred_temp = np.where(pred_temp < threshold, 0, 1)
    return matthews_corrcoef(gt_temp, pred_temp)

def nde(app_name, app_gt, app_pred):
    """Normalized Disaggregation Error (NILMTK Standard)"""
    numerator = np.sum((app_gt - app_pred) ** 2)
    denominator = np.sum(app_gt ** 2)
    return numerator / denominator if denominator > 0 else 0

def energy_accuracy(app_name, app_gt, app_pred):
    """Energy Accuracy (NILMTK Standard)"""
    numerator = np.sum(app_gt * app_pred)
    denominator = np.sum(app_gt ** 2)
    return numerator / denominator if denominator > 0 else 0

def disaggregation_accuracy(app_name, app_gt, app_pred):
    """Disaggregation Accuracy (NILMTK Standard)"""
    total_error = np.sum(np.abs(app_gt - app_pred))
    total_energy = np.sum(np.abs(app_gt))
    return 1 - (total_error / total_energy) if total_energy > 0 else 0

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
    """
    Ultra-simplified one-line data loading function
    Following toolkit standards for maximum simplicity
    """
    data_loader = NILMDataLoader()
    return data_loader.get_ready_dataloaders(batch_size=512)


def train_model_process(model, train_dataloader, val_dataloader, epochs, target_appliance):
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

        # Reset accumulators for this epoch
        train_loss = 0.0
        val_loss = 0.0
        train_num = 0
        val_num = 0

        # Training phase
        model.train()
        train_mae = 0.0
        train_rmse = 0.0
        
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
            
            # Calculate NILM metrics for this batch (detach from computation graph)
            with torch.no_grad():
                batch_mae = mae(target_appliance, b_y.detach().cpu().numpy().flatten(), output.detach().cpu().numpy().flatten())
                batch_rmse = rmse(target_appliance, b_y.detach().cpu().numpy().flatten(), output.detach().cpu().numpy().flatten())
                train_mae += batch_mae * b_x.size(0)
                train_rmse += batch_rmse * b_x.size(0)
            
            # Show progress every 1000 batches with NILM metrics
            if step % 1000 == 0 and step > 0:  # Skip batch 0 to avoid misleading spike
                current_loss = train_loss / train_num
                current_mae = train_mae / train_num
                current_rmse = train_rmse / train_num
                print(f"  Batch {step}/{len(train_dataloader)} - Loss: {current_loss:.4f}, MAE: {current_mae:.4f}, RMSE: {current_rmse:.4f}")
        
        # Validation phase
        model.eval()
        val_mae = 0.0
        val_rmse = 0.0
        val_f1 = 0.0
        val_omae = 0.0
        val_nde = 0.0
        val_energy_acc = 0.0
        val_disagg_acc = 0.0
        
        with torch.no_grad():
            for step, (b_x, b_y) in enumerate(val_dataloader):
                b_x = b_x.to(device)
                b_y = b_y.to(device)

                output = model(b_x)
                loss = criterion(output, b_y)

                val_loss += loss.item() * b_x.size(0)
                val_num += b_x.size(0)
                
                # Calculate NILM metrics for this batch
                batch_mae = mae(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_rmse = rmse(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_f1 = f1score(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_omae = omae(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_nde = nde(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_energy_acc = energy_accuracy(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                batch_disagg_acc = disaggregation_accuracy(target_appliance, b_y.cpu().numpy().flatten(), output.cpu().numpy().flatten())
                
                val_mae += batch_mae * b_x.size(0)
                val_rmse += batch_rmse * b_x.size(0)
                val_f1 += batch_f1 * b_x.size(0)
                val_omae += batch_omae * b_x.size(0)
                val_nde += batch_nde * b_x.size(0)
                val_energy_acc += batch_energy_acc * b_x.size(0)
                val_disagg_acc += batch_disagg_acc * b_x.size(0)
                
                # Show validation progress every 500 batches with NILM metrics
                if step % 500 == 0 and step > 0:  # Skip batch 0 to avoid misleading spike
                    current_val_loss = val_loss / val_num
                    current_val_mae = val_mae / val_num
                    current_val_rmse = val_rmse / val_num
                    current_val_f1 = val_f1 / val_num
                    current_val_omae = val_omae / val_num
                    print(f"  Val Batch {step}/{len(val_dataloader)} - Loss: {current_val_loss:.4f}, MAE: {current_val_mae:.4f}, RMSE: {current_val_rmse:.4f}, F1: {current_val_f1:.4f}, OMAE: {current_val_omae:.4f}")
        
        # Calculate average losses and metrics
        avg_train_loss = train_loss / train_num
        avg_val_loss = val_loss / val_num
        avg_train_mae = train_mae / train_num
        avg_train_rmse = train_rmse / train_num
        avg_val_mae = val_mae / val_num
        avg_val_rmse = val_rmse / val_num
        avg_val_f1 = val_f1 / val_num
        avg_val_omae = val_omae / val_num
        avg_val_nde = val_nde / val_num
        avg_val_energy_acc = val_energy_acc / val_num
        avg_val_disagg_acc = val_disagg_acc / val_num
        
        train_losses_all.append(avg_train_loss)
        val_losses_all.append(avg_val_loss)
        
        # Display comprehensive NILM metrics
        print(f"Train - Loss: {avg_train_loss:.4f}, MAE: {avg_train_mae:.4f}, RMSE: {avg_train_rmse:.4f}")
        print(f"Val   - Loss: {avg_val_loss:.4f}, MAE: {avg_val_mae:.4f}, RMSE: {avg_val_rmse:.4f}, F1: {avg_val_f1:.4f}, OMAE: {avg_val_omae:.4f}")
        print(f"Val   - NDE: {avg_val_nde:.4f}, Energy Acc: {avg_val_energy_acc:.4f}, Disagg Acc: {avg_val_disagg_acc:.4f}")
        
        # Save best model FIRST
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'model_train.pth')
            print("🎉 New best model saved!")
        
        # Calculate time per epoch
        epoch_time = time.time() - since
        print(f"Epoch {epoch+1} completed in {epoch_time:.1f}s")
        
        # Only print best loss if it's been updated (not infinity)
        if best_loss != float('inf'):
            print(f"Best Val Loss so far: {best_loss:.4f}")
        else:
            print("Best Val Loss so far: Not set yet")
        print("=" * 50)
        
        since = time.time()  # Reset timer for next epoch

    # Load best model
    print("Loading best model weights...")
    model.load_state_dict(best_model_wts)
    print("Best model loaded successfully!")
    
    print("Creating training history DataFrame...")
    train_process = pd.DataFrame(data = {"epoch": range(epochs), 
                                         "train_loss": train_losses_all, 
                                         "val_loss": val_losses_all})
    print("DataFrame created successfully!")

    return train_process

def matplot_loss(train_process):
    print("Creating loss plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(train_process["epoch"], train_process["train_loss"], 'ro-', label="Train Loss")
    plt.plot(train_process["epoch"], train_process["val_loss"], 'bo-', label="Val Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training and Validation Loss")
    plt.grid(True)
    print("Plot created, showing...")
    plt.show()
    print("Plot displayed successfully!")


if __name__ == "__main__":
    # Create NILM CNN model
    model = NILMCNN(window_size=99)
    
    # Load NILM data
    train_dataloader, val_dataloader, target_appliance = load_nilm_data()
    
    # Train model
    print("Starting training...")
    train_process = train_model_process(model, train_dataloader, val_dataloader, epochs=40, target_appliance=target_appliance)
    print("Training completed!")
    
    # Plot results
    print("Starting to plot results...")
    matplot_loss(train_process)
    print("All done!")

