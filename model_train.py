import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import os
import pandas as pd
from cnn_model import NILM_CNN, NILM_CNN_Simple
from nilm_dataset import NILMDataManager

def load_nilm_data(dataset_path, target_appliance, window_size=99, batch_size=32):
    """
    Load and preprocess NILM data for training
    
    Args:
        dataset_path (str): Path to the HDF5 dataset file
        target_appliance (str): Name of the target appliance
        window_size (int): Size of sliding window
        batch_size (int): Batch size for training
        
    Returns:
        tuple: (train_loader, val_loader, test_loader, data_stats)
    """
    print("="*60)
    print("LOADING NILM DATA")
    print("="*60)
    
    # Initialize data manager
    data_manager = NILMDataManager(dataset_path)
    
    # Load and preprocess data
    processed_data = data_manager.load_and_preprocess_data(
        target_appliances=[target_appliance],
        train_months=6,  # 6 months of training data
        test_months=1,   # 1 month of test data
        window_size=window_size,
        stride=1,
        force_reprocess=False  # Use cached data if available
    )
    
    # Create data loaders
    train_loader, val_loader, test_loader = data_manager.create_dataloaders(
        target_appliance=target_appliance,
        batch_size=batch_size,
        num_workers=4
    )
    
    # Get data statistics
    data_stats = data_manager.get_data_stats()
    
    return train_loader, val_loader, test_loader, data_stats


def train_nilm_model(model, train_loader, val_loader, epochs, device, model_save_path):
    """
    Train NILM model with proper regression loss and metrics
    
    Args:
        model: NILM CNN model
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of training epochs
        device: Device to train on (cuda/cpu)
        model_save_path: Path to save the best model
        
    Returns:
        dict: Training history with losses and metrics
    """
    print("="*60)
    print("TRAINING NILM MODEL")
    print("="*60)
    
    # Move model to device
    model = model.to(device)
    
    # Use MSE loss for regression (not CrossEntropy for classification)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # Store the best model weights
    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')
    
    # Training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    
    since = time.time()
    
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}/{epochs}")
        print("-" * 30)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_mae = 0.0
        train_samples = 0
        
        for batch_idx, (mains_batch, appliance_batch) in enumerate(train_loader):
            # Move data to device
            mains_batch = mains_batch.to(device)
            appliance_batch = appliance_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(mains_batch)
            loss = criterion(outputs, appliance_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Calculate metrics
            train_loss += loss.item() * mains_batch.size(0)
            mae = torch.mean(torch.abs(outputs - appliance_batch)).item()
            train_mae += mae * mains_batch.size(0)
            train_samples += mains_batch.size(0)
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}: Loss={loss.item():.6f}, MAE={mae:.6f}")
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_samples = 0
        
        with torch.no_grad():
            for mains_batch, appliance_batch in val_loader:
                mains_batch = mains_batch.to(device)
                appliance_batch = appliance_batch.to(device)
                
                outputs = model(mains_batch)
                loss = criterion(outputs, appliance_batch)
                
                val_loss += loss.item() * mains_batch.size(0)
                mae = torch.mean(torch.abs(outputs - appliance_batch)).item()
                val_mae += mae * mains_batch.size(0)
                val_samples += mains_batch.size(0)
        
        # Calculate average metrics
        avg_train_loss = train_loss / train_samples
        avg_val_loss = val_loss / val_samples
        avg_train_mae = train_mae / train_samples
        avg_val_mae = val_mae / val_samples
        
        # Store metrics
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_maes.append(avg_train_mae)
        val_maes.append(avg_val_mae)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Train Loss: {avg_train_loss:.6f}, Train MAE: {avg_train_mae:.6f}")
        print(f"Val Loss: {avg_val_loss:.6f}, Val MAE: {avg_val_mae:.6f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), model_save_path)
            print(f"✓ New best model saved! Val Loss: {best_val_loss:.6f}")
        
        print()
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Training time
    time_elapsed = time.time() - since
    print(f"Training completed in {time_elapsed//60:.0f}m {time_elapsed%60:.0f}s")
    print(f"Best validation loss: {best_val_loss:.6f}")
    
    # Create training history
    training_history = {
        'epoch': list(range(1, epochs + 1)),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_mae': train_maes,
        'val_mae': val_maes
    }
    
    return training_history

def plot_training_history(training_history):
    """
    Plot training history for NILM model
    
    Args:
        training_history (dict): Training history with losses and metrics
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    axes[0].plot(training_history['epoch'], training_history['train_loss'], 'r-', label='Train Loss', linewidth=2)
    axes[0].plot(training_history['epoch'], training_history['val_loss'], 'b-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('MSE Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot MAE
    axes[1].plot(training_history['epoch'], training_history['train_mae'], 'r-', label='Train MAE', linewidth=2)
    axes[1].plot(training_history['epoch'], training_history['val_mae'], 'b-', label='Val MAE', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Mean Absolute Error')
    axes[1].set_title('Training and Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def evaluate_model(model, test_loader, device, data_stats):
    """
    Evaluate the trained model on test data
    
    Args:
        model: Trained NILM model
        test_loader: Test data loader
        device: Device to run evaluation on
        data_stats: Data statistics for denormalization
        
    Returns:
        dict: Evaluation metrics
    """
    print("="*60)
    print("EVALUATING MODEL")
    print("="*60)
    
    model.eval()
    test_loss = 0.0
    test_mae = 0.0
    test_samples = 0
    
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for mains_batch, appliance_batch in test_loader:
            mains_batch = mains_batch.to(device)
            appliance_batch = appliance_batch.to(device)
            
            outputs = model(mains_batch)
            loss = nn.MSELoss()(outputs, appliance_batch)
            
            test_loss += loss.item() * mains_batch.size(0)
            mae = torch.mean(torch.abs(outputs - appliance_batch)).item()
            test_mae += mae * mains_batch.size(0)
            test_samples += mains_batch.size(0)
            
            # Store predictions and targets for further analysis
            all_predictions.append(outputs.cpu().numpy())
            all_targets.append(appliance_batch.cpu().numpy())
    
    # Calculate final metrics
    avg_test_loss = test_loss / test_samples
    avg_test_mae = test_mae / test_samples
    
    print(f"Test Loss (MSE): {avg_test_loss:.6f}")
    print(f"Test MAE: {avg_test_mae:.6f}")
    
    # Calculate additional metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # RMSE
    rmse = np.sqrt(np.mean((all_predictions - all_targets) ** 2))
    print(f"Test RMSE: {rmse:.6f}")
    
    # R² Score
    ss_res = np.sum((all_targets - all_predictions) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2_score = 1 - (ss_res / ss_tot)
    print(f"Test R² Score: {r2_score:.6f}")
    
    evaluation_metrics = {
        'mse': avg_test_loss,
        'mae': avg_test_mae,
        'rmse': rmse,
        'r2_score': r2_score
    }
    
    return evaluation_metrics


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5"
    TARGET_APPLIANCE = "washer dryer"  # Change this to your target appliance
    WINDOW_SIZE = 99
    BATCH_SIZE = 32
    EPOCHS = 20
    MODEL_SAVE_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\best_nilm_model.pth"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load NILM data
    train_loader, val_loader, test_loader, data_stats = load_nilm_data(
        dataset_path=DATASET_PATH,
        target_appliance=TARGET_APPLIANCE,
        window_size=WINDOW_SIZE,
        batch_size=BATCH_SIZE
    )
    
    # Create NILM model
    print(f"\nCreating NILM CNN model for {TARGET_APPLIANCE}...")
    model = NILM_CNN_Simple(window_size=WINDOW_SIZE)
    
    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    
    # Train the model
    training_history = train_nilm_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=EPOCHS,
        device=device,
        model_save_path=MODEL_SAVE_PATH
    )
    
    # Plot training history
    plot_training_history(training_history)
    
    # Evaluate the model
    evaluation_metrics = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        data_stats=data_stats
    )
    
    # Save training history
    history_df = pd.DataFrame(training_history)
    history_df.to_csv("training_history.csv", index=False)
    print(f"\nTraining history saved to training_history.csv")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("="*60)
    print(f"Best model saved to: {MODEL_SAVE_PATH}")
    print(f"Final test metrics:")
    for metric, value in evaluation_metrics.items():
        print(f"  {metric.upper()}: {value:.6f}")




    





  

