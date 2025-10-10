"""
Custom PyTorch Dataset for NILM (Non-Intrusive Load Monitoring) Data

This module provides a PyTorch Dataset class that integrates with the preprocessing.py
to create trainable datasets for NILM models.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import os
from preprocessing import NILMPreprocessor


class NILMDataset(Dataset):
    """
    Custom PyTorch Dataset for NILM data
    
    This dataset loads preprocessed NILM data and provides it in the format
    expected by PyTorch models for training.
    """
    
    def __init__(self, mains_data, appliance_data, appliance_name):
        """
        Initialize the NILM Dataset
        
        Args:
            mains_data (np.array): Mains power data (X) - shape (n_samples, window_size)
            appliance_data (np.array): Appliance power data (y) - shape (n_samples, window_size)
            appliance_name (str): Name of the target appliance
        """
        self.mains_data = torch.FloatTensor(mains_data)
        self.appliance_data = torch.FloatTensor(appliance_data)
        self.appliance_name = appliance_name
        
        # Ensure data shapes match
        assert self.mains_data.shape == self.appliance_data.shape, \
            f"Shape mismatch: mains {self.mains_data.shape} vs appliance {self.appliance_data.shape}"
        
        print(f"✓ NILM Dataset created for {appliance_name}")
        print(f"  - Samples: {len(self.mains_data):,}")
        print(f"  - Window size: {self.mains_data.shape[1]}")
        print(f"  - Mains range: [{self.mains_data.min():.3f}, {self.mains_data.max():.3f}]")
        print(f"  - Appliance range: [{self.appliance_data.min():.3f}, {self.appliance_data.max():.3f}]")
    
    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.mains_data)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset
        
        Args:
            idx (int): Index of the sample
            
        Returns:
            tuple: (mains_power, appliance_power)
                - mains_power: Input time series (window_size,)
                - appliance_power: Target time series (window_size,)
        """
        return self.mains_data[idx], self.appliance_data[idx]


class NILMDataManager:
    """
    Manager class to handle NILM data loading and dataset creation
    """
    
    def __init__(self, dataset_path, building_id=1, sample_period=6):
        """
        Initialize the data manager
        
        Args:
            dataset_path (str): Path to the HDF5 dataset file
            building_id (int): Building ID to use
            sample_period (int): Sampling period in seconds
        """
        self.dataset_path = dataset_path
        self.building_id = building_id
        self.sample_period = sample_period
        self.preprocessor = None
        self.processed_data = None
        
    def load_and_preprocess_data(self, target_appliances, train_months=6, test_months=1, 
                                window_size=99, stride=1, force_reprocess=False):
        """
        Load and preprocess NILM data
        
        Args:
            target_appliances (list): List of target appliance names
            train_months (int): Number of months for training
            test_months (int): Number of months for testing
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            force_reprocess (bool): Force reprocessing even if cached data exists
            
        Returns:
            dict: Processed data dictionary
        """
        cache_file = f"preprocessed_data_{target_appliances[0]}_{window_size}.pkl"
        
        # Check if cached data exists and load it
        if os.path.exists(cache_file) and not force_reprocess:
            print(f"Loading cached preprocessed data from {cache_file}...")
            with open(cache_file, 'rb') as f:
                self.processed_data = pickle.load(f)
            print("✓ Cached data loaded successfully!")
            return self.processed_data
        
        # Initialize preprocessor
        print("Initializing NILM preprocessor...")
        self.preprocessor = NILMPreprocessor(
            self.dataset_path, 
            self.building_id, 
            self.sample_period
        )
        
        # Check data range
        self.preprocessor.check_data_range()
        
        # Set time windows
        train_start, train_end, test_start, test_end = self.preprocessor.set_time_windows(
            train_months, test_months
        )
        
        # Load training data
        self.preprocessor.load_training_data(train_start, train_end, target_appliances)
        
        # Load testing data
        self.preprocessor.load_testing_data(test_start, test_end, target_appliances)
        
        # Create windows and normalize
        self.processed_data = self.preprocessor.create_windows_and_normalize(
            window_size, stride
        )
        
        # Cache the processed data
        print(f"Saving processed data to {cache_file}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(self.processed_data, f)
        print("✓ Data cached successfully!")
        
        return self.processed_data
    
    def create_datasets(self, target_appliance):
        """
        Create PyTorch datasets for training, validation, and testing
        
        Args:
            target_appliance (str): Name of the target appliance
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        if self.processed_data is None:
            raise ValueError("No processed data available. Call load_and_preprocess_data() first.")
        
        # Extract data for the target appliance
        X_train = self.processed_data['X_train']
        X_val = self.processed_data['X_val']
        X_test = self.processed_data['X_test']
        
        y_train = self.processed_data['y_train'][target_appliance]
        y_val = self.processed_data['y_val'][target_appliance]
        y_test = self.processed_data['y_test'][target_appliance]
        
        # Create datasets
        train_dataset = NILMDataset(X_train, y_train, target_appliance)
        val_dataset = NILMDataset(X_val, y_val, target_appliance)
        test_dataset = NILMDataset(X_test, y_test, target_appliance)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, target_appliance, batch_size=32, num_workers=4):
        """
        Create PyTorch DataLoaders for training, validation, and testing
        
        Args:
            target_appliance (str): Name of the target appliance
            batch_size (int): Batch size for training
            num_workers (int): Number of worker processes for data loading
            
        Returns:
            tuple: (train_loader, val_loader, test_loader)
        """
        train_dataset, val_dataset, test_dataset = self.create_datasets(target_appliance)
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            pin_memory=True
        )
        
        print(f"✓ DataLoaders created for {target_appliance}")
        print(f"  - Train batches: {len(train_loader)}")
        print(f"  - Val batches: {len(val_loader)}")
        print(f"  - Test batches: {len(test_loader)}")
        
        return train_loader, val_loader, test_loader
    
    def get_data_stats(self):
        """
        Get statistics about the processed data
        
        Returns:
            dict: Data statistics
        """
        if self.processed_data is None:
            return None
        
        return self.processed_data['stats']


def test_nilm_dataset():
    """
    Test function to verify the NILM dataset works correctly
    """
    print("Testing NILM Dataset...")
    
    # Create dummy data for testing
    n_samples = 1000
    window_size = 99
    
    # Create dummy mains and appliance data
    mains_data = np.random.randn(n_samples, window_size) * 100 + 1000  # ~1000W average
    appliance_data = np.random.randn(n_samples, window_size) * 50 + 200  # ~200W average
    
    # Create dataset
    dataset = NILMDataset(mains_data, appliance_data, "test_appliance")
    
    # Test dataset functionality
    print(f"Dataset length: {len(dataset)}")
    
    # Test getting a sample
    mains_sample, appliance_sample = dataset[0]
    print(f"Sample shapes: mains {mains_sample.shape}, appliance {appliance_sample.shape}")
    print(f"Sample types: mains {type(mains_sample)}, appliance {type(appliance_sample)}")
    
    # Test with DataLoader
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    for batch_idx, (mains_batch, appliance_batch) in enumerate(dataloader):
        print(f"Batch {batch_idx}: mains {mains_batch.shape}, appliance {appliance_batch.shape}")
        if batch_idx >= 2:  # Test first 3 batches
            break
    
    print("✓ NILM Dataset test completed successfully!")


if __name__ == "__main__":
    test_nilm_dataset()
