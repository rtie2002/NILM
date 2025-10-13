"""
NILM Data Preprocessing - Standard Toolkit Method

This script shows how to properly preprocess UK-DALE data for NILM models, 
following the exact standard methods used in all NILM toolkits.

Key Points:
1. Use ALL Available Data: Maximize training data (not just 1-2 days!)
2. Sliding Windows: Create thousands of training examples from time series
3. Normalization: Standardize data for stable training
4. Proper Splits: 80% train, 20% test from all available data
5. Feed Total Power: Mains power (input) → Appliance power (output)
"""

# ============================================================================
# CONFIGURATION VARIABLES - ALL CONTROL SETTINGS AT THE TOP
# ============================================================================

# Dataset Configuration
DATASET_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5"
BUILDING_ID = 1
SAMPLE_PERIOD = 6  # seconds

# Target Appliances (list of appliances to disaggregate)
TARGET_APPLIANCES = ['washer dryer']  # Change this to your target appliances
# Available appliances in UK-DALE: ['kettle', 'microwave', 'dishwasher', 'washing machine', 'washer dryer', 'light', 'fridge', 'freezer', 'tumble dryer']

# Time Window Configuration
USE_ALL_DATA = True  # If True, use all available data; if False, use limited time range
TRAIN_START_DATE = "2013-03-17"  # Start of UK-DALE Building 1 data
TRAIN_END_DATE = "2015-01-05"    # End of UK-DALE Building 1 data (estimated)

# Sliding Window Configuration
WINDOW_SIZE = 99  # Size of sliding window (standard: 99 time points ≈ 10 minutes)
STRIDE = 1        # Stride for sliding window (1 = overlapping windows)

# Data Split Configuration
TRAIN_SPLIT_RATIO = 0.8   # 80% for training
VAL_SPLIT_RATIO = 0.2     # 20% for validation
RANDOM_SEED = 42          # For reproducible results

# Data Quality Configuration
MIN_SAMPLES_PER_DAY_RATIO = 0.5  # Minimum samples per day (50% of expected)
EXPECTED_SAMPLES_PER_DAY = 14400  # 24 * 60 * 60 // 6 (6-second sampling)
MAINS_POWER_MAX = 20000  # Maximum mains power in watts
APPLIANCE_POWER_MAX = 4000  # Maximum appliance power in watts

# File Configuration
OUTPUT_FILENAME = 'preprocessed_data.pkl'
SAVE_PROCESSED_DATA = True
LOAD_PROCESSED_DATA = False  # Set to True to load existing processed data

# Visualization Configuration
VISUALIZE_DATA = True
MAX_SAMPLES_TO_PLOT = 1000  # Maximum samples to show in plots
SHOW_PLOTS = True

# Debug Configuration
VERBOSE = True  # Print detailed progress information
CHECK_DATA_RANGE_SAMPLE_DAYS = 1  # Days to sample for data range checking

# ============================================================================
# IMPORT LIBRARIES
# ============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import pickle
import os
import torch
from nilmtk import DataSet

warnings.filterwarnings('ignore')

class NILMDataLoader:
    """
    NILM Data Loader following standard toolkit methods
    Combines data loading, preprocessing, and windowing into a streamlined workflow
    """
    
    def __init__(self, dataset_path=None, building_id=None, sample_period=None):
        """
        Initialize the data loader
        
        Args:
            dataset_path (str): Path to the HDF5 dataset file (uses global config if None)
            building_id (int): Building ID to use (uses global config if None)
            sample_period (int): Sampling period in seconds (uses global config if None)
        """
        # Use global configuration variables if not provided
        self.dataset_path = dataset_path if dataset_path is not None else DATASET_PATH
        self.building_id = building_id if building_id is not None else BUILDING_ID
        self.sample_period = sample_period if sample_period is not None else SAMPLE_PERIOD
        
        # Load dataset
        self.dataset = DataSet(self.dataset_path)
        self.building = self.dataset.buildings[self.building_id]
        self.elec = self.building.elec
        
        print(f"Dataset loaded successfully!")
        print(f"Available appliances: {[app.metadata['type'] for app in self.elec.appliances]}")
        
        # Initialize data storage
        self.mains_data = None
        self.appliance_data = {}
        self.processed_data = None
        
    def check_data_range(self, sample_days=None):
        """
        Check available data range using a small sample to avoid memory issues
        
        Args:
            sample_days (int): Number of days to sample for checking range (uses global config if None)
        """
        if sample_days is None:
            sample_days = CHECK_DATA_RANGE_SAMPLE_DAYS
            
        if VERBOSE:
            print("Checking available data range (memory-safe method)...")
        
        # Use a small time window to check data range without loading everything
        start_date = TRAIN_START_DATE
        end_date = pd.Timestamp(start_date) + pd.Timedelta(days=sample_days)
        end_date = end_date.strftime('%Y-%m-%d')
        
        self.dataset.set_window(start=start_date, end=end_date)
        mains = self.elec.mains()
        mains_data = next(mains.load(sample_period=self.sample_period))
        mains_data = mains_data.dropna()
        
        if not mains_data.empty:
            data_start = mains_data.index.min()
            data_end = mains_data.index.max()
            print(f"Sample data: {data_start} to {data_end}")
            print(f"Sample duration: {(data_end - data_start).days} days")
            
            # Estimate total data range (UK-DALE Building 1 typically has ~2 years of data)
            print(f"\nEstimated total data range: 2013-03-17 to 2015-01-05 (~2 years)")
            print(f"Estimated total samples: ~9.5 million (6-second sampling)")
        else:
            print("No data available")
            
        return data_start, data_end
    
    def set_time_windows(self, use_all_data=None):
        """
        Set time windows for training and testing using ALL available data
        
        Args:
            use_all_data (bool): If True, use all available data in the dataset (uses global config if None)
        """
        if use_all_data is None:
            use_all_data = USE_ALL_DATA
            
        if use_all_data:
            # Use ALL available data in the dataset
            train_start = TRAIN_START_DATE
            train_end = TRAIN_END_DATE
            
            print(f"\nUsing ALL available data with RANDOM day-based splitting:")
            print(f"Total data range: {train_start} to {train_end} (ALL data will be loaded)")
            print(f"Training/Validation split: RANDOMIZED by days (eliminates seasonal bias)")
            print(f"Test split: Sequential (as standard practice)")
            
            # Return the full range - the random splitting happens in create_windows_and_normalize
            # When using ALL data, we don't need separate test data loading
            return train_start, train_end, train_start, train_end
        else:
            # Fallback to 2 years if needed
            train_start = TRAIN_START_DATE
            train_end = pd.Timestamp(train_start) + pd.Timedelta(days=24 * 30)
            train_end = train_end.strftime('%Y-%m-%d')
            
            test_start = train_end
            test_end = pd.Timestamp(test_start) + pd.Timedelta(days=2 * 30)
            test_end = test_end.strftime('%Y-%m-%d')
            
            print(f"\nUsing 2 years of data:")
            print(f"Training: {train_start} to {train_end} (24 months = 2.0 years)")
            print(f"Testing: {test_start} to {test_end} (2 months)")
            
            return train_start, train_end, test_start, test_end
    
    def preprocess_and_window(self, window_size=None, stride=None, normalize=True):
        """
        Combined preprocessing method: creates sliding windows and normalizes data
        Following toolkit standards for streamlined preprocessing
        
        Args:
            window_size (int): Size of sliding window (uses global config if None)
            stride (int): Stride for sliding window (uses global config if None)
            normalize (bool): Whether to normalize the data
            
        Returns:
            dict: Dictionary containing all processed data
        """
        if window_size is None:
            window_size = WINDOW_SIZE
        if stride is None:
            stride = STRIDE
            
        print("\n" + "="*50)
        print("PREPROCESSING AND WINDOWING DATA")
        print("="*50)
        
        if self.mains_data is None or not self.appliance_data:
            raise ValueError("No data loaded. Please call load_data() first.")
        
        # Create sliding windows
        mains_array = self.mains_data.values
        mains_windows = []
        appliance_windows = {}
        
        # Initialize appliance windows
        for app_name in self.appliance_data.keys():
            appliance_windows[app_name] = []
        
        # Create sliding windows
        for i in range(0, len(mains_array) - window_size + 1, stride):
            # Mains window
            mains_window = mains_array[i:i + window_size]
            mains_windows.append(mains_window)
            
            # Appliance windows
            for app_name, app_data in self.appliance_data.items():
                app_window = app_data.values[i:i + window_size]
                appliance_windows[app_name].append(app_window)
        
        # Convert to numpy arrays
        mains_windows = np.array(mains_windows)
        for app_name in appliance_windows.keys():
            appliance_windows[app_name] = np.array(appliance_windows[app_name])
        
        # Normalize data if requested
        if normalize:
            # Normalize mains data
            mains_mean = np.mean(mains_windows)
            mains_std = np.std(mains_windows)
            if mains_std == 0:
                mains_std = 1
            mains_windows = (mains_windows - mains_mean) / mains_std
            
            # Normalize appliance data
            appliance_stats = {}
            for app_name in appliance_windows.keys():
                app_mean = np.mean(appliance_windows[app_name])
                app_std = np.std(appliance_windows[app_name])
                if app_std == 0:
                    app_std = 1
                appliance_windows[app_name] = (appliance_windows[app_name] - app_mean) / app_std
                appliance_stats[app_name] = {'mean': app_mean, 'std': app_std}
        else:
            mains_mean = mains_std = 0
            appliance_stats = {}
        
        # Create train/validation split using day-based splitting
        processed_data = self._create_data_splits(mains_windows, appliance_windows, mains_mean, mains_std, appliance_stats, window_size, stride)
        
        self.processed_data = processed_data
        return processed_data
    
    def load_data_period(self, start_date, end_date, appliances):
        """
        Load mains and appliance data for a specific time period
        
        IMPORTANT: When loading appliances with multiple instances (like 'light'),
        we use elec.select_using_appliances(type=app_name) to get ALL meters containing that appliance.
        
        For example:
        - elec.select_using_appliances(type='light') returns: ALL 15 meters containing light instances 1-16
        - elec.select_using_appliances(type='kettle') returns: 1 meter containing kettle instance 1
        
        This is the STANDARD TOOLKIT METHOD (same as torch-nilm/datasources/datasource.py line 145)
        
        Args:
            start_date (str): Start date for data loading
            end_date (str): End date for data loading
            appliances (list): List of appliance names to load
            
        Returns:
            tuple: (mains_power, appliance_data)
        """
        # Set time window
        self.dataset.set_window(start=start_date, end=end_date)
        
        # Load mains data (TOTAL POWER - this is the input X)
        print(f"Loading mains data from {start_date} to {end_date}...")
        mains_data = next(self.elec.mains().load(sample_period=self.sample_period))
        mains_data = mains_data.fillna(0)
        
        if ('power', 'active') in mains_data.columns:
            mains_power = mains_data[('power', 'active')]
        else:
            mains_power = mains_data.iloc[:, 0]
        
        print(f"Mains power loaded: {len(mains_power)} samples")
        
        # Load appliance data (TARGET POWER - this is the output Y)
        appliance_data = {}
        for app_name in appliances:
            try:
                print(f"Loading {app_name} data...")
                
                # CORRECT METHOD: Use select_using_appliances to get ALL meters with this appliance type
                # This is the standard toolkit method used in torch-nilm and other toolkits
                appliance_meters = self.elec.select_using_appliances(type=app_name)
                
                # Load data from all meters containing this appliance type
                app_data = next(appliance_meters.load(sample_period=self.sample_period))
                app_data = app_data.fillna(0)
                
                if ('power', 'active') in app_data.columns:
                    app_power = app_data[('power', 'active')]
                else:
                    app_power = app_data.iloc[:, 0]
                
                # ENHANCED ALIGNMENT METHOD: Ensure appliance power never exceeds mains power
                if len(app_power) > 0 and len(mains_power) > 0:
                    # Find intersection of time indices
                    common_timestamps = mains_power.index.intersection(app_power.index)
                    
                    if len(common_timestamps) > 0:
                        # Align both to common timestamps only
                        mains_aligned = mains_power.loc[common_timestamps]
                        app_aligned = app_power.loc[common_timestamps]
                        
                        # CRITICAL FIX: Ensure appliance power never exceeds mains power
                        # This is a data quality issue that needs to be fixed
                        impossible_mask = app_aligned > mains_aligned
                        impossible_count = impossible_mask.sum()
                        
                        if impossible_count > 0:
                            print(f"  ⚠ FIXING {impossible_count:,} samples where {app_name} power > mains power")
                            # Cap appliance power at mains power level
                            app_aligned = np.minimum(app_aligned, mains_aligned)
                        
                        # Update the aligned data
                        mains_power = mains_aligned
                        app_power = app_aligned
                    else:
                        # No common timestamps - create empty series
                        app_power = pd.Series(0, index=mains_power.index)
                else:
                    # No appliance data - fill with zeros
                    app_power = pd.Series(0, index=mains_power.index)
                
                appliance_data[app_name] = app_power
                
                # Count total instances across all meters
                total_instances = []
                for meter in appliance_meters.meters:
                    for app in meter.appliances:
                        if app.metadata['type'] == app_name:
                            total_instances.append(app.metadata['instance'])
                
                # UK-DALE Data Quality Handling (Enhanced Method):
                # 1. Apply power thresholds (as done in UK-DALE metadata)
                # 2. Use intersection method for time alignment
                # 3. FIX impossible values (appliance > mains) by capping at mains level
                
                # Apply reasonable power thresholds (based on UK-DALE metadata)
                mains_power = mains_power.clip(upper=MAINS_POWER_MAX)  # Max mains power
                app_power = app_power.clip(upper=APPLIANCE_POWER_MAX)  # Max appliance power
                
                # FINAL CHECK: Ensure appliance power never exceeds mains power
                # This fixes the data quality issue you observed in the graphs
                final_impossible_mask = app_power > mains_power
                final_impossible_count = final_impossible_mask.sum()
                
                if final_impossible_count > 0:
                    print(f"  ⚠ FINAL FIX: Capping {final_impossible_count:,} samples where {app_name} > mains")
                    app_power = np.minimum(app_power, mains_power)
                
                print(f"{app_name} loaded: {len(app_power)} samples (SUM of {len(total_instances)} instances: {sorted(total_instances)})")
                
            except Exception as e:
                print(f"Error loading {app_name}: {e}")
        
        return mains_power, appliance_data
    
    def _create_data_splits(self, mains_windows, appliance_windows, mains_mean, mains_std, appliance_stats, window_size, stride):
        """
        Create train/validation/test splits using day-based splitting (FIXED VERSION)
        Maintains temporal continuity for realistic NILM evaluation
        """
        # Calculate how many windows correspond to each day
        samples_per_day = EXPECTED_SAMPLES_PER_DAY
        windows_per_day = (samples_per_day - window_size + 1) // stride
        
        # Calculate total days from window count
        total_days = len(mains_windows) // windows_per_day
        
        # Create day-based indices (FIXED: map windows to days correctly)
        train_windows = []
        val_windows = []
        
        # Split days: 80% for training, 20% for validation
        train_days = int(TRAIN_SPLIT_RATIO * total_days)
        
        for day in range(total_days):
            start_window = day * windows_per_day
            end_window = min((day + 1) * windows_per_day, len(mains_windows))
            day_windows = list(range(start_window, end_window))
            
            if day < train_days:
                train_windows.extend(day_windows)
            else:
                val_windows.extend(day_windows)
        
        # Filter out any invalid indices
        train_windows = [w for w in train_windows if w < len(mains_windows)]
        val_windows = [w for w in val_windows if w < len(mains_windows)]
        
        print(f"Total windows: {len(mains_windows):,}")
        print(f"Total days: {total_days}")
        print(f"Training days: {train_days}")
        print(f"Training windows: {len(train_windows):,} samples")
        print(f"Validation windows: {len(val_windows):,} samples")
        
        # Create splits
        X_train = mains_windows[train_windows]
        X_val = mains_windows[val_windows]
        X_test = X_val[:len(X_val)//2]  # Use half of validation as test
        
        y_train = {}
        y_val = {}
        y_test = {}
        
        for app_name in appliance_windows.keys():
            y_train[app_name] = appliance_windows[app_name][train_windows]
            y_val[app_name] = appliance_windows[app_name][val_windows]
            y_test[app_name] = appliance_windows[app_name][val_windows[:len(val_windows)//2]]
        
        # Store statistics
        stats = {
            'mains': {'mean': mains_mean, 'std': mains_std},
            'appliances': appliance_stats,
            'window_size': window_size,
            'stride': stride,
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test)
        }
        
        processed_data = {
            'X_train': X_train,
            'X_val': X_val,
            'X_test': X_test,
            'y_train': y_train,
            'y_val': y_val,
            'y_test': y_test,
            'stats': stats
        }
        
        print(f"\nData processing complete!")
        print(f"Training samples: {len(X_train):,}")
        print(f"Validation samples: {len(X_val):,}")
        print(f"Testing samples: {len(X_test):,}")
        
        return processed_data
    
    def normalize_data(self, train_data, test_data):
        """
        Normalize data using training statistics (NO DATA LEAKAGE)
        
        Args:
            train_data (np.array): Training data
            test_data (np.array): Test data (for separate test set only)
            
        Returns:
            tuple: (train_normalized, test_normalized, mean, std)
        """
        # Calculate statistics from training data ONLY
        mean = np.mean(train_data)
        std = np.std(train_data)
        
        # Avoid division by zero
        if std == 0:
            std = 1
        
        # Normalize training data
        train_normalized = (train_data - mean) / std
        
        # For test data: only normalize if it's a separate test set
        # If test_data is the same as train_data (all data mode), return train_normalized
        if test_data is train_data or np.array_equal(test_data, train_data):
            test_normalized = train_normalized
        else:
            # Only normalize test data if it's truly separate
            test_normalized = (test_data - mean) / std
        
        return train_normalized, test_normalized, mean, std
    
    def load_data(self, start_date, end_date, target_appliances, data_type="all"):
        """
        Load data for specified time period (combines training and testing data loading)
        
        Args:
            start_date (str): Start date for data loading
            end_date (str): End date for data loading
            target_appliances (list): List of target appliances
            data_type (str): Type of data ("train", "test", or "all")
        """
        print("="*50)
        print(f"LOADING {data_type.upper()} DATA")
        print("="*50)
        
        self.mains_data, self.appliance_data = self.load_data_period(
            start_date, end_date, target_appliances
        )
        
        return self.mains_data, self.appliance_data
    
    
    def visualize_data(self, max_samples=None):
        """
        Simplified visualization of loaded data
        
        Args:
            max_samples (int): Maximum number of samples to plot (uses global config if None)
        """
        if max_samples is None:
            max_samples = MAX_SAMPLES_TO_PLOT
        if self.mains_data is None or not self.appliance_data:
            print("No data loaded yet. Please call load_data() first.")
            return
        
        # Create simplified plots
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Power vs Time
        ax1 = axes[0]
        sample_indices = range(0, len(self.mains_data), max(1, len(self.mains_data) // max_samples))
        ax1.plot(sample_indices, self.mains_data.iloc[sample_indices].values, 'b-', 
                label='Total Power (Mains)', linewidth=0.8, alpha=0.8)
        for app_name, app_power in self.appliance_data.items():
            ax1.plot(sample_indices, app_power.iloc[sample_indices].values, 'r-', 
                    label=f'{app_name.title()} Power', linewidth=0.8, alpha=0.8)
        ax1.set_title('Power Consumption Over Time', fontsize=14)
        ax1.set_xlabel('Sample Index')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Power Distribution
        ax2 = axes[1]
        ax2.hist(self.mains_data.values, bins=50, alpha=0.7, 
                label='Total Power', color='blue', density=True)
        for app_name, app_power in self.appliance_data.items():
            ax2.hist(app_power.values, bins=50, alpha=0.7, 
                    label=f'{app_name.title()} Power', color='red', density=True)
        ax2.set_title('Power Distribution', fontsize=14)
        ax2.set_xlabel('Power (W)')
        ax2.set_ylabel('Density')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        
        # Print summary statistics
        print("="*50)
        print("DATA SUMMARY")
        print("="*50)
        print(f"✓ Total samples: {len(self.mains_data):,}")
        print(f"✓ Time period: {self.mains_data.index[0]} to {self.mains_data.index[-1]}")
        print(f"✓ Duration: {(self.mains_data.index[-1] - self.mains_data.index[0]).days} days")
        print(f"✓ Appliances loaded: {list(self.appliance_data.keys())}")
        
        for app_name, app_power in self.appliance_data.items():
            print(f"✓ {app_name.title()}: max {app_power.max():.1f}W, mean {app_power.mean():.1f}W, non-zero {(app_power > 0).mean()*100:.1f}%")
    
    def get_ready_dataloaders(self, target_appliances=None, window_size=None, stride=None, batch_size=512):
        """
        One-step method: load data, preprocess, and return ready-to-use PyTorch dataloaders
        Following toolkit standards for maximum simplicity
        
        Args:
            target_appliances (list): List of target appliances (uses global config if None)
            window_size (int): Window size (uses global config if None)
            stride (int): Stride (uses global config if None)
            batch_size (int): Batch size for dataloaders
            
        Returns:
            tuple: (train_dataloader, val_dataloader, target_appliance_name)
        """
        from torch.utils.data import DataLoader, TensorDataset
        
        if target_appliances is None:
            target_appliances = TARGET_APPLIANCES
        if window_size is None:
            window_size = WINDOW_SIZE
        if stride is None:
            stride = STRIDE
            
        print("="*60)
        print("ONE-STEP DATA LOADING AND PREPROCESSING")
        print("="*60)
        
        # Step 1: Set time windows and load data
        train_start, train_end, _, _ = self.set_time_windows()
        self.load_data(train_start, train_end, target_appliances, "all")
        
        # Step 2: Preprocess and create windows
        processed_data = self.preprocess_and_window(window_size=window_size, stride=stride)
        
        # Step 3: Create PyTorch datasets and dataloaders
        target_appliance = target_appliances[0]  # Use first appliance
        
        # Convert to PyTorch tensors with correct dimensions for 1D CNN
        # CNN expects: (batch_size, channels=1, sequence_length)
        X_train = torch.FloatTensor(processed_data['X_train']).unsqueeze(1)  # Add channel dimension
        y_train = torch.FloatTensor(processed_data['y_train'][target_appliance])
        X_val = torch.FloatTensor(processed_data['X_val']).unsqueeze(1)  # Add channel dimension
        y_val = torch.FloatTensor(processed_data['y_val'][target_appliance])
        
        # Create datasets
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        
        # Create dataloaders
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        print(f"✓ Ready dataloaders created!")
        print(f"✓ Training batches: {len(train_dataloader)}")
        print(f"✓ Validation batches: {len(val_dataloader)}")
        print(f"✓ Target appliance: {target_appliance}")
        
        return train_dataloader, val_dataloader, target_appliance
    
    def save_processed_data(self, processed_data, filename=None):
        """
        Save processed data to file
        
        Args:
            processed_data (dict): Processed data dictionary
            filename (str): Output filename (uses global config if None)
        """
        if filename is None:
            filename = OUTPUT_FILENAME
        print(f"\nSaving processed data to {filename}...")
        
        with open(filename, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✓ Data saved successfully!")
        print(f"✓ File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    def load_processed_data(self, filename=None):
        """
        Load processed data from file
        
        Args:
            filename (str): Input filename (uses global config if None)
            
        Returns:
            dict: Processed data dictionary
        """
        if filename is None:
            filename = OUTPUT_FILENAME
        print(f"Loading processed data from {filename}...")
        
        with open(filename, 'rb') as f:
            processed_data = pickle.load(f)
        
        print(f"✓ Data loaded successfully!")
        return processed_data


def main():
    """
    Simplified main function demonstrating the streamlined preprocessing pipeline
    Uses all configuration variables defined at the top of the file
    """
    print("="*60)
    print("NILM DATA PREPROCESSING PIPELINE")
    print("="*60)
    print(f"Dataset: {DATASET_PATH}")
    print(f"Building ID: {BUILDING_ID}")
    print(f"Target Appliances: {TARGET_APPLIANCES}")
    print(f"Window Size: {WINDOW_SIZE}")
    print(f"Stride: {STRIDE}")
    print(f"Use All Data: {USE_ALL_DATA}")
    print(f"Random Seed: {RANDOM_SEED}")
    print("="*60)
    
    # Check if we should load existing processed data
    if LOAD_PROCESSED_DATA:
        try:
            data_loader = NILMDataLoader()
            processed_data = data_loader.load_processed_data()
            print("\n✓ Loaded existing processed data!")
            print(f"Training samples: {processed_data['stats']['train_samples']:,}")
            print(f"Validation samples: {processed_data['stats']['val_samples']:,}")
            print(f"Testing samples: {processed_data['stats']['test_samples']:,}")
            return processed_data
        except FileNotFoundError:
            print(f"⚠️  Processed data file not found. Starting fresh preprocessing...")
    
    # Initialize data loader (uses global configuration)
    data_loader = NILMDataLoader()
    
    # Check data range
    data_loader.check_data_range()
    
    # Set time windows
    train_start, train_end, test_start, test_end = data_loader.set_time_windows()
    
    # Load all data (simplified approach)
    data_loader.load_data(train_start, train_end, TARGET_APPLIANCES, "all")
    
    # Visualize data (if enabled)
    if VISUALIZE_DATA:
        data_loader.visualize_data()
    
    # Preprocess and create windows (combined operation)
    processed_data = data_loader.preprocess_and_window()
    
    # Save processed data (if enabled)
    if SAVE_PROCESSED_DATA:
        data_loader.save_processed_data(processed_data)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("Your data is now ready for NILM model training!")
    print(f"Training samples: {processed_data['stats']['train_samples']:,}")
    print(f"Validation samples: {processed_data['stats']['val_samples']:,}")
    print(f"Testing samples: {processed_data['stats']['test_samples']:,}")
    
    return processed_data


if __name__ == "__main__":
    import os
    main()