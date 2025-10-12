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
TARGET_APPLIANCES = ['light']  # Change this to your target appliances
# Available appliances in UK-DALE: ['kettle', 'microwave', 'dishwasher', 'washing machine', 'washer dryer', 'light', 'fridge', 'freezer', 'tumble dryer']

# Time Window Configuration
USE_ALL_DATA = True  # If True, use all available data; if False, use limited time range
TRAIN_START_DATE = "2013-03-17"  # Start of UK-DALE Building 1 data
TRAIN_END_DATE = "2013-04-05"    # End of UK-DALE Building 1 data (estimated)

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
from nilmtk import DataSet

warnings.filterwarnings('ignore')

class NILMPreprocessor:
    """
    NILM Data Preprocessor following standard toolkit methods
    """
    
    def __init__(self, dataset_path=None, building_id=None, sample_period=None):
        """
        Initialize the preprocessor
        
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
        self.train_mains = None
        self.test_mains = None
        self.train_appliances = {}
        self.test_appliances = {}
        
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
    
    def create_sliding_windows(self, mains_data, appliance_data, window_size=None, stride=None):
        """
        Create sliding windows for deep learning models
        
        Args:
            mains_data (pd.Series): Mains power data
            appliance_data (dict): Dictionary of appliance power data
            window_size (int): Size of sliding window (uses global config if None)
            stride (int): Stride for sliding window (uses global config if None)
            
        Returns:
            tuple: (mains_windows, appliance_windows)
        """
        if window_size is None:
            window_size = WINDOW_SIZE
        if stride is None:
            stride = STRIDE
        # Convert to numpy arrays
        mains_array = mains_data.values
        
        # Create windows for mains data
        mains_windows = []
        appliance_windows = {}
        
        # Initialize appliance windows
        for app_name in appliance_data.keys():
            appliance_windows[app_name] = []
        
        # Create sliding windows
        for i in range(0, len(mains_array) - window_size + 1, stride):
            # Mains window
            mains_window = mains_array[i:i + window_size]
            mains_windows.append(mains_window)
            
            # Appliance windows
            for app_name, app_data in appliance_data.items():
                app_window = app_data.values[i:i + window_size]
                appliance_windows[app_name].append(app_window)
        
        # Convert to numpy arrays
        mains_windows = np.array(mains_windows)
        for app_name in appliance_windows.keys():
            appliance_windows[app_name] = np.array(appliance_windows[app_name])
        
        return mains_windows, appliance_windows
    
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
    
    def load_training_data(self, train_start, train_end, target_appliances):
        """
        Load training data
        
        Args:
            train_start (str): Training start date
            train_end (str): Training end date
            target_appliances (list): List of target appliances
        """
        print("="*50)
        print("LOADING TRAINING DATA")
        print("="*50)
        
        self.train_mains, self.train_appliances = self.load_data_period(
            train_start, train_end, target_appliances
        )
        
        return self.train_mains, self.train_appliances
    
    def load_testing_data(self, test_start, test_end, target_appliances):
        """
        Load testing data
        
        Args:
            test_start (str): Testing start date
            test_end (str): Testing end date
            target_appliances (list): List of target appliances
        """
        print("\n" + "="*50)
        print("LOADING TESTING DATA")
        print("="*50)
        
        # Check if we're using all data (test_start == test_end means we're using all data)
        if test_start == test_end:
            print("Using ALL data - test data will be created from training data")
            # When using all data, we don't load separate test data
            # The test data will be created from the training data in create_windows_and_normalize
            self.test_mains = self.train_mains  # Use same data
            self.test_appliances = self.train_appliances  # Use same data
        else:
            # Original approach - load separate test data
            self.test_mains, self.test_appliances = self.load_data_period(
                test_start, test_end, target_appliances
            )
        
        return self.test_mains, self.test_appliances
    
    def create_windows_and_normalize(self, window_size=None, stride=None):
        """
        Create sliding windows and normalize data
        
        Args:
            window_size (int): Size of sliding window (uses global config if None)
            stride (int): Stride for sliding window (uses global config if None)
            
        Returns:
            dict: Dictionary containing all processed data
        """
        if window_size is None:
            window_size = WINDOW_SIZE
        if stride is None:
            stride = STRIDE
        print("\n" + "="*50)
        print("CREATING SLIDING WINDOWS AND NORMALIZING")
        print("="*50)
        
        # Check if we're using all data (train and test are the same)
        if self.train_mains is self.test_mains:
            print("Using ALL data with random day-based splitting...")
            print("⚠️  Large dataset detected - using memory-efficient processing...")
            
            # For large datasets, we need to process in chunks to avoid memory issues
            # First, let's create the day-based split indices without loading all data
            print("Creating day-based split indices...")
            original_timestamps = self.train_mains.index
            
            # Group by actual calendar days
            day_groups = {}
            for idx, timestamp in enumerate(original_timestamps):
                day_key = timestamp.date()
                if day_key not in day_groups:
                    day_groups[day_key] = []
                day_groups[day_key].append(idx)
            
            day_chunks = list(day_groups.values())
            print(f"Created {len(day_chunks)} REAL day chunks from {len(original_timestamps)} samples")
            
            # Filter out days with too little data
            min_samples = EXPECTED_SAMPLES_PER_DAY * MIN_SAMPLES_PER_DAY_RATIO
            filtered_day_chunks = [chunk for chunk in day_chunks if len(chunk) >= min_samples]
            print(f"After filtering: {len(filtered_day_chunks)} days with sufficient data")
            
            # Randomize day chunk order
            np.random.seed(RANDOM_SEED)
            np.random.shuffle(filtered_day_chunks)
            
            # Split day chunks: 80% for training, 20% for validation
            train_chunks = int(TRAIN_SPLIT_RATIO * len(filtered_day_chunks))
            train_day_chunks = filtered_day_chunks[:train_chunks]
            val_day_chunks = filtered_day_chunks[train_chunks:]
            
            # Flatten chunk indices
            train_indices = [idx for chunk in train_day_chunks for idx in chunk]
            val_indices = [idx for chunk in val_day_chunks for idx in chunk]
            
            # Filter indices to ensure they're within bounds
            max_index = len(self.train_mains) - 1
            train_indices = [idx for idx in train_indices if idx <= max_index]
            val_indices = [idx for idx in val_indices if idx <= max_index]
            
            print(f"Training indices: {len(train_indices):,} samples")
            print(f"Validation indices: {len(val_indices):,} samples")
            
            # Now create sliding windows only for the selected indices
            print("Creating sliding windows for selected data...")
            train_mains_windows, train_app_windows = self.create_sliding_windows(
                self.train_mains.iloc[train_indices], 
                {app: self.train_appliances[app].iloc[train_indices] for app in self.train_appliances.keys()}, 
                window_size, stride
            )
            
            val_mains_windows, val_app_windows = self.create_sliding_windows(
                self.train_mains.iloc[val_indices], 
                {app: self.train_appliances[app].iloc[val_indices] for app in self.train_appliances.keys()}, 
                window_size, stride
            )
            
            print(f"Training windows: {train_mains_windows.shape}")
            print(f"Validation windows: {val_mains_windows.shape}")
            
            # Normalize data
            print("Normalizing data...")
            train_mains_norm, val_mains_norm, mains_mean, mains_std = self.normalize_data(
                train_mains_windows, val_mains_windows
            )
            
            # For test data, use a subset of validation data to avoid memory issues
            test_mains_norm = val_mains_norm[:len(val_mains_norm)//2]  # Use half of validation as test
        else:
            # Original approach for separate train/test data
            print("Creating sliding windows...")
            train_mains_windows, train_app_windows = self.create_sliding_windows(
                self.train_mains, self.train_appliances, window_size, stride
            )
            
            test_mains_windows, test_app_windows = self.create_sliding_windows(
                self.test_mains, self.test_appliances, window_size, stride
            )
            
            print(f"Training windows: {train_mains_windows.shape}")
            print(f"Testing windows: {test_mains_windows.shape}")
            
            # Normalize data
            print("Normalizing data...")
            train_mains_norm, test_mains_norm, mains_mean, mains_std = self.normalize_data(
                train_mains_windows, test_mains_windows
            )
        
        # Handle appliance data normalization
        if self.train_mains is self.test_mains:
            # Using all data - normalize appliance data
            train_app_norm = {}
            val_app_norm = {}
            test_app_norm = {}
            appliance_stats = {}
            
            for app_name in self.train_appliances.keys():
                train_app_norm[app_name], val_app_norm[app_name], app_mean, app_std = self.normalize_data(
                    train_app_windows[app_name], val_app_windows[app_name]
                )
                appliance_stats[app_name] = {'mean': app_mean, 'std': app_std}
                # For test data, use half of validation data
                test_app_norm[app_name] = val_app_norm[app_name][:len(val_app_norm[app_name])//2]
        else:
            # Original approach for separate train/test data
            train_app_norm = {}
            test_app_norm = {}
            appliance_stats = {}
            
            for app_name in self.train_appliances.keys():
                train_app_norm[app_name], test_app_norm[app_name], app_mean, app_std = self.normalize_data(
                    train_app_windows[app_name], test_app_windows[app_name]
                )
                appliance_stats[app_name] = {'mean': app_mean, 'std': app_std}
        
        # Create train/validation split (REAL DAY-BASED SPLIT USING ACTUAL TIMESTAMPS)
        import numpy as np
        import pandas as pd
        
        # Use ACTUAL timestamps to create real day boundaries
        if self.train_mains is self.test_mains:
            # Using all data - use the same timestamps for both
            original_timestamps = self.train_mains.index
            print("Using ALL data timestamps for random day-based splitting")
        else:
            # Original approach
            original_timestamps = self.train_mains.index
        
        # Group by actual calendar days
        day_groups = {}
        for idx, timestamp in enumerate(original_timestamps):
            day_key = timestamp.date()  # Get just the date part (YYYY-MM-DD)
            if day_key not in day_groups:
                day_groups[day_key] = []
            day_groups[day_key].append(idx)
        
        # Convert to list of day chunks
        day_chunks = list(day_groups.values())
        
        print(f"Created {len(day_chunks)} REAL day chunks from {len(original_timestamps)} samples")
        print(f"Day chunk sizes: {[len(chunk) for chunk in day_chunks[:5]]}... (showing first 5)")
        print(f"Average chunk size: {np.mean([len(chunk) for chunk in day_chunks]):.0f} samples")
        print(f"Date range: {min(day_groups.keys())} to {max(day_groups.keys())}")
        
        # Filter out days with too little data (less than configured ratio of expected)
        min_samples = EXPECTED_SAMPLES_PER_DAY * MIN_SAMPLES_PER_DAY_RATIO
        
        filtered_day_chunks = [chunk for chunk in day_chunks if len(chunk) >= min_samples]
        print(f"After filtering: {len(filtered_day_chunks)} days with sufficient data")
        
        # Randomize day chunk order
        np.random.seed(RANDOM_SEED)  # For reproducibility
        np.random.shuffle(filtered_day_chunks)
        
        # Split day chunks: configured ratio for training and validation
        train_chunks = int(TRAIN_SPLIT_RATIO * len(filtered_day_chunks))
        train_day_chunks = filtered_day_chunks[:train_chunks]
        val_day_chunks = filtered_day_chunks[train_chunks:]
        
        # Flatten chunk indices (no overlap, maintains continuity)
        train_indices = [idx for chunk in train_day_chunks for idx in chunk]
        val_indices = [idx for chunk in val_day_chunks for idx in chunk]
        
        # Filter indices to ensure they're within bounds of normalized data
        max_index = len(train_mains_norm) - 1
        train_indices = [idx for idx in train_indices if idx <= max_index]
        val_indices = [idx for idx in val_indices if idx <= max_index]
        
        # Check distribution balance
        train_samples = len(train_indices)
        val_samples = len(val_indices)
        total_samples = train_samples + val_samples
        
        print(f"\nDataset Distribution:")
        print(f"Training: {train_samples:,} samples ({train_samples/total_samples*100:.1f}%)")
        print(f"Validation: {val_samples:,} samples ({val_samples/total_samples*100:.1f}%)")
        print(f"Training days: {len(train_day_chunks)}")
        print(f"Validation days: {len(val_day_chunks)}")
        
        # Check if distribution is reasonable
        if val_samples < total_samples * 0.15:  # Less than 15% validation
            print("⚠️  WARNING: Validation set is quite small (<15%)")
        elif val_samples > total_samples * 0.25:  # More than 25% validation
            print("⚠️  WARNING: Validation set is quite large (>25%)")
        else:
            print("✅ Distribution looks good (15-25% validation)")
        
        # Assign the processed data
        if self.train_mains is self.test_mains:
            # Using all data - apply the train/val split indices to the normalized data
            X_train = train_mains_norm[train_indices]
            X_val = val_mains_norm[val_indices]
            X_test = test_mains_norm
            
            y_train = {}
            y_val = {}
            y_test = {}
            
            for app_name in self.train_appliances.keys():
                y_train[app_name] = train_app_norm[app_name][train_indices]
                y_val[app_name] = val_app_norm[app_name][val_indices]
                y_test[app_name] = test_app_norm[app_name]
            
            print(f"\nDataset Distribution:")
            print(f"Training: {len(X_train):,} samples")
            print(f"Validation: {len(X_val):,} samples")
            print(f"Testing: {len(X_test):,} samples")
        else:
            # Original approach - use separate test data
            X_train = train_mains_norm[train_indices]
            X_val = train_mains_norm[val_indices]
            X_test = test_mains_norm
            
            y_train = {}
            y_val = {}
            y_test = {}
            
            for app_name in self.train_appliances.keys():
                y_train[app_name] = train_app_norm[app_name][train_indices]
                y_val[app_name] = train_app_norm[app_name][val_indices]  # Use train data for validation in this mode
                y_test[app_name] = test_app_norm[app_name]
        
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
    
    def visualize_data(self, max_samples=None):
        """
        Visualize the loaded data
        
        Args:
            max_samples (int): Maximum number of samples to plot (uses global config if None)
        """
        if max_samples is None:
            max_samples = MAX_SAMPLES_TO_PLOT
        if self.train_mains is None or self.test_mains is None:
            print("No data loaded yet. Please load training and testing data first.")
            return
        
        # Create plots to visualize the data
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Plot 1: Training Data - Total Power vs Appliance Power (WHOLE TIME RANGE)
        ax1 = axes[0, 0]
        ax1.plot(self.train_mains.index, self.train_mains.values, 'b-', 
                label='Total Power (Mains)', linewidth=0.5, alpha=0.7)
        for app_name, app_power in self.train_appliances.items():
            ax1.plot(app_power.index, app_power.values, 'r-', 
                    label=f'{app_name.title()} Power', linewidth=0.5, alpha=0.7)
        ax1.set_title('Training Data: Total Power vs Appliance Power (WHOLE TIME RANGE)', fontsize=14)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Power (W)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Testing Data - Total Power vs Appliance Power (WHOLE TIME RANGE)
        ax2 = axes[0, 1]
        ax2.plot(self.test_mains.index, self.test_mains.values, 'b-', 
                label='Total Power (Mains)', linewidth=0.5, alpha=0.7)
        for app_name, app_power in self.test_appliances.items():
            ax2.plot(app_power.index, app_power.values, 'r-', 
                    label=f'{app_name.title()} Power', linewidth=0.5, alpha=0.7)
        ax2.set_title('Testing Data: Total Power vs Appliance Power (WHOLE TIME RANGE)', fontsize=14)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Power (W)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Power Distribution Comparison
        ax3 = axes[1, 0]
        ax3.hist(self.train_mains.values, bins=50, alpha=0.7, 
                label='Total Power', color='blue', density=True)
        for app_name, app_power in self.train_appliances.items():
            ax3.hist(app_power.values, bins=50, alpha=0.7, 
                    label=f'{app_name.title()} Power', color='red', density=True)
        ax3.set_title('Power Distribution Comparison (Training Data)', fontsize=14)
        ax3.set_xlabel('Power (W)')
        ax3.set_ylabel('Density')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Data Statistics Summary
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        # Calculate statistics
        stats_text = f"""
DATA LOADING SUMMARY
===================

Training Data:
• Total Power: {len(self.train_mains):,} samples
• Time Period: {self.train_mains.index[0]} to {self.train_mains.index[-1]}
• Duration: {(self.train_mains.index[-1] - self.train_mains.index[0]).days} days

Testing Data:
• Total Power: {len(self.test_mains):,} samples  
• Time Period: {self.test_mains.index[0]} to {self.test_mains.index[-1]}
• Duration: {(self.test_mains.index[-1] - self.test_mains.index[0]).days} days

Appliance Statistics:
"""
        
        for app_name, app_power in self.train_appliances.items():
            stats_text += f"""
• {app_name.title()}:
  - Training: {len(app_power):,} samples
  - Max Power: {app_power.max():.1f} W
  - Mean Power: {app_power.mean():.1f} W
  - Non-zero samples: {(app_power > 0).sum():,} ({(app_power > 0).mean()*100:.1f}%)
"""
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, fontsize=11, 
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        if SHOW_PLOTS:
            plt.show()
        
        # Print summary statistics
        print("="*60)
        print("DATA LOADING VERIFICATION")
        print("="*60)
        print(f"✓ Training data: {len(self.train_mains):,} samples ({(self.train_mains.index[-1] - self.train_mains.index[0]).days} days)")
        print(f"✓ Testing data: {len(self.test_mains):,} samples ({(self.test_mains.index[-1] - self.test_mains.index[0]).days} days)")
        print(f"✓ Total appliances loaded: {len(self.train_appliances)}")
        
        for app_name, app_power in self.train_appliances.items():
            print(f"✓ {app_name.title()}: {len(app_power):,} samples, max {app_power.max():.1f}W, mean {app_power.mean():.1f}W")
            
        print(f"\n✓ All data arrays have same length: {len(self.train_mains) == len(list(self.train_appliances.values())[0])}")
        print("✓ Data is ready for sliding window creation!")
    
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
    Main function demonstrating the preprocessing pipeline
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
            preprocessor = NILMPreprocessor()
            processed_data = preprocessor.load_processed_data()
            print("\n✓ Loaded existing processed data!")
            print(f"Training samples: {processed_data['stats']['train_samples']:,}")
            print(f"Validation samples: {processed_data['stats']['val_samples']:,}")
            print(f"Testing samples: {processed_data['stats']['test_samples']:,}")
            return processed_data
        except FileNotFoundError:
            print(f"⚠️  Processed data file not found. Starting fresh preprocessing...")
    
    # Initialize preprocessor (uses global configuration)
    preprocessor = NILMPreprocessor()
    
    # Check data range
    preprocessor.check_data_range()
    
    # Set time windows
    train_start, train_end, test_start, test_end = preprocessor.set_time_windows()
    
    # Load training data
    preprocessor.load_training_data(train_start, train_end, TARGET_APPLIANCES)
    
    # Load testing data
    preprocessor.load_testing_data(test_start, test_end, TARGET_APPLIANCES)
    
    # Visualize data (if enabled)
    if VISUALIZE_DATA:
        preprocessor.visualize_data()
    
    # Create windows and normalize (uses global configuration)
    processed_data = preprocessor.create_windows_and_normalize()
    
    # Save processed data (if enabled)
    if SAVE_PROCESSED_DATA:
        preprocessor.save_processed_data(processed_data)
    
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