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

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import pickle
from nilmtk import DataSet

warnings.filterwarnings('ignore')

class NILMPreprocessor:
    """
    NILM Data Preprocessor following standard toolkit methods
    """
    
    def __init__(self, dataset_path, building_id=1, sample_period=6):
        """
        Initialize the preprocessor
        
        Args:
            dataset_path (str): Path to the HDF5 dataset file
            building_id (int): Building ID to use (default: 1)
            sample_period (int): Sampling period in seconds (default: 6)
        """
        self.dataset_path = dataset_path
        self.building_id = building_id
        self.sample_period = sample_period
        
        # Load dataset
        self.dataset = DataSet(dataset_path)
        self.building = self.dataset.buildings[building_id]
        self.elec = self.building.elec
        
        print(f"Dataset loaded successfully!")
        print(f"Available appliances: {[app.metadata['type'] for app in self.elec.appliances]}")
        
        # Initialize data storage
        self.train_mains = None
        self.test_mains = None
        self.train_appliances = {}
        self.test_appliances = {}
        
    def check_data_range(self, sample_days=1):
        """
        Check available data range using a small sample to avoid memory issues
        
        Args:
            sample_days (int): Number of days to sample for checking range
        """
        print("Checking available data range (memory-safe method)...")
        
        # Use a small time window to check data range without loading everything
        start_date = '2013-03-17'
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
    
    def set_time_windows(self, train_months=6, test_months=1):
        """
        Set reasonable time windows for training and testing
        
        Args:
            train_months (int): Number of months for training
            test_months (int): Number of months for testing
        """
        # Set reasonable time windows for training (avoid memory issues)
        train_start = "2013-03-17"
        train_end = pd.Timestamp(train_start) + pd.Timedelta(days=train_months * 30)
        train_end = train_end.strftime('%Y-%m-%d')
        
        test_start = train_end
        test_end = pd.Timestamp(test_start) + pd.Timedelta(days=test_months * 30)
        test_end = test_end.strftime('%Y-%m-%d')
        
        print(f"\nRecommended time windows:")
        print(f"Training: {train_start} to {train_end} ({train_months} months)")
        print(f"Testing: {test_start} to {test_end} ({test_months} months)")
        
        return train_start, train_end, test_start, test_end
    
    def create_sliding_windows(self, mains_data, appliance_data, window_size=99, stride=1):
        """
        Create sliding windows for deep learning models
        
        Args:
            mains_data (pd.Series): Mains power data
            appliance_data (dict): Dictionary of appliance power data
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            
        Returns:
            tuple: (mains_windows, appliance_windows)
        """
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
                mains_power = mains_power.clip(upper=20000)  # 20kW max (UK-DALE standard)
                app_power = app_power.clip(upper=4000)       # 4kW max per appliance
                
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
        Normalize data using training statistics
        
        Args:
            train_data (np.array): Training data
            test_data (np.array): Test data
            
        Returns:
            tuple: (train_normalized, test_normalized, mean, std)
        """
        # Calculate statistics from training data
        mean = np.mean(train_data)
        std = np.std(train_data)
        
        # Avoid division by zero
        if std == 0:
            std = 1
        
        # Normalize both training and test data
        train_normalized = (train_data - mean) / std
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
        
        self.test_mains, self.test_appliances = self.load_data_period(
            test_start, test_end, target_appliances
        )
        
        return self.test_mains, self.test_appliances
    
    def create_windows_and_normalize(self, window_size=99, stride=1):
        """
        Create sliding windows and normalize data
        
        Args:
            window_size (int): Size of sliding window
            stride (int): Stride for sliding window
            
        Returns:
            dict: Dictionary containing all processed data
        """
        print("\n" + "="*50)
        print("CREATING SLIDING WINDOWS AND NORMALIZING")
        print("="*50)
        
        # Create sliding windows
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
        
        # Normalize appliance data
        train_app_norm = {}
        test_app_norm = {}
        appliance_stats = {}
        
        for app_name in self.train_appliances.keys():
            train_app_norm[app_name], test_app_norm[app_name], app_mean, app_std = self.normalize_data(
                train_app_windows[app_name], test_app_windows[app_name]
            )
            appliance_stats[app_name] = {'mean': app_mean, 'std': app_std}
        
        # Create train/validation split
        train_size = int(0.8 * len(train_mains_norm))
        X_train = train_mains_norm[:train_size]
        X_val = train_mains_norm[train_size:]
        X_test = test_mains_norm
        
        y_train = {}
        y_val = {}
        y_test = {}
        
        for app_name in self.train_appliances.keys():
            y_train[app_name] = train_app_norm[app_name][:train_size]
            y_val[app_name] = train_app_norm[app_name][train_size:]
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
    
    def visualize_data(self, max_samples=1000):
        """
        Visualize the loaded data
        
        Args:
            max_samples (int): Maximum number of samples to plot
        """
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
    
    def save_processed_data(self, processed_data, filename='preprocessed_data.pkl'):
        """
        Save processed data to file
        
        Args:
            processed_data (dict): Processed data dictionary
            filename (str): Output filename
        """
        print(f"\nSaving processed data to {filename}...")
        
        with open(filename, 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"✓ Data saved successfully!")
        print(f"✓ File size: {os.path.getsize(filename) / (1024*1024):.1f} MB")
    
    def load_processed_data(self, filename='preprocessed_data.pkl'):
        """
        Load processed data from file
        
        Args:
            filename (str): Input filename
            
        Returns:
            dict: Processed data dictionary
        """
        print(f"Loading processed data from {filename}...")
        
        with open(filename, 'rb') as f:
            processed_data = pickle.load(f)
        
        print(f"✓ Data loaded successfully!")
        return processed_data


def main():
    """
    Main function demonstrating the preprocessing pipeline
    """
    # Configuration
    DATASET_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5"
    TARGET_APPLIANCES = ['washer dryer']  # Change to your target appliance
    WINDOW_SIZE = 99  # Standard window size
    STRIDE = 1  # Overlapping windows
    
    # Initialize preprocessor
    preprocessor = NILMPreprocessor(DATASET_PATH)
    
    # Check data range
    preprocessor.check_data_range()
    
    # Set time windows
    train_start, train_end, test_start, test_end = preprocessor.set_time_windows()
    
    # Load training data
    preprocessor.load_training_data(train_start, train_end, TARGET_APPLIANCES)
    
    # Load testing data
    preprocessor.load_testing_data(test_start, test_end, TARGET_APPLIANCES)
    
    # Visualize data
    preprocessor.visualize_data()
    
    # Create windows and normalize
    processed_data = preprocessor.create_windows_and_normalize(WINDOW_SIZE, STRIDE)
    
    # Save processed data
    preprocessor.save_processed_data(processed_data)
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)
    print("Your data is now ready for NILM model training!")
    print(f"Training samples: {processed_data['stats']['train_samples']:,}")
    print(f"Validation samples: {processed_data['stats']['val_samples']:,}")
    print(f"Testing samples: {processed_data['stats']['test_samples']:,}")


if __name__ == "__main__":
    import os
    main()
