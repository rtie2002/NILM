"""
Data loading and processing utilities for NILM.
"""

from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from pathlib import Path

# Try to import NILMTK, but make it optional
try:
    from nilmtk import DataSet
    NILMTK_AVAILABLE = True
except ImportError:
    NILMTK_AVAILABLE = False

class DataLoader:
    """Class to handle loading and processing of NILM datasets."""
    
    def __init__(self, dataset_path: Optional[str] = None):
        """Initialize the data loader.
        
        Args:
            dataset_path: Path to the dataset file (e.g., .h5 for NILMTK)
        """
        self.dataset_path = Path(dataset_path) if dataset_path else None
        self.dataset = None
        
    def load_ukdale(
        self, 
        building: int = 1, 
        start_time: str = "2013-11-08 18:00:00", 
        end_time: str = "2013-11-08 18:02:00"
    ) -> Tuple[Any, Any, Any, Any, Any]:
        """Load data from UK-DALE dataset.
        
        Args:
            building: Building number
            start_time: Start time for the data window
            end_time: End time for the data window
            
        Returns:
            Tuple containing:
                - mains_meter: Mains meter object
                - appliance_meter: Appliance meter object
                - mains_power: Mains power series
                - appliance_power: Appliance power series
                - dataset: The dataset object (needs to be closed after use)
        """
        if not NILMTK_AVAILABLE:
            raise ImportError("NILMTK is not installed. Please install it with 'pip install nilmtk'")
            
        if not self.dataset_path or not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found at {self.dataset_path}")
        
        # Initialize dataset
        dataset = DataSet(str(self.dataset_path))
        
        try:
            # Set the time window
            dataset.set_window(start=start_time, end=end_time)
            
            # Get the mains meter and appliance meter
            mains_meter = dataset.buildings[building].elec.mains()
            appliance_meter = dataset.buildings[building].elec['kettle']  # Default to kettle
            
            # Load power data
            mains_power = next(mains_meter.power_series_all_data())
            appliance_power = next(appliance_meter.power_series_all_data())
            
            return mains_meter, appliance_meter, mains_power, appliance_power, dataset
            
        except Exception as e:
            dataset.store.close()
            raise e
    
    def load_simulated_data(
        self, 
        duration: float = 2.0, 
        sampling_rate: int = 16000,
        transient_start: float = 0.5,
        transient_duration: float = 0.2
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate simulated data for testing.
        
        Args:
            duration: Duration of the signal in seconds
            sampling_rate: Sampling rate in Hz
            transient_start: When to start the transient (in seconds)
            transient_duration: Duration of the transient (in seconds)
            
        Returns:
            Tuple of (time_array, grid_voltage, aggregated_current)
        """
        # Time array
        t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)
        
        # Simulate grid voltage (50Hz sine wave)
        grid_voltage = 325 * np.sin(2 * np.pi * 50 * t)
        
        # Simulate background current (fundamental + some harmonics + noise)
        background_current = (
            10 * np.sin(2 * np.pi * 50 * t + 0.2) +  # Fundamental
            2 * np.sin(2 * np.pi * 150 * t) +         # 3rd harmonic
            1 * np.sin(2 * np.pi * 250 * t - 0.1) +   # 5th harmonic
            0.5 * np.random.randn(len(t))              # Noise
        )
        
        # Create a transient signal (kettle-like)
        transient_samples = int(transient_duration * sampling_rate)
        transient_time = np.linspace(0, transient_duration, transient_samples)
        transient_signal = 15 * np.exp(-transient_time * 50) * np.sin(2 * np.pi * 100 * transient_time)
        
        # Add transient to background current
        start_idx = int(transient_start * sampling_rate)
        end_idx = start_idx + transient_samples
        
        # Ensure we don't go out of bounds
        if end_idx > len(background_current):
            transient_signal = transient_signal[:len(background_current) - start_idx]
            end_idx = len(background_current)
        
        aggregated_current = background_current.copy()
        aggregated_current[start_idx:end_idx] += transient_signal
        
        return t, grid_voltage, aggregated_current, start_idx
