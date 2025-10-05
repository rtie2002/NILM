import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
from nilmtk import DataSet
import os
from datetime import datetime
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from typing import Tuple, Dict, Any, Optional, List, Union
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('tvdr_tdws.log'),
        logging.StreamHandler()
    ]
)

class TVDRTDWS:
    """Implementation of the TVDR-TDWS algorithm for NILM."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = {
            'dataset_path': str(Path.home() / "NILM" / "datasets" / "ukdale.h5"),
            'reference_path': str(Path.home() / "NILM" / "datasets" / "reference.h5"),
            'building': 1,
            'appliance': "kettle",
            'sampling_rate': 16000,
            'voltage_freq': 50,
            'voltage_peak': 325,
            'window_cycles': 8,
            'lpf_cutoff': 1000,
            'on_power_threshold': 50,
            'min_off_duration': 60,
            'min_on_duration': 60,
            'start_time': "2013-11-08 00:00:00",
            'end_time': "2013-11-08 23:59:00",
            'max_events': 5,
            'simulate_data': False,
            'output_dir': 'results',
            'use_reference': False  # Whether to use clean reference for validation
        }
        
        if config:
            self.config.update(config)
            
        os.makedirs(self.config['output_dir'], exist_ok=True)
        self.voltage = None
        self.current = None
        self.power = None
        self.reference_signal = None
        self.activation_indices = []
        self.results = []
    
    def load_reference_signal(self):
        """Load clean reference signal for validation."""
        if not self.config['use_reference'] or not os.path.exists(self.config['reference_path']):
            return None
            
        try:
            dataset = DataSet(self.config['reference_path'])
            elec = dataset.buildings[self.config['building']].elec
            appliance = elec[self.config['appliance']]
            
            # Load reference current signal
            current_data = next(appliance.current().load(sample_period=1/self.config['sampling_rate']))
            self.reference_signal = current_data.values.flatten()
            logging.info(f"Loaded reference signal with {len(self.reference_signal)} samples")
            
        except Exception as e:
            logging.warning(f"Could not load reference signal: {e}")
            self.reference_signal = None
    
    def validate_result(self, window_a: np.ndarray, window_b: np.ndarray, 
                       extracted_signal: np.ndarray) -> Dict[str, float]:
        """
        Validate TVDR-TDWS result using multiple metrics.
        
        Args:
            window_a: Background window (before event)
            window_b: Event window
            extracted_signal: Extracted signal from TVDR-TDWS
            
        Returns:
            Dictionary of validation metrics
        """
        metrics = {}
        
        # Calculate expected signal (B - A)
        expected_signal = window_b - window_a
        
        # 1. Correlation with expected signal (should be high)
        min_len = min(len(extracted_signal), len(expected_signal))
        if min_len > 1:
            metrics['correlation_expected'] = float(np.corrcoef(
                extracted_signal[:min_len], 
                expected_signal[:min_len]
            )[0, 1])
        
        # 2. Correlation with background (should be low)
        min_len = min(len(extracted_signal), len(window_a))
        if min_len > 1:
            metrics['correlation_background'] = float(np.corrcoef(
                extracted_signal[:min_len],
                window_a[:min_len]
            )[0, 1])
        
        # 3. Energy ratio (should be close to 1.0)
        if len(extracted_signal) > 0 and np.var(expected_signal) > 0:
            metrics['energy_ratio'] = float(np.var(extracted_signal) / np.var(expected_signal))
        
        # 4. If reference signal is available, compare with it
        if self.reference_signal is not None:
            ref_len = min(len(extracted_signal), len(self.reference_signal))
            if ref_len > 1:
                metrics['correlation_reference'] = float(np.corrcoef(
                    extracted_signal[:ref_len],
                    self.reference_signal[:ref_len]
                )[0, 1])
        
        return metrics
    
    def plot_comparison(self, window_a: np.ndarray, window_b: np.ndarray, 
                       extracted_signal: np.ndarray, event_idx: int, save_path: str = None):
        """
        Plot comparison between signals with proper validation metrics.
        """
        expected_signal = window_b - window_a
        metrics = self.validate_result(window_a, window_b, extracted_signal)
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
        
        # Time vectors (ms)
        t = np.arange(len(extracted_signal)) * 1000 / self.config['sampling_rate']
        
        # Plot 1: Extracted vs Expected
        ax1.plot(t, expected_signal[:len(t)], 'b-', alpha=0.7, label='Expected (B - A)')
        ax1.plot(t, extracted_signal, 'r-', alpha=0.8, label='Extracted')
        ax1.set_ylabel('Current (A)')
        ax1.set_title(f'Event {event_idx + 1}: Extracted vs Expected Signal')
        ax1.legend()
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Extracted vs Background
        ax2.plot(t, window_a[:len(t)], 'g-', alpha=0.7, label='Background (A)')
        ax2.plot(t, extracted_signal, 'r-', alpha=0.8, label='Extracted')
        ax2.set_ylabel('Current (A)')
        ax2.set_title('Extracted vs Background')
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 3: Validation Metrics
        if self.reference_signal is not None:
            ref_len = min(len(extracted_signal), len(self.reference_signal))
            ax3.plot(t, self.reference_signal[:len(t)], 'm-', alpha=0.7, label='Reference')
            ax3.plot(t, extracted_signal, 'r-', alpha=0.5, label='Extracted')
            ax3.set_ylabel('Current (A)')
            ax3.set_title('Extracted vs Reference')
            ax3.legend()
        else:
            # If no reference, show FFT comparison
            fft_extracted = np.abs(np.fft.rfft(extracted_signal))
            fft_expected = np.abs(np.fft.rfft(expected_signal[:len(extracted_signal)]))
            freqs = np.fft.rfftfreq(len(extracted_signal), 1/self.config['sampling_rate'])
            
            ax3.semilogy(freqs, fft_expected, 'b-', alpha=0.7, label='Expected')
            ax3.semilogy(freqs, fft_extracted, 'r-', alpha=0.7, label='Extracted')
            ax3.set_xlabel('Frequency (Hz)')
            ax3.set_ylabel('Magnitude')
            ax3.set_title('Frequency Domain Comparison')
            ax3.legend()
            ax3.grid(True, linestyle='--', alpha=0.7)
        
        # Add metrics as text
        metrics_text = '\n'.join([f'{k}: {v:.4f}' for k, v in metrics.items()])
        fig.text(0.8, 0.9, metrics_text, bbox=dict(facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        return metrics
    
    def process_event(self, voltage: np.ndarray, current: np.ndarray, event_idx: int) -> Optional[Dict[str, Any]]:
        """Process a single event using TVDR-TDWS algorithm."""
        try:
            # [Previous TVDR-TDWS implementation here...]
            # ... (keep the existing implementation)
            
            # After processing, validate the result
            metrics = self.validate_result(
                result['windows']['A'],
                result['windows']['B'],
                result['windows']['filtered']
            )
            
            # Add metrics to result
            result['metrics'] = metrics
            
            # Plot comparison
            if self.config.get('plot_results', True):
                self.plot_comparison(
                    result['windows']['A'],
                    result['windows']['B'],
                    result['windows']['filtered'],
                    len(self.results),
                    os.path.join(self.config['output_dir'], f'event_{len(self.results)+1}_comparison.png')
                )
            
            return result
            
        except Exception as e:
            logging.error(f"Error processing event at index {event_idx}: {e}")
            return None

def main():
    """Main function to run TVDR-TDWS analysis."""
    config = {
        'simulate_data': True,  # Set to False to use real data
        'use_reference': False,  # Set to True if you have reference data
        'output_dir': 'results',
        'plot_results': True
    }
    
    # Initialize processor
    processor = TVDRTDWS(config)
    
    # Load reference signal if available
    if config['use_reference']:
        processor.load_reference_signal()
    
    # Load or generate data
    if config['simulate_data']:
        # Generate simulated data
        duration = 10  # seconds
        n_samples = int(duration * config['sampling_rate'])
        t = np.linspace(0, duration, n_samples, endpoint=False)
        
        # Generate clean voltage signal with noise
        voltage = config['voltage_peak'] * np.sin(2 * np.pi * config['voltage_freq'] * t)
        voltage += np.random.normal(0, 1, n_samples)
        
        # Generate current with transients
        current = np.random.normal(0, 0.1, n_samples)
        
        # Add appliance activation
        event_idx = n_samples // 3
        transient = np.exp(-np.linspace(0, 10, 1000)) * np.sin(2 * np.pi * 100 * np.linspace(0, 1, 1000))
        current[event_idx:event_idx+len(transient)] += transient * 5
        
        # Store data
        processor.voltage = voltage
        processor.current = current
        processor.activation_indices = [event_idx]
    else:
        # Load real data
        # [Add your data loading code here]
        pass
    
    # Process events
    for i, idx in enumerate(processor.activation_indices):
        if i >= processor.config['max_events']:
            break
            
        result = processor.process_event(
            processor.voltage,
            processor.current,
            idx
        )
        
        if result:
            processor.results.append(result)
            print(f"Processed event {i+1} with metrics:")
            for k, v in result.get('metrics', {}).items():
                print(f"  {k}: {v:.4f}")
    
    return processor

if __name__ == "__main__":
    processor = main()
