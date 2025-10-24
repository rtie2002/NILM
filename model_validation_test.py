#!/usr/bin/env python3
"""
Model Validation Test - Load and Test Trained Model
==================================================
This script loads your trained model and validates it using one day of real UK-DALE data.
It includes proper denormalization and comprehensive validation metrics.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cnn_model import NILMCNN
from preprocessing import NILMDataLoader, TARGET_APPLIANCES, WINDOW_SIZE
from sklearn.metrics import mean_squared_error, mean_absolute_error, f1_score
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# USER SETTINGS - CHANGE THESE
# ============================================================================
BUILDING_ID = 1                           # UK-DALE building number (1, 2, 3, 4, or 5)
TEST_DATE = "2014-12-07"                  # Test date: YYYY-MM-DD (one full day)
MODEL_FILENAME = 'model_train.pth'        # Trained model file
# ============================================================================

# Standard NILM metrics thresholds
on_threshold = {'washer dryer': 20, 'fridge': 50, 'kettle': 2000, 'dish washer': 20, 'washing machine': 20}

def calculate_nilm_metrics(app_name, y_true, y_pred):
    """Calculate comprehensive NILM validation metrics"""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Basic metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    # NILM-specific metrics
    threshold = on_threshold.get(app_name, 10)
    
    # On-state MAE (only when appliance is ON)
    on_mask = y_true > threshold
    if np.sum(on_mask) > 0:
        omae = mean_absolute_error(y_true[on_mask], y_pred[on_mask])
    else:
        omae = 0
    
    # F1 Score for ON/OFF classification
    y_true_binary = (y_true > threshold).astype(int)
    y_pred_binary = (y_pred > threshold).astype(int)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    # Normalized Disaggregation Error (NDE)
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum(y_true ** 2)
    nde = numerator / denominator if denominator > 0 else 0
    
    # Energy Accuracy
    energy_acc = np.sum(y_true * y_pred) / np.sum(y_true ** 2) if np.sum(y_true ** 2) > 0 else 0
    
    # Disaggregation Accuracy
    total_error = np.sum(np.abs(y_true - y_pred))
    total_energy = np.sum(np.abs(y_true))
    disagg_acc = 1 - (total_error / total_energy) if total_energy > 0 else 0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'omae': omae,
        'f1': f1,
        'nde': nde,
        'energy_acc': energy_acc,
        'disagg_acc': disagg_acc,
        'on_samples': np.sum(on_mask),
        'total_samples': len(y_true)
    }

def load_model_and_stats():
    """Load trained model and normalization statistics"""
    print("Loading trained model...")
    
    # Load model
    model = NILMCNN(window_size=WINDOW_SIZE)
    try:
        checkpoint = torch.load(MODEL_FILENAME, map_location='cpu')
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            stats = checkpoint.get('normalization_stats', None)
            print(f"✓ Model loaded with normalization stats")
        else:
            model.load_state_dict(checkpoint)
            stats = None
            print(f"✓ Model loaded (no normalization stats found)")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, None
    
    model.eval()
    return model, stats

def load_test_data(start_date, end_date, target_appliances):
    """Load one day of test data"""
    print(f"Loading test data from {start_date} to {end_date}...")
    
    try:
        # Load data using your preprocessing pipeline
        data_loader = NILMDataLoader(building_id=BUILDING_ID)
        mains_data, appliance_data = data_loader.load_data_period(start_date, end_date, target_appliances)
        
        print(f"✓ Mains data loaded: {len(mains_data)} samples")
        print(f"✓ Appliance data loaded: {list(appliance_data.keys())}")
        
        return mains_data, appliance_data, data_loader
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None, None, None

def create_test_windows(mains_data, appliance_data, window_size, stride=1):
    """Create sliding windows for testing"""
    print(f"Creating test windows (size={window_size}, stride={stride})...")
    
    mains_array = mains_data.values
    appliance_array = appliance_data.values
    
    # Create windows
    windows = []
    for i in range(0, len(mains_array) - window_size + 1, stride):
        mains_window = mains_array[i:i + window_size]
        app_window = appliance_array[i:i + window_size]
        windows.append((mains_window, app_window))
    
    print(f"✓ Created {len(windows)} test windows")
    return windows

def normalize_test_data(windows, stats=None):
    """Normalize test data using training statistics or calculate new ones"""
    print("Normalizing test data...")
    
    if stats is None:
        print("⚠ No training stats available, calculating normalization from test data...")
        # Calculate normalization from test data (not ideal but necessary)
        all_mains = np.concatenate([window[0] for window in windows])
        all_app = np.concatenate([window[1] for window in windows])
        
        mains_mean = np.mean(all_mains)
        mains_std = np.std(all_mains)
        app_mean = np.mean(all_app)
        app_std = np.std(all_app)
    else:
        # Use training statistics
        mains_mean = stats['mains']['mean']
        mains_std = stats['mains']['std']
        app_mean = stats['appliances'][TARGET_APPLIANCES[0]]['mean']
        app_std = stats['appliances'][TARGET_APPLIANCES[0]]['std']
        print(f"✓ Using training normalization stats")
    
    # Normalize windows
    normalized_windows = []
    for mains_window, app_window in windows:
        norm_mains = (mains_window - mains_mean) / mains_std
        norm_app = (app_window - app_mean) / app_std
        normalized_windows.append((norm_mains, norm_app))
    
    print(f"✓ Normalization applied (mains: mean={mains_mean:.1f}, std={mains_std:.1f})")
    print(f"✓ Normalization applied (app: mean={app_mean:.1f}, std={app_std:.1f})")
    
    return normalized_windows, {
        'mains': {'mean': mains_mean, 'std': mains_std},
        'appliance': {'mean': app_mean, 'std': app_std}
    }

def run_validation(model, normalized_windows, normalization_stats):
    """Run validation on test windows"""
    print("Running model validation...")
    
    all_predictions = []
    all_targets = []
    
    model.eval()
    with torch.no_grad():
        for i, (norm_mains, norm_app) in enumerate(normalized_windows):
            # Prepare input tensor
            input_tensor = torch.FloatTensor(norm_mains).unsqueeze(0).unsqueeze(0)  # (1, 1, window_size)
            
            # Make prediction
            prediction = model(input_tensor)
            prediction = prediction.squeeze().numpy()  # Remove batch and channel dimensions
            
            # Store results
            all_predictions.append(prediction)
            all_targets.append(norm_app)
            
            # Show progress
            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{len(normalized_windows)} windows...")
    
    # Flatten all predictions and targets
    all_predictions = np.concatenate(all_predictions)
    all_targets = np.concatenate(all_targets)
    
    print(f"✓ Validation completed: {len(all_predictions)} predictions")
    
    return all_predictions, all_targets

def denormalize_predictions(predictions, targets, normalization_stats):
    """Denormalize predictions and targets back to real power values"""
    print("Denormalizing predictions to real power values...")
    
    app_mean = normalization_stats['appliance']['mean']
    app_std = normalization_stats['appliance']['std']
    
    # Denormalize
    real_predictions = (predictions * app_std) + app_mean
    real_targets = (targets * app_std) + app_mean
    
    print(f"✓ Denormalized to real power values")
    print(f"  Prediction range: {real_predictions.min():.1f}W to {real_predictions.max():.1f}W")
    print(f"  Target range: {real_targets.min():.1f}W to {real_targets.max():.1f}W")
    
    return real_predictions, real_targets

def plot_combined_analysis(mains_data, real_predictions, real_targets, metrics, test_date):
    """Plot combined analysis showing total power, washer dryer real vs predicted"""
    print("Creating combined power analysis plot...")
    
    # Create time index for plotting
    prediction_length = len(real_predictions)
    print(f"Creating time index for {prediction_length} predictions...")
    
    # Create a time index that matches the prediction length
    if len(mains_data) >= prediction_length:
        time_index = mains_data.index[:prediction_length]
    else:
        # Extend the time index if needed
        original_index = mains_data.index
        time_delta = original_index[1] - original_index[0] if len(original_index) > 1 else pd.Timedelta(seconds=6)
        time_index = pd.date_range(start=original_index[0], periods=prediction_length, freq=time_delta)
    
    # Sample data for better visualization
    max_points = 3000  # Maximum points to plot
    if len(real_predictions) > max_points:
        step = len(real_predictions) // max_points
        time_index = time_index[::step]
        real_predictions = real_predictions[::step]
        real_targets = real_targets[::step]
        # Also sample mains data to match
        mains_sampled = mains_data.iloc[::step]
        print(f"Sampled data to {len(real_predictions)} points for better visualization")
    else:
        mains_sampled = mains_data.iloc[:len(real_predictions)]
    
    # Create combined plot
    plt.figure(figsize=(16, 10))
    
    # Plot total power consumption (background)
    plt.plot(time_index, mains_sampled.values, 'g-', label='Total Power Consumption', linewidth=2, alpha=0.6)
    
    # Plot real and predicted washer dryer power (foreground)
    plt.plot(time_index, real_targets, 'b-', label='Real Washer Dryer Power', linewidth=2, alpha=0.9)
    plt.plot(time_index, real_predictions, 'r-', label='Predicted Washer Dryer Power', linewidth=2, alpha=0.9)
    
    plt.title(f'Power Consumption Analysis - {test_date}\nTotal Power vs Washer Dryer Disaggregation', 
              fontsize=16, fontweight='bold')
    plt.xlabel('Time (24-hour format)', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    max_total = max(mains_sampled.values)
    max_app = max(max(real_targets), max(real_predictions))
    plt.ylim(0, max(max_total, max_app) * 1.1)
    
    # Format x-axis to show 24-hour time format
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Show every 4 hours
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour
    plt.xticks(rotation=45)
    
    # Add statistics text boxes
    # Total power stats
    avg_total = np.mean(mains_sampled.values)
    max_total = np.max(mains_sampled.values)
    
    # Washer dryer stats
    avg_wd = np.mean(real_targets)
    max_wd = np.max(real_targets)
    
    plt.text(0.02, 0.98, f'TOTAL POWER:\nAvg: {avg_total:.1f}W\nMax: {max_total:.1f}W', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    plt.text(0.02, 0.75, f'WASHER DRYER:\nMAE: {metrics["mae"]:.1f}W\nRMSE: {metrics["rmse"]:.1f}W\nF1: {metrics["f1"]:.3f}', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plot_filename = f'washer_dryer_analysis_{test_date.replace("-", "_")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Combined analysis plot saved as '{plot_filename}'")
    plt.show()

def plot_validation_results(mains_data, real_predictions, real_targets, metrics, test_date):
    """Simple plot showing real vs predicted washer dryer power consumption"""
    print("Creating simple washer dryer power consumption plot...")
    
    # Create time index for plotting
    prediction_length = len(real_predictions)
    print(f"Creating time index for {prediction_length} predictions...")
    
    # Create a time index that matches the prediction length
    if len(mains_data) >= prediction_length:
        time_index = mains_data.index[:prediction_length]
    else:
        # Extend the time index if needed
        original_index = mains_data.index
        time_delta = original_index[1] - original_index[0] if len(original_index) > 1 else pd.Timedelta(seconds=6)
        time_index = pd.date_range(start=original_index[0], periods=prediction_length, freq=time_delta)
    
    # Sample data for better visualization
    max_points = 3000  # Maximum points to plot
    if len(real_predictions) > max_points:
        step = len(real_predictions) // max_points
        time_index = time_index[::step]
        real_predictions = real_predictions[::step]
        real_targets = real_targets[::step]
        print(f"Sampled data to {len(real_predictions)} points for better visualization")
    
    # Create simple plot
    plt.figure(figsize=(15, 8))
    
    # Plot real and predicted washer dryer power
    plt.plot(time_index, real_targets, 'b-', label='Real Washer Dryer Power', linewidth=1.5, alpha=0.8)
    plt.plot(time_index, real_predictions, 'r-', label='Predicted Washer Dryer Power', linewidth=1.5, alpha=0.8)
    
    plt.title(f'Washer Dryer Power Consumption - {test_date}', fontsize=16, fontweight='bold')
    plt.xlabel('Time (24-hour format)', fontsize=12)
    plt.ylabel('Power (W)', fontsize=12)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0
    plt.ylim(0, max(max(real_targets), max(real_predictions)) * 1.1)
    
    # Format x-axis to show 24-hour time format
    import matplotlib.dates as mdates
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.gca().xaxis.set_major_locator(mdates.HourLocator(interval=4))  # Show every 4 hours
    plt.gca().xaxis.set_minor_locator(mdates.HourLocator(interval=1))  # Minor ticks every hour
    plt.xticks(rotation=45)
    
    # Add simple metrics text
    plt.text(0.02, 0.98, f'MAE: {metrics["mae"]:.1f}W\nRMSE: {metrics["rmse"]:.1f}W', 
             transform=plt.gca().transAxes, fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plot_filename = f'washer_dryer_simple_{test_date.replace("-", "_")}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"✓ Simple washer dryer plot saved as '{plot_filename}'")
    plt.show()

def print_validation_metrics(metrics, test_date):
    """Print comprehensive validation metrics for Washer Dryer"""
    print("\n" + "="*70)
    print(f"WASHER DRYER VALIDATION RESULTS - {test_date}")
    print("="*70)
    
    print(f"📊 BASIC METRICS:")
    print(f"  Mean Absolute Error (MAE): {metrics['mae']:.2f} W")
    print(f"  Root Mean Square Error (RMSE): {metrics['rmse']:.2f} W")
    
    print(f"\n📊 NILM-SPECIFIC METRICS:")
    print(f"  On-state MAE (OMAE): {metrics['omae']:.2f} W")
    print(f"  F1 Score (ON/OFF classification): {metrics['f1']:.3f}")
    print(f"  Normalized Disaggregation Error (NDE): {metrics['nde']:.3f}")
    print(f"  Energy Accuracy: {metrics['energy_acc']:.3f}")
    print(f"  Disaggregation Accuracy: {metrics['disagg_acc']:.3f}")
    
    print(f"\n📊 WASHER DRYER DATA STATISTICS:")
    print(f"  Total samples: {metrics['total_samples']:,}")
    print(f"  ON samples: {metrics['on_samples']:,} ({metrics['on_samples']/metrics['total_samples']*100:.1f}%)")
    print(f"  OFF samples: {metrics['total_samples'] - metrics['on_samples']:,} ({(metrics['total_samples'] - metrics['on_samples'])/metrics['total_samples']*100:.1f}%)")
    
    print(f"\n🎯 INTERPRETATION:")
    if metrics['f1'] > 0.8:
        print(f"  ✅ Excellent ON/OFF detection (F1 > 0.8)")
    elif metrics['f1'] > 0.6:
        print(f"  ⚠️  Good ON/OFF detection (F1 > 0.6)")
    else:
        print(f"  ❌ Poor ON/OFF detection (F1 < 0.6)")
        
    if metrics['mae'] < 50:
        print(f"  ✅ Good power prediction accuracy (MAE < 50W)")
    elif metrics['mae'] < 100:
        print(f"  ⚠️  Moderate power prediction accuracy (MAE < 100W)")
    else:
        print(f"  ❌ Poor power prediction accuracy (MAE > 100W)")
    
    print("\n" + "="*70)

def main():
    """Main validation function"""
    print("="*60)
    print("MODEL VALIDATION TEST")
    print("="*60)
    print(f"Building ID: {BUILDING_ID}")
    print(f"Test Date: {TEST_DATE}")
    print(f"Model File: {MODEL_FILENAME}")
    print(f"Target Appliance: {TARGET_APPLIANCES[0]}")
    print("="*60)
    
    # Step 1: Load model and normalization stats
    model, stats = load_model_and_stats()
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return False
    
    # Step 2: Load test data (one full day)
    start_date = f"{TEST_DATE} 00:00:00"
    end_date = f"{TEST_DATE} 23:59:59"
    
    mains_data, appliance_data, data_loader = load_test_data(start_date, end_date, TARGET_APPLIANCES)
    if mains_data is None:
        print("❌ Failed to load test data. Exiting.")
        return False
    
    # Step 3: Create test windows
    test_windows = create_test_windows(mains_data, appliance_data[TARGET_APPLIANCES[0]], WINDOW_SIZE, stride=10)
    
    # Step 4: Normalize test data
    normalized_windows, normalization_stats = normalize_test_data(test_windows, stats)
    
    # Step 5: Run validation
    predictions, targets = run_validation(model, normalized_windows, normalization_stats)
    
    # Step 6: Denormalize results
    real_predictions, real_targets = denormalize_predictions(predictions, targets, normalization_stats)
    
    # Step 7: Calculate metrics
    print("Calculating validation metrics...")
    metrics = calculate_nilm_metrics(TARGET_APPLIANCES[0], real_targets, real_predictions)
    
    # Step 8: Print results
    print_validation_metrics(metrics, TEST_DATE)
    
    # Step 9: Create plots
    plot_validation_results(mains_data, real_predictions, real_targets, metrics, TEST_DATE)
    
    # Step 10: Create total power consumption plot
    plot_total_power_consumption(mains_data, TEST_DATE)
    
    print("\n🎉 Validation completed successfully!")
    return True

if __name__ == "__main__":
    main()
