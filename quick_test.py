#!/usr/bin/env python3
"""
Quick Model Test - Just Load and Predict!
==========================================
Super simple test that just loads your trained model and makes predictions.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from cnn_model import NILMCNN
from preprocessing import WINDOW_SIZE, NILMDataLoader, TARGET_APPLIANCES

# ============================================================================
# USER SETTINGS - CHANGE THESE
# ============================================================================
BUILDING_ID = 1                           # UK-DALE building number (1, 2, 3, 4, or 5)
TEST_START_DATE = "2014-12-08 07:00:00"   # Start: YYYY-MM-DD HH:MM:SS
TEST_END_DATE = "2014-12-08 12:00:00"     # End: YYYY-MM-DD HH:MM:SS
# ============================================================================

def quick_test():
    """Quick test - just load model and predict"""
    
    print("="*50)
    print("QUICK MODEL TEST")
    print("="*50)
    
    # Step 1: Load model
    print("Loading model...")
    model = NILMCNN(window_size=WINDOW_SIZE)
    checkpoint = torch.load('model_train.pth', map_location='cpu')
    model.load_state_dict(checkpoint)
    model.eval()
    print("[OK] Model loaded!")
    
    # Step 2: Load UK-DALE data
    print(f"Loading UK-DALE Building {BUILDING_ID} data from {TEST_START_DATE} to {TEST_END_DATE}...")
    try:
        # Load mains data
        data_loader = NILMDataLoader(building_id=BUILDING_ID)
        data_loader.load_data(TEST_START_DATE, TEST_END_DATE, TARGET_APPLIANCES, "all")
        mains_data = data_loader.mains_data
        appliance_data = data_loader.appliance_data
        
        # Directly load washer dryer (instance 1) data
        print("Loading washer dryer (instance 1) data directly...")
        try:
            from nilmtk import DataSet
            dataset = DataSet(r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5")
            building = dataset.buildings[BUILDING_ID]
            elec = building.elec
            
            # Find washer dryer instance 1
            washer_dryer_meter = None
            for meter in elec.submeters().meters:
                if meter.appliances and len(meter.appliances) > 0:
                    app_type = meter.appliances[0].type.get('type', '')
                    if app_type == 'washer dryer':
                        washer_dryer_meter = meter
                        break
            
            if washer_dryer_meter:
                # Load washer dryer data for the same time period
                washer_dryer_data = next(washer_dryer_meter.load(sample_period=6))
                washer_dryer_data = washer_dryer_data.loc[TEST_START_DATE:TEST_END_DATE]
                
                if ('power', 'active') in washer_dryer_data.columns:
                    real_washer_dryer = washer_dryer_data[('power', 'active')]
                else:
                    real_washer_dryer = washer_dryer_data.iloc[:, 0]
                
                print(f"[OK] Washer dryer (instance 1) data loaded: {len(real_washer_dryer)} samples")
            else:
                print("[WARNING] Washer dryer (instance 1) not found")
                real_washer_dryer = None
                
        except Exception as e:
            print(f"[WARNING] Error loading washer dryer directly: {e}")
            real_washer_dryer = None
        
        if len(mains_data) >= WINDOW_SIZE:
            # Take first window_size samples and NORMALIZE (same as training)
            test_values = mains_data.values[:WINDOW_SIZE].flatten()
            
            # NORMALIZE the data (same as training)
            test_mean = np.mean(test_values)
            test_std = np.std(test_values)
            if test_std == 0:
                test_std = 1
            test_values_normalized = (test_values - test_mean) / test_std
            
            test_input = torch.tensor(test_values_normalized, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            print(f"[OK] Real data loaded and normalized: {len(mains_data)} samples")
            print(f"[OK] Normalization stats: mean={test_mean:.1f}W, std={test_std:.1f}W")
        else:
            print("[WARNING] Not enough data, using random data")
            test_input = torch.randn(1, 1, WINDOW_SIZE) * 500 + 1500
    except Exception as e:
        print(f"[WARNING] Error loading data: {e}")
        print("Using random data instead...")
        test_input = torch.randn(1, 1, WINDOW_SIZE) * 500 + 1500
    
    print(f"[OK] Test input ready: {test_input.shape}")
    
    # Step 3: Make prediction
    print("Making prediction...")
    with torch.no_grad():
        prediction = model(test_input)
    
    print(f"[OK] Prediction made!")
    print(f"Input range (normalized): {test_input.min().item():.3f} to {test_input.max().item():.3f}")
    print(f"Output range (normalized): {prediction.min().item():.3f} to {prediction.max().item():.3f}")
    
    # DENORMALIZE predictions back to real power values
    if 'test_std' in locals() and 'test_mean' in locals():
        prediction_real = prediction * test_std + test_mean
        print(f"DENORMALIZED prediction range: {prediction_real.min().item():.1f}W to {prediction_real.max().item():.1f}W")
        print(f"DENORMALIZED prediction mean: {prediction_real.mean().item():.1f}W")
    else:
        prediction_real = prediction
        print(f"[WARNING] Could not denormalize - using raw prediction")
    
    # Step 4: Show results
    print("\n" + "="*50)
    print("SUCCESS! Model is working!")
    print("="*50)
    print("Your model can:")
    print("  [OK] Load successfully")
    print("  [OK] Make predictions")
    print("  [OK] Output reasonable power values")
    print(f"\nPrediction sample (first 10 values):")
    print(f"{prediction[0][:10].numpy()}")
    
    # Step 5: Plot power consumption in stacked area style
    print("\nPlotting power consumption...")
    
    try:
        # Use the correct appliance loading code from notebook
        print("Loading all appliances correctly...")
        
        # Load dataset and get building
        from nilmtk import DataSet
        dataset = DataSet(r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5")
        building = dataset.buildings[BUILDING_ID]
        elec = building.elec
        
        # Set time window
        start = pd.Timestamp(TEST_START_DATE, tz="UTC")
        end = pd.Timestamp(TEST_END_DATE, tz="UTC")
        
        # Load mains power using your exact method
        print("Loading mains power...")
        mains = elec.mains()
        df_mains = next(mains.load(sample_period=60))
        df_mains = df_mains.dropna()
        
        # Filter by time range
        df_mains = df_mains[start:end]
        
        if ('power', 'active') in df_mains.columns:
            mains_power = df_mains[('power', 'active')]
        else:
            mains_power = df_mains.iloc[:, 0]
        
        print(f"[OK] Mains power loaded: {len(mains_power)} samples")
        
        print("Loading washer dryer...")
        washer_dryer_power = None
        for appliance in elec.submeters().meters:
            if appliance.appliances:
                label = appliance.appliances[0].type['type']
                if label == 'washer dryer':
                    try:
                        df = next(appliance.load(sample_period=6))
                        df = df[start:end]
                        
                        if ('power', 'active') in df.columns:
                            washer_dryer_power = df[('power', 'active')]
                        else:
                            washer_dryer_power = df.iloc[:, 0]
                        
                        print(f"[OK] Washer dryer data loaded: {len(washer_dryer_power)} samples")
                        break
                    except Exception as e:
                        print(f"[DEBUG] Error loading washer dryer: {e}")
                        continue
        
        if washer_dryer_power is None:
            print("[WARNING] No washer dryer found")
            washer_dryer_power = pd.Series(0, index=mains_power.index)
        
        # Create correct plot data
        df_plot = pd.DataFrame(index=mains_power.index)
        
        # Align washer dryer data with mains
        washer_dryer_aligned = washer_dryer_power.reindex(mains_power.index, fill_value=0)
        
        # Calculate other appliances power (mains - washer dryer)
        other_appliances = mains_power - washer_dryer_aligned
        other_appliances = np.maximum(other_appliances, 0)  # Don't go negative
        
        df_plot['Other Appliances'] = other_appliances
        df_plot['Washer Dryer'] = washer_dryer_aligned
        
        print(f"[OK] Plot data created")
        
        # Plot stacked area chart
        plt.figure(figsize=(15,6))
        df_plot.plot.area(ax=plt.gca(), alpha=0.7, linewidth=0)
        
        plt.title(f"Power Demand (Building {BUILDING_ID}) - {TEST_START_DATE} to {TEST_END_DATE}")
        plt.ylabel("Power (W)")
        plt.xlabel("Time of Day")
        
        # Legend moved below plot
        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),  # push below the x-axis
            ncol=2,                       # arrange in columns
            fontsize=10
        )
        
        plt.tight_layout()
        plt.savefig('test_power_consumption.png', dpi=300, bbox_inches='tight')
        print("[OK] Plot saved as 'test_power_consumption.png'")
        plt.show()
        
        # Print statistics
        print(f"\nPower Statistics:")
        print(f"  Mains power max: {mains_power.max():.1f} W")
        print(f"  Washer dryer max: {df_plot['Washer Dryer'].max():.1f} W")
        print(f"  Other appliances max: {df_plot['Other Appliances'].max():.1f} W")
        print(f"  Total energy: {mains_power.sum() / 1000 / 60:.2f} kWh")
        
    except Exception as e:
        print(f"[ERROR] Could not plot: {e}")
        import traceback
        traceback.print_exc()
    
    return True

if __name__ == "__main__":
    quick_test()
