#!/usr/bin/env python3
"""
Script to explore and get feature names from NILM datasets.
"""

import nilmtk
import os

def check_nilmtk_features():
    """Check what NILM features and datasets are available."""
    print("🔍 Exploring NILM Dataset Features")
    print("=" * 50)
    
    # Check nilmtk version and capabilities
    print(f"📦 NILMTK Version: {nilmtk.__version__}")
    
    # Show correct API usage
    print("\n💡 Correct NILMTK API Usage:")
    print("  • Access appliances through: elec.appliances")
    print("  • Access submeters through: elec.submeters()")
    print("  • Access meters through: elec.meters")
    print("  • Building has: building.elec (not building.appliances)")
    
    # Check available datasets in current directory
    print(f"\n📁 Current directory: {os.getcwd()}")
    print("📋 Files in current directory:")
    for file in os.listdir('.'):
        if file.endswith(('.h5', '.hdf5', '.csv')):
            print(f"  📊 {file}")
    
    # Check if we can load any datasets
    dataset_files = [f for f in os.listdir('.') if f.endswith(('.h5', '.hdf5'))]
    
    if dataset_files:
        print(f"\n🚀 Found {len(dataset_files)} potential dataset files:")
        for i, file in enumerate(dataset_files, 1):
            print(f"  {i}. {file}")
        
        # Try to load the first dataset file
        try:
            dataset_path = dataset_files[0]
            print(f"\n📖 Attempting to load: {dataset_path}")
            
            dataset = nilmtk.DataSet(dataset_path)
            print(f"✅ Successfully loaded dataset!")
            
            # Get basic info
            print(f"\n🏢 Buildings: {list(dataset.buildings.keys())}")
            
            # Explore first building
            if dataset.buildings:
                building_key = list(dataset.buildings.keys())[0]
                building = dataset.buildings[building_key]
                elec = building.elec
                
                print(f"\n🏠 Building {building_key} Details:")
                print(f"  ⚡ Total meters: {len(elec.meters)}")
                
                # Show meter features
                print(f"\n⚡ Meter Features:")
                for meter_id, meter in elec.meters.items():
                    print(f"    Meter {meter_id}: {meter.label()}")
                
                # Appliance details - FIXED API USAGE
                print(f"\n🏠 Appliances:")
                try:
                    for appliance in elec.appliances:
                        app_type = appliance.metadata.get('type', 'Unknown')
                        print(f"    {appliance.identifier}: {app_type}")
                except AttributeError:
                    print("    ⚠️  elec.appliances not available, trying submeters()...")
                    for meter in elec.submeters():
                        print(f"    {meter.identifier}: {meter.label()}")
                
                # Alternative way to get appliances
                print(f"\n🔌 Appliance Meters (submeters):")
                for meter in elec.submeters():
                    print(f"    {meter.identifier}: {meter.label()}")
                
                # Try to get sample data features
                try:
                    sample_meter = next(iter(elec.meters.values()))
                    print(f"\n📊 Sample Data Features:")
                    print(f"  Available AC types: {sample_meter.available_ac_types()}")
                    print(f"  Sample period: {sample_meter.sample_period()}")
                except Exception as e:
                    print(f"  Could not get sample data features: {e}")
                    
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
    else:
        print("\n❌ No dataset files (.h5, .hdf5) found in current directory.")
        print("💡 Place your NILM dataset files in this directory to explore them.")
    
    print(f"\n🎯 Common NILM dataset features you can expect:")
    print("  • Power measurements (active, reactive, apparent)")
    print("  • Voltage and current")
    print("  • Timestamp data")
    print("  • Appliance labels and metadata")
    print("  • Building and meter information")
    
    print(f"\n🔧 How to Fix AttributeError:")
    print("  ❌ Wrong: building.appliances.keys()")
    print("  ✅ Correct: building.elec.appliances (or elec.submeters())")
    print("  ✅ Correct: list(building.elec.appliances)")
    print("  ✅ Correct: [meter.label() for meter in building.elec.submeters()]")

if __name__ == "__main__":
    check_nilmtk_features()
