# Standard library imports
import sys
import os
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# NILMTK imports
from nilmtk import DataSet
from nilmtk.electric import Electric
from nilmtk.metergroup import MeterGroup
from nilmtk.elecmeter import ElecMeter

# Matplotlib configuration
plt.ion()  # Turn on interactive mode

# Project path configuration - automatically detect nilmtk folder
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
NILMTK_PATH = os.path.join(CURRENT_DIR, 'nilmtk')
PROJECT_ROOT = NILMTK_PATH  # Keep for compatibility
if NILMTK_PATH not in sys.path:
    sys.path.insert(0, NILMTK_PATH)

# Dataset path configuration
DATASET_PATH = PROJECT_ROOT + r"\data\random.h5"

print("sys.path[0] =", sys.path[0])
print("DATASET_PATH =", DATASET_PATH)

# Validate nilmtk detection
print("\n=== NILMTK Detection Validation ===")

# Check if nilmtk folder exists
if os.path.exists(PROJECT_ROOT):
    print(f"✓ nilmtk folder found at: {PROJECT_ROOT}")
else:
    print(f"✗ nilmtk folder NOT found at: {PROJECT_ROOT}")
    exit(1)

# Check if nilmtk module can be imported
try:
    import nilmtk
    print("✓ nilmtk module imported successfully")
    print(f"  nilmtk version: {getattr(nilmtk, '__version__', 'Unknown')}")
except ImportError as e:
    print(f"✗ Failed to import nilmtk: {e}")
    exit(1)
except Exception as e:
    print(f"✗ Error importing nilmtk: {e}")
    exit(1)

# Check if DataSet class is available
try:
    from nilmtk import DataSet
    print("✓ DataSet class imported successfully")
except ImportError as e:
    print(f"✗ Failed to import DataSet: {e}")
    exit(1)

# Check if dataset file exists
if os.path.exists(DATASET_PATH):
    print(f"✓ Dataset file found at: {DATASET_PATH}")
else:
    print(f"✗ Dataset file NOT found at: {DATASET_PATH}")
    print("  Please ensure the data folder and random.h5 file exist")
    exit(1)

print("\n=== All validations passed! ===")

# Try to load the dataset
try:
    ds = DataSet(DATASET_PATH)
    print("✓ Dataset loaded successfully")
    
    # Check if buildings are available
    if hasattr(ds, 'buildings') and len(ds.buildings) > 0:
        print(f"✓ Found {len(ds.buildings)} building(s) in dataset")
        
        # Try to access building 1
        if 1 in ds.buildings:
            elec = ds.buildings[1].elec
            print("✓ Building 1 electricity data accessed successfully")
            print(f"  Electricity object: {elec}")
        else:
            print("✗ Building 1 not found in dataset")
    else:
        print("✗ No buildings found in dataset")
        
except Exception as e:
    print(f"✗ Error loading dataset: {e}")
    exit(1)

print(elec)