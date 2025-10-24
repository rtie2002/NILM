from nilmtk import DataSet
import pandas as pd
import numpy as np
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Use data in  house 1,2 and 5 only
house_indicies = [1, 2, 5]

# Load UK-DALE .h5 dataset
ukdale = DataSet(r'C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5')

#Hyperparameters
sample_period = 6;
noise_threshold = 5; #Noise threshold in Watts

# NEW APPROACH: First capture all raw data, then process
print("="*60)
print("STEP 1: CAPTURING ALL RAW DATA")
print("="*60)

# Store all processed data separately for each appliance
all_house_mains_data = []
all_house_kettle_data = []
all_house_microwave_data = []
all_house_fridge_data = []
all_house_dishwasher_data = []
all_house_washing_machine_data = []
raw_mains_data = []
raw_kettle_data = []
raw_microwave_data = []
raw_fridge_data = []
raw_dishwasher_data = []
raw_washing_machine_data = []

# Standardized appliance names (final output)
appliance_name2 = ['kettle', 'microwave', 'fridge', 'dishwasher', 'washing_machine']

# Mapping from original names to standardized names
appliance_mapping = {
    'fridge freezer': 'fridge',
    'fridge': 'fridge',
    'dish washer': 'dishwasher', 
    'dishwasher': 'dishwasher',
    'washer dryer': 'washing_machine',
    'washing machine': 'washing_machine',
    'washing_machine': 'washing_machine'
}

# Original appliance names per building (for loading)
appliance_name = [['kettle', 'microwave', 'fridge freezer', 'dish washer', 'washer dryer'],
                   ['kettle', 'microwave', 'fridge', 'dish washer', 'washing machine'],
                   ['kettle', 'microwave', 'fridge freezer', 'dish washer', 'washer dryer']]

#Load Power in the selected houses
for idx, house_id in enumerate(house_indicies):
    print("************************************************")
    print(f"Loading RAW data for Building {house_id}")

    #Get electricity data for that house
    elec = ukdale.buildings[house_id].elec

    #Load mains(aggregated power) - RAW DATA
    mains = elec.mains()
    df_mains_raw = next(mains.load(sample_period=sample_period))
    df_mains_raw = df_mains_raw['power']['active'].to_frame(name='P_mains')
    raw_mains_data.append(df_mains_raw)
    print(f"Raw mains data: {len(df_mains_raw)} samples")

    #Load each appliance - RAW DATA
    for app in appliance_name[idx]:
        std_app = appliance_mapping.get(app, app)
        
        if std_app in appliance_name2:
            #Check if appliance exists in the building
            appliance_found = False
            for appliances in elec.appliances:
                if appliances.identifier.type == app:
                    appliance_found = True
                    break

            if appliance_found:
                #Load appliance data - RAW
                df_app_raw = next(elec[app].load(sample_period=sample_period))
                df_app_raw = df_app_raw['power']['active'].to_frame(name=std_app)
                
                # Store raw data
                if std_app == 'kettle':
                    raw_kettle_data.append(df_app_raw)
                elif std_app == 'microwave':
                    raw_microwave_data.append(df_app_raw)
                elif std_app == 'fridge':
                    raw_fridge_data.append(df_app_raw)
                elif std_app == 'dishwasher':
                    raw_dishwasher_data.append(df_app_raw)
                elif std_app == 'washing_machine':
                    raw_washing_machine_data.append(df_app_raw)
                
                print(f"Raw {app} -> {std_app} data: {len(df_app_raw)} samples")
            else:
                print(f"  ❌ {app} -> {std_app} not found")



print("\n" + "="*60)
print("STEP 2: PROCESSING AND RESAMPLING")
print("="*60)

# Now process each building's data with resampling
for idx, house_id in enumerate(house_indicies):
    print(f"\nProcessing Building {house_id}...")
    
    # Get raw data for this building
    df_mains_raw = raw_mains_data[idx]
    
    # Process each appliance separately - resample mains with each appliance
    for std_app in appliance_name2:
        app_data_list = []
        if std_app == 'kettle' and idx < len(raw_kettle_data):
            app_data_list = raw_kettle_data
        elif std_app == 'microwave' and idx < len(raw_microwave_data):
            app_data_list = raw_microwave_data
        elif std_app == 'fridge' and idx < len(raw_fridge_data):
            app_data_list = raw_fridge_data
        elif std_app == 'dishwasher' and idx < len(raw_dishwasher_data):
            app_data_list = raw_dishwasher_data
        elif std_app == 'washing_machine' and idx < len(raw_washing_machine_data):
            app_data_list = raw_washing_machine_data
        
        if app_data_list and idx < len(app_data_list):
            df_app_raw = app_data_list[idx]
            
            # Find common time range between mains and this appliance
            mains_start = df_mains_raw.index.min()
            mains_end = df_mains_raw.index.max()
            app_start = df_app_raw.index.min()
            app_end = df_app_raw.index.max()
            
            # Get overlapping time range
            start_time = max(mains_start, app_start)
            end_time = min(mains_end, app_end)
            
            if start_time < end_time:
                # Resample mains for this appliance's time range
                mains_for_app = df_mains_raw.loc[start_time:end_time]['P_mains'].resample(f'{sample_period}S').mean().ffill(limit=30)
                mains_for_app = mains_for_app.dropna()
                mains_for_app = mains_for_app[mains_for_app > 0]  # Remove negative values
                mains_for_app[mains_for_app < noise_threshold] = 0  # Set noise to 0
                
                # Resample appliance for the same time range
                app_for_mains = df_app_raw.loc[start_time:end_time][std_app].resample(f'{sample_period}S').mean().ffill(limit=30)
                app_for_mains = app_for_mains.dropna()
                
                # Align to common timeline
                common_times = mains_for_app.index.intersection(app_for_mains.index)
                if len(common_times) > 0:
                    mains_aligned = mains_for_app.loc[common_times]
                    app_aligned = app_for_mains.loc[common_times]
                    
                    # Build combined array: time index + mains + appliance
                    combined = pd.DataFrame({
                        'P_mains': mains_aligned,
                        std_app: app_aligned
                    }, index=common_times)  # time is the index
                    
                    # Store in appropriate list
                    if std_app == 'kettle':
                        all_house_kettle_data.append(combined)
                    elif std_app == 'microwave':
                        all_house_microwave_data.append(combined)
                    elif std_app == 'fridge':
                        all_house_fridge_data.append(combined)
                    elif std_app == 'dishwasher':
                        all_house_dishwasher_data.append(combined)
                    elif std_app == 'washing_machine':
                        all_house_washing_machine_data.append(combined)
                    
                    print(f"  📊 Processed {std_app}: {len(combined)} samples")
                else:
                    print(f"  ❌ No common time range for {std_app}")
            else:
                print(f"  ❌ No overlapping time range for {std_app}")
        else:
            print(f"  ❌ No data for {std_app}")
    
    # Store mains data (full timeline)
    mains_full = df_mains_raw['P_mains'].resample(f'{sample_period}S').mean().ffill(limit=30)
    mains_full = mains_full.dropna()
    mains_full = mains_full[mains_full > 0]
    mains_full[mains_full < noise_threshold] = 0
    all_house_mains_data.append(mains_full.to_frame('P_mains'))

print("\n" + "="*60)
print("STEP 3: CONCATENATE ALL DATA")
print("="*60)

# Concatenate all data separately
if all_house_mains_data:
    entire_data_mains = pd.concat(all_house_mains_data, ignore_index=True)
    print(f"✅ Mains data: {entire_data_mains.shape}")

if all_house_kettle_data:
    entire_data_kettle = pd.concat(all_house_kettle_data, ignore_index=True)
    print(f"✅ Kettle data: {entire_data_kettle.shape}")

if all_house_microwave_data:
    entire_data_microwave = pd.concat(all_house_microwave_data, ignore_index=True)
    print(f"✅ Microwave data: {entire_data_microwave.shape}")

if all_house_fridge_data:
    entire_data_fridge = pd.concat(all_house_fridge_data, ignore_index=True)
    print(f"✅ Fridge data: {entire_data_fridge.shape}")

if all_house_dishwasher_data:
    entire_data_dishwasher = pd.concat(all_house_dishwasher_data, ignore_index=True)
    print(f"✅ Dishwasher data: {entire_data_dishwasher.shape}")

if all_house_washing_machine_data:
    entire_data_washing_machine = pd.concat(all_house_washing_machine_data, ignore_index=True)
    print(f"✅ Washing machine data: {entire_data_washing_machine.shape}")

print("\n✅ All appliance data saved separately!")
print(entire_data_mains)
print(entire_data_kettle)
print(entire_data_microwave)
print(entire_data_fridge)
print(entire_data_dishwasher)
print(entire_data_washing_machine)

print("*********************************************************************") 

print("NaN | Length Count:")
print(f"Mains: {entire_data_mains.isnull().sum().sum()} | {len(entire_data_mains)}")
print(f"Kettle: {entire_data_kettle.isnull().sum().sum()} | {len(entire_data_kettle)}")
print(f"Microwave: {entire_data_microwave.isnull().sum().sum()} | {len(entire_data_microwave)}")
print(f"Fridge: {entire_data_fridge.isnull().sum().sum()} | {len(entire_data_fridge)}")
print(f"Dishwasher: {entire_data_dishwasher.isnull().sum().sum()} | {len(entire_data_dishwasher)}")
print(f"Washing Machine: {entire_data_washing_machine.isnull().sum().sum()} | {len(entire_data_washing_machine)}")

    #****************************************************************************
#                                Normalize the data
#****************************************************************************
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

print("="*60)
print("STEP 4: NORMALIZATION PER DATASET (Z-score)")
print("="*60)

# Collect available datasets
dataset_objs = {
    'mains': locals().get('entire_data_mains', None),
    'kettle': locals().get('entire_data_kettle', None),
    'microwave': locals().get('entire_data_microwave', None),
    'fridge': locals().get('entire_data_fridge', None),
    'dishwasher': locals().get('entire_data_dishwasher', None),
    'washing_machine': locals().get('entire_data_washing_machine', None),
}

# Dictionary to store normalization parameters (mean and std)
normalization_params = {}

for name, df in dataset_objs.items():
    if df is None:
        print(f"❌ Skipping {name}: dataset not found")
        continue
    if not isinstance(df, pd.DataFrame) or df.empty:
        print(f"❌ Skipping {name}: not a valid non-empty DataFrame")
        continue

    print(f"\nProcessing {name} dataset...")
    print(f"Data shape: {df.shape}")
    print(f"Original data stats:")
    print(df.describe())

    # Convert to numpy array for normalization
    X = df.values  # Shape: (samples, features)

    # Z-score normalization (mean=0, std=1)
    scaler_z = StandardScaler()
    X_norm_z = scaler_z.fit_transform(X)

    # Convert back to DataFrame
    df_norm = pd.DataFrame(X_norm_z, 
                           index=df.index, 
                           columns=df.columns)

    print(f"\nZ-score normalized data stats:")
    print(df_norm.describe())

    # Save normalization parameters for later use
    normalization_params[name] = {
        'mean': scaler_z.mean_,
        'std': scaler_z.scale_
    }

    print(f"\nNormalization parameters for {name}:")
    print(f"Mean: {normalization_params[name]['mean']}")
    print(f"Std: {normalization_params[name]['std']}")

    # Verify normalization worked
    print(f"\nVerification for {name}:")
    print(f"Normalized data mean: {df_norm.mean().mean():.6f} (should be ~0)")
    print(f"Normalized data std: {df_norm.std().mean():.6f} (should be ~1)")

    # Store normalized dataset in memory (no saving)
    var_norm_name = f"entire_data_{name}_norm"
    globals()[var_norm_name] = df_norm
    print(f"✅ {name}: Normalized and stored in memory")

print("\n" + "="*60)
print("NORMALIZATION COMPLETE - READY FOR NEXT STEP")
print("="*60)
print("✅ All datasets normalized and stored in memory")
print("✅ No .npy files saved - in-memory normalization only")
print("✅ Normalization parameters accessible via normalization_params['<dataset_name>']")
#****************************************************************************
#               Create sequences for each appliance using normalized data
#****************************************************************************
import numpy as np

chunk_size = 240
window_length = 480

print("="*60)
print("STEP 5: CREATE TIME WINDOWS FROM NORMALIZED DATA")
print("="*60)

# List of normalized datasets to process
normalized_datasets = {
    'mains': 'entire_data_mains_norm',
    'kettle': 'entire_data_kettle_norm',
    'microwave': 'entire_data_microwave_norm',
    'fridge': 'entire_data_fridge_norm',
    'dishwasher': 'entire_data_dishwasher_norm',
    'washing_machine': 'entire_data_washing_machine_norm'
}

# Dictionary to store generated sequences in memory
sequences_data = {}

for std_app, dataset_name in normalized_datasets.items():
    if dataset_name in locals():
        data = locals()[dataset_name]
        print(f"\nProcessing {std_app} sequences...")
        print(f"Dataset shape: {data.shape}")
        
        # Convert to numpy
        data_array = data.values
        
        # Create overlapping windows
        sequences = []
        for i in range(0, len(data_array) - window_length, chunk_size):
            window = data_array[i:i + window_length]
            sequences.append(window)
        
        if len(sequences) > 0:
            # Convert to 3D array
            sequences_3d = np.stack(sequences)
            print(f"✅ {std_app} sequences: {sequences_3d.shape}")
            print(f"   - Number of windows: {sequences_3d.shape[0]}")
            print(f"   - Window length: {sequences_3d.shape[1]}")
            print(f"   - Features per timestep: {sequences_3d.shape[2]}")
            
            # Store in memory instead of saving
            sequences_data[std_app] = sequences_3d
            print(f"   - Stored in memory as: sequences_data['{std_app}']")
        else:
            print(f"❌ {std_app}: Not enough data for windowing (need {window_length} samples)")
    else:
        print(f"❌ {std_app}: Dataset {dataset_name} not found")

print("\n" + "="*60)
print("SEQUENCE CREATION COMPLETE - READY FOR NEXT STEP")
print("="*60)
print("✅ All sequences stored in memory")
print("✅ Access them via: sequences_data['<appliance_name>']")
print("✅ No .npy files saved - data kept in RAM for next stage")
#****************************************************************************
#                        Shuffle sequences for each appliance
#****************************************************************************
import numpy as np

print("="*60)
print("STEP 6: SHUFFLE SEQUENCES (IN-MEMORY)")
print("="*60)

# Set random seed for reproducibility
np.random.seed(42)

# Ensure sequences_data exists
if 'sequences_data' not in locals():
    print("❌ sequences_data not found in memory. Please run Step 5 first.")
else:
    shuffled_sequences = {}

    for std_app in normalized_datasets.keys():
        if std_app in sequences_data:
            sequences = sequences_data[std_app]
            print(f"📊 {std_app}: {sequences.shape} sequences in memory")

            # Shuffle sequences
            np.random.shuffle(sequences)

            # Store shuffled sequences in a new dictionary
            shuffled_sequences[std_app] = sequences
            print(f"✅ {std_app}: Sequences shuffled and stored in memory")
        else:
            print(f"❌ {std_app}: No sequence data found in memory")

    print("\n" + "="*60)
    print("SHUFFLING COMPLETE - READY FOR NEXT STEP")
    print("="*60)
    print("✅ All sequences shuffled and stored in memory")
    print("✅ Access shuffled data via: shuffled_sequences['<appliance_name>']")
    print("✅ No files saved - in-memory workflow only")
    print("Random seed set to 42 for reproducibility.")
#****************************************************************************
#             Split shuffled sequences into train / val / test sets
#****************************************************************************
import numpy as np

print("="*60)
print("STEP 7: SPLIT SEQUENCES INTO TRAIN/VAL/TEST")
print("="*60)

# Split ratios
train_ratio = 0.8
val_ratio = 0.1
test_ratio = 0.1

print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")

# Ensure shuffled_sequences exist
if 'shuffled_sequences' not in locals():
    print("❌ shuffled_sequences not found in memory. Please run Step 6 first.")
else:
    for std_app in normalized_datasets.keys():
        try:
            # Get shuffled sequence data from memory
            if std_app not in shuffled_sequences:
                print(f"❌ {std_app}: No shuffled data found in memory")
                continue

            sequences = shuffled_sequences[std_app]
            print(f"\n📊 {std_app}: {sequences.shape} sequences loaded from memory")

            # Calculate split indices
            n_samples = len(sequences)
            train_end = int(n_samples * train_ratio)
            val_end = train_end + int(n_samples * val_ratio)

            # Split sequences
            train_set = sequences[:train_end]
            val_set = sequences[train_end:val_end]
            test_set = sequences[val_end:]

            # Save splits to .npy files
            np.save(f'train_{std_app}.npy', train_set)
            np.save(f'val_{std_app}.npy', val_set)
            np.save(f'test_{std_app}.npy', test_set)

            # Report split sizes
            print(f"✅ {std_app} splits:")
            print(f"   - Train: {train_set.shape[0]:,} sequences ({train_set.shape[0]/n_samples:.1%})")
            print(f"   - Val:   {val_set.shape[0]:,} sequences ({val_set.shape[0]/n_samples:.1%})")
            print(f"   - Test:  {test_set.shape[0]:,} sequences ({test_set.shape[0]/n_samples:.1%})")

        except Exception as e:
            print(f"❌ {std_app}: Error - {str(e)}")

    print("\n" + "="*60)
    print("SPLITTING COMPLETE")
    print("="*60)
    print("Saved files:")
    for std_app in normalized_datasets.keys():
        print(f"✅ train_{std_app}.npy")
        print(f"✅ val_{std_app}.npy")
        print(f"✅ test_{std_app}.npy")
