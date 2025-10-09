# NILM Toolkit Functions Guide for AI Development

This comprehensive guide provides an overview of reusable functions across all NILM toolkits in the `./toolkit/` directory. Use this as a reference for AI systems to leverage existing functionality when developing NILM solutions.

## Table of Contents
1. [Core Data Management](#core-data-management)
2. [Data Loading & Preprocessing](#data-loading--preprocessing)
3. [Disaggregation Algorithms](#disaggregation-algorithms)
4. [Neural Network Models](#neural-network-models)
5. [Evaluation & Metrics](#evaluation--metrics)
6. [Utilities & Helpers](#utilities--helpers)
7. [Visualization & Reporting](#visualization--reporting)
8. [Experiment Management](#experiment-management)

---

## Core Data Management

### NILMTK Core Classes
**Location**: `./toolkit/nilmtk/nilmtk/`

```python
# Essential data structures
from nilmtk import DataSet, DataStore, HDFDataStore, CSVDataStore
from nilmtk import MeterGroup, ElecMeter, Appliance, Building
from nilmtk import TimeFrame, TimeFrameGroup

# Usage examples:
dataset = DataSet('path/to/dataset.h5')  # Load NILM datasets
building = dataset.buildings[1]          # Access building data
mains = building.elec.mains()            # Get mains power data
appliances = building.elec.submeters()   # Get appliance meters
```

### Data Storage Functions
**Location**: `./toolkit/nilmtk/nilmtk/utils.py`

```python
from nilmtk.utils import get_datastore, normalise_timestamp, convert_to_timestamp

# Data store operations
datastore = get_datastore('data.h5', format='HDF', mode='r')
timestamp = convert_to_timestamp('2023-01-01 00:00:00')
normalized = normalise_timestamp(timestamp, freq='6T')
```

---

## Data Loading & Preprocessing

### NILM-Analyzer Loaders
**Location**: `./toolkit/nilm_analyzer/loaders.py`

```python
from nilm_analyzer import REFIT_Loader, UKDALE_Loader, AMPDS_Loader, IAWE_Loader

# Dataset-specific loaders
refit_data = REFIT_Loader().load_data(building=1)
ukdale_data = UKDALE_Loader().load_data(building=1)
ampds_data = AMPDS_Loader().load_data(building=1)
iawe_data = IAWE_Loader().load_data(building=1)
```

### Preprocessing Functions
**Location**: `./toolkit/nilmtk-contrib/nilmtk_contrib/torch/preprocessing.py`

```python
from nilmtk_contrib.torch.preprocessing import preprocess

# Standard preprocessing pipeline
processed_data = preprocess(
    sequence_length=512,
    mains_mean=mean_value,
    mains_std=std_value,
    mains_lst=mains_data,
    submeters_lst=appliance_data,
    method="train",  # or "test"
    appliance_params=appliance_info
)
```

### Data Validation
**Location**: `./toolkit/nilm_analyzer/modules/validations.py`

```python
from nilm_analyzer.modules.validations import (
    check_house_availability,
    check_correct_datatype,
    check_list_validations
)

# Validation functions
is_available = check_house_availability('building', 1, dataset)
is_valid_type = check_correct_datatype('data', data_array, np.ndarray)
is_valid_list = check_list_validations('appliances', appliance_list, str)
```

---

## Disaggregation Algorithms

### NILMTK Classic Algorithms
**Location**: `./toolkit/nilmtk/nilmtk/disaggregate/`

```python
from nilmtk.disaggregate import FHMMExact, CO, Mean, Hart85

# Factorial Hidden Markov Model
fhmm = FHMMExact()
fhmm.train(mains, submeters)
predictions = fhmm.disaggregate(test_mains)

# Combinatorial Optimization
co = CO()
co.train(mains, submeters)
predictions = co.disaggregate(test_mains)

# Mean baseline
mean_model = Mean()
mean_model.train(mains, submeters)
predictions = mean_model.disaggregate(test_mains)
```

### NILMTK-Contrib Deep Learning Models
**Location**: `./toolkit/nilmtk-contrib/nilmtk_contrib/torch/`

```python
from nilmtk_contrib.torch import (
    Seq2PointTorch, RNN, ResNet, 
    RNN_attention, ResNet_classification
)

# Sequence-to-Point model
seq2point = Seq2PointTorch()
seq2point.train(train_mains, train_appliances)
predictions = seq2point.disaggregate(test_mains)

# Recurrent Neural Network
rnn = RNN()
rnn.train(train_mains, train_appliances)
predictions = rnn.disaggregate(test_mains)

# ResNet model
resnet = ResNet()
resnet.train(train_mains, train_appliances)
predictions = resnet.disaggregate(test_mains)
```

### Torch-NILM Neural Networks
**Location**: `./toolkit/torch-nilm/neural_networks/`

```python
from torch_nilm.neural_networks import VIBNet, VIB_SAED, VIB_SimpleGru, VIBWGRU

# Variational Information Bottleneck networks
vib_net = VIBNet()
vib_net.train(train_data)
predictions = vib_net.predict(test_data)

# Self-Attention Encoder-Decoder with VIB
saed_vib = VIB_SAED()
saed_vib.train(train_data)
predictions = saed_vib.predict(test_data)
```

### Deep-NILMTK Models
**Location**: `./toolkit/deep-nilmtk-v1/deep_nilmtk/models/`

```python
from deep_nilmtk.models.tensorflow.seq2point import Seq2Point

# TensorFlow-based Seq2Point
tf_seq2point = Seq2Point()
tf_seq2point.train(train_data)
predictions = tf_seq2point.predict(test_data)
```

---

## Neural Network Models

### Base Model Interface
**Location**: `./toolkit/torch-nilm/neural_networks/base_models.py`

```python
from torch_nilm.neural_networks.base_models import BaseModel

class CustomNILMModel(BaseModel):
    def __init__(self):
        super().__init__()
        # Define your model architecture
    
    def supports_classic_training(self) -> bool:
        return True
    
    def supports_vib(self) -> bool:
        return False  # Set to True if VIB supported
```

### Model Support Detection
```python
# Check model capabilities
model = YourNILMModel()
if model.supports_vib():
    # Use VIB training
if model.supports_bayes():
    # Use Bayesian inference
if model.supports_bert():
    # Use BERT-style training
```

---

## Evaluation & Metrics

### NILM Metrics
**Location**: `./toolkit/torch-nilm/utils/nilm_metrics.py`

```python
from torch_nilm.utils.nilm_metrics import NILMmetrics

# Comprehensive NILM evaluation
metrics = NILMmetrics(
    pred=predictions,
    ground=ground_truth,
    threshold=40,  # Power threshold in watts
    rounding_digit=3
)

# Returns: MAE, RMSE, F1-Score, Precision, Recall, etc.
```

### RMSE Computation
**Location**: `./toolkit/nilmtk/nilmtk/utils.py`

```python
from nilmtk.utils import compute_rmse

rmse_value = compute_rmse(
    ground_truth=actual_power,
    predictions=predicted_power,
    pretty=True  # Format output nicely
)
```

---

## Utilities & Helpers

### Time Series Utilities
**Location**: `./toolkit/nilmtk/nilmtk/utils.py`

```python
from nilmtk.utils import (
    timedelta64_to_secs, find_nearest, 
    convert_to_timestamp, get_index,
    flatten_2d_list, most_common
)

# Time operations
seconds = timedelta64_to_secs(timedelta)
nearest_idx = find_nearest(known_array, test_value)
timestamp = convert_to_timestamp('2023-01-01')

# Data manipulation
flat_list = flatten_2d_list(nested_list)
common_item = most_common(item_list)
```

### Torch-NILM Helpers
**Location**: `./toolkit/torch-nilm/utils/helpers.py`

```python
from torch_nilm.utils.helpers import (
    create_tree_dir, get_tree_paths, create_timeframes,
    destandardize, denormalize, pd_mean, pd_std
)

# Directory management
tree_levels = {'experiment': 'exp1', 'model': 'lstm'}
create_tree_dir(tree_levels, clean=True, plots=True)

# Data normalization
original_data = destandardize(normalized_data, means, stds)
original_data = denormalize(normalized_data, max_value)

# Pandas operations
mean_data = pd_mean(dataframe)
std_data = pd_std(dataframe)
```

### NILM-Analyzer Utilities
**Location**: `./toolkit/nilm_analyzer/utilities.py`

```python
from nilm_analyzer.utilities import (
    convert_timestamps2minutes, convert_object2timestamps,
    get_module_directory
)

# Timestamp conversions
minutes = convert_timestamps2minutes(timestamp_series)
timestamps = convert_object2timestamps(dataframe, unit_value='s')
module_dir = get_module_directory()
```

---

## Visualization & Reporting

### Torch-NILM Plotting
**Location**: `./toolkit/torch-nilm/utils/plotting.py`

```python
from torch_nilm.utils.plotting import (
    plot_radar_chart, plot_dataframe, plot_results_from_report
)

# Radar chart for multi-metric visualization
plot_radar_chart(
    data=metrics_dataframe,
    num_columns=['mae', 'rmse', 'f1_score'],
    plot_title='Model Performance Comparison'
)

# Dataframe plotting
plot_dataframe(
    data=results_df,
    metrics=['mae', 'rmse'],
    appliances=['fridge', 'washing_machine']
)
```

### Reporting System
**Location**: `./toolkit/torch-nilm/utils/nilm_reporting.py`

```python
from torch_nilm.utils.nilm_reporting import (
    get_statistical_report, get_final_report, save_appliance_report
)

# Generate statistical report
report = get_statistical_report(
    save_name='experiment_results',
    data=results_dataframe
)

# Final comprehensive report
final_report = get_final_report(
    tree_levels=experiment_config,
    save=True,
    root_dir='./results'
)
```

---

## Experiment Management

### Deep-NILMTK Experiment Framework
**Location**: `./toolkit/deep-nilmtk-v1/deep_nilmtk/trainers/`

```python
from deep_nilmtk.trainers import Trainer, TorchTrainer, KerasTrainer
from deep_nilmtk.utils.logger import start_logging, save_results

# Initialize trainer
trainer = TorchTrainer()
trainer.setup_experiment(experiment_config)

# Start logging
start_logging('experiment.log', params=hyperparams)

# Save results
save_results(
    api_results_f1=f1_scores,
    time=training_time,
    experiment_name='my_experiment'
)
```

### Cross-Validation
**Location**: `./toolkit/deep-nilmtk-v1/deep_nilmtk/trainers/utils/`

```python
from deep_nilmtk.trainers.utils.cross_validator import CrossValidator

# Cross-validation setup
cv = CrossValidator()
cv.setup_folds(data, n_folds=5)
results = cv.run_cross_validation(model, data)
```

### Hyperparameter Optimization
**Location**: `./toolkit/deep-nilmtk-v1/deep_nilmtk/trainers/utils/`

```python
from deep_nilmtk.trainers.utils.hparams_optimizer import HparamsOptimiser

# Hyperparameter optimization
optimizer = HparamsOptimiser()
best_params = optimizer.optimize(
    model_class=YourModel,
    data=train_data,
    search_space=param_space
)
```

---

## Activity Analysis

### NILM-Analyzer Activity Detection
**Location**: `./toolkit/nilm_analyzer/modules/active_durations.py`

```python
from nilm_analyzer.modules.active_durations import get_activities

# Detect appliance activities
activities = get_activities(
    data=power_data,
    target_appliance='washing_machine',
    threshold_x=50,  # Power threshold
    threshold_y=30,  # Duration threshold
    min_limit=60,    # Minimum duration (seconds)
    max_limit=3600   # Maximum duration (seconds)
)
```

---

## Usage Recommendations for AI Systems

### 1. **Data Loading Strategy**
- Use NILM-Analyzer loaders for specific datasets (REFIT, UK-DALE, etc.)
- Use NILMTK DataSet for standard NILM format files
- Validate data using NILM-Analyzer validation functions

### 2. **Model Selection Strategy**
- Start with NILMTK classic algorithms (FHMM, CO) for baseline
- Use NILMTK-Contrib models for deep learning approaches
- Use Torch-NILM for advanced architectures (VIB, attention mechanisms)
- Use Deep-NILMTK for comprehensive experiment management

### 3. **Evaluation Strategy**
- Use Torch-NILM metrics for comprehensive evaluation
- Use NILMTK RMSE for quick performance checks
- Generate reports using Torch-NILM reporting system

### 4. **Experiment Management**
- Use Deep-NILMTK framework for large-scale experiments
- Use Torch-NILM helpers for directory and file management
- Implement cross-validation using Deep-NILMTK validators

### 5. **Code Reuse Patterns**
```python
# Example: Reusable NILM pipeline
def create_nilm_pipeline(dataset_path, model_type='seq2point'):
    # 1. Load data
    if 'refit' in dataset_path.lower():
        loader = REFIT_Loader()
    elif 'ukdale' in dataset_path.lower():
        loader = UKDALE_Loader()
    else:
        dataset = DataSet(dataset_path)
    
    # 2. Select model
    if model_type == 'seq2point':
        model = Seq2PointTorch()
    elif model_type == 'fhmm':
        model = FHMMExact()
    elif model_type == 'rnn':
        model = RNN()
    
    # 3. Train and evaluate
    model.train(train_data)
    predictions = model.disaggregate(test_data)
    
    # 4. Calculate metrics
    metrics = NILMmetrics(predictions, ground_truth)
    
    return metrics
```

This guide provides a comprehensive foundation for AI systems to leverage existing NILM toolkit functionality effectively.
