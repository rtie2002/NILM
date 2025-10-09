# Detailed NILM Toolkit Technical Analysis

This comprehensive analysis provides deep insights into all NILM toolkits, focusing on practical implementation details, windowing strategies, activation detection, and operational procedures that are crucial for future NILM projects.

## Table of Contents
1. [NILMTK - Core Foundation](#nilmtk---core-foundation)
2. [NILMTK-Contrib - Advanced Deep Learning](#nilmtk-contrib---advanced-deep-learning)
3. [Torch-NILM - Advanced Neural Architectures](#torch-nilm---advanced-neural-architectures)
4. [Deep-NILMTK - Experiment Management](#deep-nilmtk---experiment-management)
5. [NeuralNILM_Pytorch - Neural Implementations](#neuralnilm_pytorch---neural-implementations)
6. [NILM-Analyzer - Data Validation & Analysis](#nilm-analyzer---data-validation--analysis)
7. [Cross-Toolkit Integration Patterns](#cross-toolkit-integration-patterns)
8. [Implementation Best Practices](#implementation-best-practices)

---

## NILMTK - Core Foundation

### **Core Data Management Architecture**

#### **DataSet Class - Central Hub**
```python
class DataSet:
    def __init__(self, filename=None, format='HDF'):
        self.store = None
        self.buildings = OrderedDict()
        self.metadata = {}
        
    def set_window(self, start=None, end=None):
        """Critical for temporal data selection"""
        tz = self.metadata.get('timezone')
        self.store.window = TimeFrame(start, end, tz)
```

**Key Features:**
- **Temporal Windowing**: `set_window()` allows non-destructive time-based filtering
- **Building Management**: Hierarchical organization of buildings, meters, appliances
- **Metadata Integration**: Automatic loading of dataset metadata and timezone handling
- **Multi-format Support**: HDF, CSV, and automatic format detection

#### **ElecMeter Class - Meter Abstraction**
```python
class ElecMeter:
    def load(self, **kwargs):
        """Load power data with flexible parameters"""
        # Supports chunked loading, column selection, timeframe filtering
        
    def get_timeframe(self):
        """Get available data timeframe"""
        return self.store.get_timeframe(key=self.key)
```

**Advanced Capabilities:**
- **Chunked Loading**: Memory-efficient data processing
- **Flexible Column Selection**: Active, reactive, apparent power
- **Automatic Caching**: Built-in statistics caching system
- **Timeframe Management**: Automatic handling of data gaps and timezones

### **Windowing and Activation Detection**

#### **Steady State Detection Algorithm**
```python
def find_steady_states(dataframe, min_n_samples=2, state_threshold=15, noise_level=70):
    """
    Hart's 1985 algorithm for detecting steady states and transients
    
    Parameters:
    - min_n_samples: Minimum samples for steady state
    - state_threshold: Power change threshold (watts)
    - noise_level: Minimum significant power change
    """
```

**Algorithm Details:**
1. **State Change Detection**: `|current - previous| > state_threshold`
2. **Steady State Averaging**: Running average of power within steady periods
3. **Transition Detection**: Significant power changes above noise level
4. **Clustering**: Automatic clustering of steady states using K-means

#### **Preprocessing Pipeline**
```python
class Apply(Node):
    def process(self):
        """Apply arbitrary function to data chunks"""
        for chunk in self.upstream.process():
            new_chunk = self.func(chunk)
            new_chunk.timeframe = chunk.timeframe
            yield new_chunk
```

**Processing Features:**
- **Streaming Processing**: Memory-efficient chunk-based processing
- **Pipeline Architecture**: Modular preprocessing components
- **Timeframe Preservation**: Automatic handling of temporal information
- **Custom Functions**: Support for arbitrary data transformations

### **Disaggregation Framework**

#### **Base Disaggregator Interface**
```python
class Disaggregator:
    def partial_fit(self, train_mains, train_appliances, **load_kwargs):
        """Incremental learning support"""
        
    def disaggregate_chunk(self, test_mains):
        """Chunk-based disaggregation"""
        
    def save_model(self, folder_name):
        """Model persistence"""
        
    def load_model(self, folder_name):
        """Model loading"""
```

**Framework Benefits:**
- **Incremental Learning**: Support for online learning scenarios
- **Model Persistence**: Automatic checkpoint management
- **Memory Management**: Automatic cleanup of temporary files
- **Standardized Interface**: Consistent API across all algorithms

---

## NILMTK-Contrib - Advanced Deep Learning

### **Sequence-to-Point Architecture**

#### **Windowing Strategy**
```python
def preprocess(sequence_length=99, mains_mean=1800, mains_std=600, 
               mains_lst=None, submeters_lst=None, method="train"):
    """
    Advanced preprocessing with windowing support
    
    Key Parameters:
    - sequence_length: Must be odd for proper centering
    - mains_mean/std: Global normalization parameters
    - windowing: Boolean for sequence vs point prediction
    """
    pad = sequence_length // 2  # Center the target point
    v = np.pad(mains.values.flatten(), (pad, pad))
    windows = np.array([v[i:i+sequence_length] 
                       for i in range(len(v)-sequence_length + 1)])
    windows = (windows - mains_mean) / mains_std
```

**Windowing Details:**
- **Centered Windows**: Target point at center of sequence
- **Padding Strategy**: Symmetric padding to handle boundaries
- **Normalization**: Z-score normalization with global statistics
- **Flexible Output**: Support for both sequence and point prediction

#### **CNN Architecture for Seq2Point**
```python
class Seq2PointTorch:
    def _build_network(self):
        model = nn.Sequential(
            # Feature extraction with 1D convolutions
            nn.Conv1d(1, 30, kernel_size=10, stride=1), nn.ReLU(),
            nn.Conv1d(30, 30, kernel_size=8, stride=1), nn.ReLU(),
            nn.Conv1d(30, 40, kernel_size=6, stride=1), nn.ReLU(),
            nn.Conv1d(40, 50, kernel_size=5, stride=1), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Conv1d(50, 50, kernel_size=5, stride=1), nn.ReLU(),
            nn.Dropout(0.2),
            
            # Dense layers for prediction
            nn.Flatten(),
            nn.Linear(50 * (seq_len - 29), 1024), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 1)  # Single power value output
        )
```

**Architecture Benefits:**
- **Progressive Feature Extraction**: Increasing filter sizes for multi-scale features
- **Regularization**: Dropout layers for overfitting prevention
- **Memory Efficiency**: 1D convolutions optimized for power sequences
- **Flexible Input**: Handles variable sequence lengths

### **Advanced Models**

#### **RNN with Attention**
```python
class RNN_attention(Disaggregator):
    """RNN with attention mechanism for sequence modeling"""
    
class AttentionLayer(nn.Module):
    """Multi-head attention for capturing long-range dependencies"""
```

#### **ResNet Classification**
```python
class ResNet_classification(Disaggregator):
    """Residual networks for appliance classification"""
    
class IdentityBlock(nn.Module):
    """Residual block with skip connections"""
```

---

## Torch-NILM - Advanced Neural Architectures

### **Base Model Framework**

#### **Capability Detection System**
```python
class BaseModel(nn.Module, abc.ABC):
    def supports_classic_training(self) -> bool:
        return True
    
    def supports_vib(self) -> bool:
        """Variational Information Bottleneck"""
        return False
    
    def supports_bayes(self) -> bool:
        """Bayesian inference"""
        return False
    
    def supports_bert(self) -> bool:
        """BERT-style training"""
        return False
```

**Advanced Training Modes:**
- **Classic Training**: Standard supervised learning
- **VIB Training**: Variational Information Bottleneck for uncertainty quantification
- **Bayesian Training**: Bayesian neural networks for model uncertainty
- **BERT Training**: Bidirectional encoder representations

### **Variational Information Bottleneck (VIB)**

#### **VIB Implementation**
```python
class VIBNet(BaseModel):
    def reparametrize_n(self, mu, std, current_epoch, n=1, 
                       prior_std=1.0, prior_mean=0.0, 
                       distribution=NORMAL_DIST):
        """
        Reparameterization trick with adaptive noise
        
        Parameters:
        - mu, std: Mean and standard deviation of latent distribution
        - current_epoch: For adaptive noise scheduling
        - distribution: Noise distribution type (Normal, Cauchy, Student-t)
        """
        noise_rate = torch.tanh(torch.tensor(current_epoch))
        if current_epoch > 0:
            noise_distribution = self.data_distribution_type(
                distribution=distribution,
                loc=prior_mean,
                scale=noise_rate * prior_std
            )
            eps = noise_distribution.sample(std.size())
        return mu + eps * std
```

**VIB Benefits:**
- **Uncertainty Quantification**: Provides confidence intervals for predictions
- **Adaptive Noise**: Noise rate increases with training epochs
- **Multiple Distributions**: Support for various noise distributions
- **Information Bottleneck**: Balances accuracy and information compression

### **Advanced Architectures**

#### **Self-Attention Encoder-Decoder**
```python
class SAED(BaseModel):
    """Self-Attention Encoder-Decoder for sequence modeling"""
    
class VIB_SAED(SAED, VIBNet):
    """SAED with VIB capabilities"""
```

#### **Wide Gated Recurrent Unit**
```python
class WGRU(BaseModel):
    """Wide Gated Recurrent Unit for temporal modeling"""
    
class VIBWGRU(WGRU, VIBNet):
    """WGRU with VIB capabilities"""
```

### **Utility Functions**

#### **Experiment Management**
```python
def create_tree_dir(tree_levels: dict = None, clean: bool = False, 
                   plots: bool = False, output_dir: str = DIR_OUTPUT_NAME):
    """
    Hierarchical directory structure creation
    
    Example tree_levels:
    {
        'experiment': 'exp1',
        'model': 'lstm',
        'dataset': 'ukdale',
        'fold': ['fold1', 'fold2', 'fold3']
    }
    """
```

#### **Time Series Operations**
```python
def create_timeframes(start: object, end: object, freq: object):
    """Create time frame ranges for cross-validation"""
    
def create_time_folds(start_date: str, end_date: str, folds: int, 
                     drop_last: bool = False):
    """Create temporal folds for time-series cross-validation"""
```

#### **Data Normalization**
```python
def destandardize(data: np.array, means: float, stds: float):
    """Reverse Z-score normalization"""
    
def denormalize(data: np.array, mmax: float):
    """Reverse min-max normalization"""
```

---

## Deep-NILMTK - Experiment Management

### **Experiment Framework**

#### **NILMExperiment Class**
```python
class NILMExperiment(Disaggregator):
    def __init__(self, params):
        self.hparams = get_exp_parameters()
        self.trainer = Trainer(self.get_trainer(), self.hparams)
        
    def partial_fit(self, mains, sub_main, do_preprocessing=True):
        """Complete experiment pipeline"""
        # STEP 01: Pre-processing
        if do_preprocessing:
            mains, params, sub_main = preprocess(mains, self.hparams['input_norm'], sub_main)
        
        # STEP 02: Feature engineering
        mains = generate_features(mains, self.hparams)
        
        # STEP 03: Model Training
        self.models, self.appliance_params = self.trainer.fit(mains, sub_main)
```

**Experiment Features:**
- **Automated Pipeline**: Complete preprocessing to training pipeline
- **Hyperparameter Management**: Centralized parameter configuration
- **Multi-Framework Support**: PyTorch and TensorFlow backends
- **MLflow Integration**: Automatic experiment tracking

### **Preprocessing System**

#### **Normalization Options**
```python
def z_norm(data, params=None):
    """Z-score normalization with parameter reuse"""
    mean = params['mean'] if params else data.mean(axis=0)
    std = params['std'] if params else data.std(axis=0)
    return {"mean": mean, "std": std}, (data - mean) / std

def min_max_norm(data, params=None):
    """Min-max normalization"""
    min_ = params['min'] if params else data.min(axis=0)
    max_ = params['max'] if params else data.max(axis=0)
    return {'min': min_, 'max': max_}, (data - min_) / (max_ - min_)

def log_norm(data, params=None):
    """Log normalization for positive values"""
    return {}, np.log(data + 1)
```

#### **Feature Engineering**
```python
def generate_features(mains, hparams):
    """Advanced feature generation"""
    # Statistical features: mean, std, min, max, quantiles
    # Temporal features: hour of day, day of week
    # Frequency domain features: FFT, spectral analysis
    # Rolling window features: moving averages, trends
```

### **Training System**

#### **Cross-Validation Support**
```python
class CrossValidator:
    """Time-series aware cross-validation"""
    
    def setup_folds(self, data, n_folds=5):
        """Create temporal folds respecting time order"""
        
    def run_cross_validation(self, model, data):
        """Execute cross-validation with proper temporal splits"""
```

#### **Hyperparameter Optimization**
```python
class HparamsOptimiser:
    """Optuna-based hyperparameter optimization"""
    
    def optimize(self, model_class, data, search_space):
        """Bayesian optimization of hyperparameters"""
```

### **MLflow Integration**

#### **Experiment Tracking**
```python
# Automatic logging of:
# - Hyperparameters
# - Metrics (MAE, RMSE, F1-Score)
# - Model artifacts
# - Training curves
# - System metrics

with mlflow.start_run(run_name=self.hparams['model_name']):
    mlflow.log_params(self.hparams)
    # Training code
    mlflow.log_metrics(final_metrics)
```

---

## NeuralNILM_Pytorch - Neural Implementations

### **Core Utilities**

#### **Enhanced Utils Functions**
```python
def show_versions():
    """Display versions of all dependencies"""
    
def timedelta64_to_secs(timedelta):
    """Convert timedelta to seconds"""
    
def find_nearest(known_array, test_array):
    """Find nearest values in arrays"""
    
def compute_rmse(ground_truth, predictions, pretty=True):
    """Compute RMSE with pretty formatting"""
```

### **Data Processing**

#### **Timestamp Handling**
```python
def convert_to_timestamp(t):
    """Robust timestamp conversion"""
    
def get_index(data):
    """Extract index from various data formats"""
    
def normalise_timestamp(timestamp, freq):
    """Normalize timestamps to specified frequency"""
```

### **Statistical Operations**
```python
def most_common(lst):
    """Find most common element in list"""
    
def capitalise_first_letter(string):
    """Capitalize first letter of string"""
    
def flatten_2d_list(list2d):
    """Flatten 2D list to 1D"""
```

---

## NILM-Analyzer - Data Validation & Analysis

### **Activation Detection System**

#### **Advanced Activity Detection**
```python
def get_activities(data, target_appliance=None, threshold_x=None, 
                  threshold_y=None, min_limit=None, max_limit=None):
    """
    Comprehensive activity detection with multiple thresholds
    
    Parameters:
    - threshold_x: Time threshold for merging close activities (minutes)
    - threshold_y: Power threshold for activity detection (watts)
    - min_limit: Minimum activity duration (minutes)
    - max_limit: Maximum activity duration (minutes)
    """
```

**Detection Algorithm:**
1. **Power Thresholding**: Identify periods above `threshold_y`
2. **Gap Merging**: Merge activities within `threshold_x` minutes
3. **Duration Filtering**: Filter by `min_limit` and `max_limit`
4. **Temporal Analysis**: Calculate start/end times and durations

#### **Activity Report Generation**
```python
def __generate_activity_report(df, target_appliance, threshold_x, 
                              threshold_y, min_limit, max_limit):
    """
    Generate detailed activity reports
    
    Returns:
    - activity_start: Start timestamps
    - activity_end: End timestamps  
    - duration_in_minutes: Activity durations
    """
```

### **Data Validation Framework**

#### **House Availability Check**
```python
def check_house_availability(arg_name, arg_value, collection):
    """Validate house existence in dataset"""
    if arg_value in collection:
        return True
    elif isinstance(arg_value, str):
        print(f"TypeError: String not accepted. Expected int")
        return False
    else:
        print(f"{arg_name} = {arg_value} does not exist in dataset")
        return False
```

#### **Data Type Validation**
```python
def check_correct_datatype(arg_name, arg_value, target_datatype):
    """Validate data types with detailed error messages"""
    
def check_list_validations(arg_name, arg_value, member_datatype):
    """Validate list contents and types"""
```

### **Dataset-Specific Loaders**

#### **Multi-Dataset Support**
```python
class REFIT_Loader(CSV_Loader):
    """REFIT dataset specific loader with validation"""
    
class UKDALE_Loader(CSV_Loader):
    """UK-DALE dataset specific loader"""
    
class AMPDS_Loader(CSV_Loader):
    """AMPDS dataset specific loader"""
    
class IAWE_Loader(CSV_Loader):
    """iAWE dataset specific loader"""
```

**Loader Features:**
- **Dataset-Specific Parsing**: Custom parsing for each dataset format
- **Automatic Validation**: Built-in data quality checks
- **Metadata Integration**: Automatic metadata loading and validation
- **Flexible Loading**: Support for partial data loading

---

## Cross-Toolkit Integration Patterns

### **Unified Data Loading Pipeline**

```python
def create_unified_loader(dataset_name, building_id):
    """Unified data loading across toolkits"""
    if dataset_name.lower() == 'refit':
        loader = REFIT_Loader()
    elif dataset_name.lower() == 'ukdale':
        loader = UKDALE_Loader()
    else:
        # Use NILMTK DataSet
        dataset = DataSet(f"{dataset_name}.h5")
        return dataset.buildings[building_id]
    
    return loader.load_data(building=building_id)
```

### **Cross-Toolkit Preprocessing**

```python
def unified_preprocessing(data, toolkit='nilmtk_contrib'):
    """Unified preprocessing across toolkits"""
    if toolkit == 'nilmtk_contrib':
        return preprocess(sequence_length=99, mains_lst=[data])
    elif toolkit == 'torch_nilm':
        return create_timeframes(data.index[0], data.index[-1], '6T')
    elif toolkit == 'deep_nilmtk':
        return normalize(data.values, type='z-norm')
    else:
        # NILMTK steady state detection
        return find_steady_states(data, noise_level=70)
```

### **Multi-Toolkit Evaluation**

```python
def comprehensive_evaluation(predictions, ground_truth):
    """Evaluation using multiple toolkits"""
    # NILMTK metrics
    rmse_nilmtk = compute_rmse(ground_truth, predictions)
    
    # Torch-NILM comprehensive metrics
    metrics_torch = NILMmetrics(predictions, ground_truth, threshold=40)
    
    # NILM-analyzer activity analysis
    activities_pred = get_activities(predictions, threshold_y=50)
    activities_true = get_activities(ground_truth, threshold_y=50)
    
    return {
        'nilmtk': {'rmse': rmse_nilmtk},
        'torch_nilm': metrics_torch,
        'activity_analysis': {
            'predicted_activities': len(activities_pred),
            'true_activities': len(activities_true)
        }
    }
```

---

## Implementation Best Practices

### **1. Windowing Strategy Selection**

#### **For Sequence Models (Seq2Seq, RNN)**
```python
# Use NILMTK-Contrib windowing
def setup_sequence_windows(data, sequence_length=99):
    return preprocess(sequence_length=sequence_length, mains_lst=[data])
```

#### **For Point Models (Seq2Point)**
```python
# Use centered windows with odd sequence length
def setup_point_windows(data, sequence_length=99):
    if sequence_length % 2 == 0:
        sequence_length += 1  # Ensure odd for centering
    return preprocess(sequence_length=sequence_length, mains_lst=[data])
```

#### **For Activity Detection**
```python
# Use NILM-analyzer with multiple thresholds
def setup_activity_detection(data, appliance):
    return get_activities(
        data, 
        target_appliance=appliance,
        threshold_x=5,    # 5 minutes gap threshold
        threshold_y=50,   # 50 watts power threshold
        min_limit=1.0,    # 1 minute minimum duration
        max_limit=120.0   # 2 hours maximum duration
    )
```

### **2. Activation Detection Pipeline**

```python
def comprehensive_activation_detection(data, appliance):
    """Multi-stage activation detection"""
    
    # Stage 1: NILMTK steady state detection
    steady_states, transients = find_steady_states(
        data, 
        state_threshold=15,  # 15W threshold
        noise_level=70       # 70W noise level
    )
    
    # Stage 2: NILM-analyzer activity detection
    activities = get_activities(
        data,
        target_appliance=appliance,
        threshold_y=50,
        min_limit=2.0
    )
    
    # Stage 3: Validation and filtering
    validated_activities = validate_activities(activities, steady_states)
    
    return validated_activities
```

### **3. Experiment Setup Best Practices**

```python
def setup_comprehensive_experiment(dataset_name, model_type, appliances):
    """Complete experiment setup using multiple toolkits"""
    
    # 1. Data loading with validation
    data = create_unified_loader(dataset_name, building_id=1)
    validate_data_quality(data, appliances)
    
    # 2. Preprocessing pipeline
    if model_type in ['seq2point', 'seq2seq']:
        processed_data = unified_preprocessing(data, 'nilmtk_contrib')
    elif model_type in ['vib', 'bayesian']:
        processed_data = unified_preprocessing(data, 'torch_nilm')
    else:
        processed_data = unified_preprocessing(data, 'deep_nilmtk')
    
    # 3. Model initialization
    if model_type == 'seq2point':
        model = Seq2PointTorch(params)
    elif model_type == 'vib':
        model = VIB_SAED()
    else:
        model = NILMExperiment(params)
    
    # 4. Training with cross-validation
    if use_cross_validation:
        cv = CrossValidator()
        results = cv.run_cross_validation(model, processed_data)
    else:
        model.partial_fit(train_data, train_appliances)
    
    # 5. Comprehensive evaluation
    predictions = model.disaggregate_chunk(test_data)
    evaluation = comprehensive_evaluation(predictions, ground_truth)
    
    return model, evaluation
```

### **4. Memory Management**

```python
def memory_efficient_processing(data, chunk_size=10000):
    """Memory-efficient processing for large datasets"""
    
    # Use NILMTK chunked loading
    for chunk in data.load(chunksize=chunk_size):
        # Process chunk
        processed_chunk = preprocess_chunk(chunk)
        
        # Yield results to avoid memory accumulation
        yield processed_chunk
        
        # Clear memory
        del chunk
        gc.collect()
```

### **5. Error Handling and Validation**

```python
def robust_experiment_pipeline(data, model_params):
    """Robust pipeline with comprehensive error handling"""
    
    try:
        # Validate inputs
        validate_input_data(data)
        validate_model_params(model_params)
        
        # Setup experiment
        experiment = NILMExperiment(model_params)
        
        # Training with checkpointing
        experiment.partial_fit(train_data, train_appliances)
        
        # Testing with validation
        predictions = experiment.disaggregate_chunk(test_data)
        validate_predictions(predictions)
        
        return experiment, predictions
        
    except SequenceLengthError as e:
        logger.error(f"Invalid sequence length: {e}")
        # Auto-fix sequence length
        model_params['sequence_length'] = 99
        return robust_experiment_pipeline(data, model_params)
        
    except ApplianceNotFoundError as e:
        logger.error(f"Appliance not found: {e}")
        # Use available appliances
        available_appliances = get_available_appliances(data)
        return robust_experiment_pipeline(data, model_params)
```

This detailed analysis provides comprehensive insights into all NILM toolkits, enabling effective reuse of existing functionality and implementation of robust NILM systems. The focus on practical implementation details, windowing strategies, and activation detection makes this guide particularly valuable for future NILM projects.
