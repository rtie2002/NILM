# NILM Toolkits Summary

This document provides an overview of all the NILM (Non-Intrusive Load Monitoring) toolkits that have been cloned into this project.

## Cloned Toolkits

### 1. NILMTK (2014) - [Reference 15]
- **Repository**: `./toolkit/nilmtk/`
- **GitHub**: https://github.com/nilmtk/nilmtk
- **Capabilities**: Energy Disaggregation (ED), Smart Disaggregation (SD)
- **Description**: The original NILMTK toolkit - Non-Intrusive Load Monitoring Toolkit
- **Key Features**: 
  - Dataset converters for multiple datasets (REDD, UK-DALE, iAWE, etc.)
  - Built-in disaggregation algorithms (FHMM, CO)
  - Data preprocessing and analysis tools
  - Visualization capabilities

### 2. NILMTK-Contrib (2019) - [Reference 16]
- **Repository**: `./toolkit/nilmtk-contrib/`
- **GitHub**: https://github.com/nilmtk/nilmtk-contrib
- **Capabilities**: Energy Disaggregation (ED), Smart Disaggregation (SD)
- **Description**: State-of-the-art algorithms for energy disaggregation using NILMTK's Rapid Experimentation API
- **Key Features**:
  - Advanced deep learning models
  - TensorFlow and PyTorch support
  - Modern API for rapid experimentation

### 3. Torch-NILM (2022) - [Reference 99]
- **Repository**: `./toolkit/torch-nilm/`
- **GitHub**: https://github.com/Virtsionis/torch-nilm
- **Capabilities**: Energy Disaggregation (ED), Smart Disaggregation (SD)
- **Description**: PyTorch-based NILM toolkit with advanced neural networks
- **Key Features**:
  - PyTorch Lightning integration
  - Advanced neural network architectures
  - Weights & Biases integration for experiment tracking
  - Interactive Dash dashboards

### 4. Deep-NILMTK (2022) - [Reference 100]
- **Repository**: `./toolkit/deep-nilmtk-v1/`
- **GitHub**: https://github.com/BHafsa/deep-nilmtk-v1
- **Capabilities**: Energy Disaggregation (ED)
- **Description**: Deep learning framework for NILM with extensive model support
- **Key Features**:
  - Multiple deep learning models (CNN, LSTM, Transformer, etc.)
  - MLflow integration for experiment tracking
  - Optuna for hyperparameter optimization
  - Comprehensive documentation and tutorials

### 5. NeuralNILM_Pytorch (2020) - [Reference 101]
- **Repository**: `./toolkit/NeuralNILM_Pytorch/`
- **GitHub**: https://github.com/Ming-er/NeuralNILM_Pytorch
- **Capabilities**: Energy Disaggregation (ED)
- **Description**: Neural network implementations for NILM using PyTorch
- **Key Features**:
  - Custom neural network architectures
  - PyTorch-based implementations
  - Tutorial notebooks and examples

### 6. NILM-analyzer (2022)
- **Repository**: `./toolkit/nilm_analyzer/`
- **GitHub**: https://github.com/mahnoor-shahid/nilm_analyzer
- **Capabilities**: Analysis and validation tools
- **Description**: Analysis toolkit for NILM datasets and results
- **Key Features**:
  - Dataset validation and analysis
  - Metadata management
  - Dask integration for large-scale processing

## Installation Notes

All toolkits have been configured in the `requirements.txt` file with their respective dependencies. To install all toolkits in editable mode:

```bash
pip install -e ./toolkit/nilmtk
pip install -e ./toolkit/nilmtk-contrib
pip install -e ./toolkit/torch-nilm
pip install -e ./toolkit/deep-nilmtk-v1
pip install -e ./toolkit/NeuralNILM_Pytorch
pip install -e ./toolkit/nilm_analyzer
```

Or install all dependencies at once:
```bash
pip install -r requirements.txt
```

## Usage

Each toolkit can be imported and used independently:

```python
# NILMTK
import nilmtk
from nilmtk import DataSet

# NILMTK-Contrib
import nilmtk_contrib

# Torch-NILM
import torch_nilm

# Deep-NILMTK
import deep_nilmtk

# NeuralNILM_Pytorch
# (Check individual repository for import structure)

# NILM-analyzer
import nilm_analyzer
```

## Compatibility

- **Python**: Most toolkits require Python 3.7+
- **PyTorch**: Version compatibility varies by toolkit
- **TensorFlow**: Some toolkits require specific TensorFlow versions
- **CUDA**: GPU support available in most deep learning toolkits

## Project Structure

Your NILM project now contains:
```
C:\Users\Raymond Tie\Desktop\NILM\
├── toolkit/                   # All NILM toolkits organized in one folder
│   ├── nilmtk/                # Original NILMTK toolkit
│   ├── nilmtk-contrib/        # Advanced NILMTK algorithms
│   ├── torch-nilm/            # PyTorch-based toolkit
│   ├── deep-nilmtk-v1/        # Deep learning framework
│   ├── NeuralNILM_Pytorch/    # Neural network implementations
│   └── nilm_analyzer/         # Analysis toolkit
├── requirements.txt           # Updated with all dependencies
├── NILM_TOOLKITS_SUMMARY.md   # Comprehensive documentation
└── [your existing files]
```

## Next Steps

1. Install dependencies using the requirements.txt file
2. Explore individual toolkit documentation
3. Run example notebooks and tutorials
4. Integrate toolkits into your NILM research pipeline
