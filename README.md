# NILM Toolkit Collection

This project contains a comprehensive collection of NILM (Non-Intrusive Load Monitoring) toolkits, organized for easy access and integration.

## 📚 Documentation

**📖 [Complete Documentation](./docs/README.md)**

The `docs/` folder contains comprehensive documentation:
- **[Toolkit Summary](./docs/NILM_TOOLKITS_SUMMARY.md)** - Overview of all toolkits
- **[Function Guide](./docs/NILM_TOOLKIT_FUNCTIONS_GUIDE.md)** - AI development guide
- **[Technical Analysis](./docs/DETAILED_NILM_TOOLKIT_ANALYSIS.md)** - Deep technical insights

## 🛠️ Available Toolkits

All toolkits are located in the `toolkit/` directory:

| Toolkit | Location | Capabilities |
|---------|----------|--------------|
| **NILMTK** | `toolkit/nilmtk/` | Core functionality, classic algorithms |
| **NILMTK-Contrib** | `toolkit/nilmtk-contrib/` | Advanced deep learning models |
| **Torch-NILM** | `toolkit/torch-nilm/` | Advanced neural architectures |
| **Deep-NILMTK** | `toolkit/deep-nilmtk-v1/` | Experiment management framework |
| **NeuralNILM_Pytorch** | `toolkit/NeuralNILM_Pytorch/` | Neural network implementations |
| **NILM-Analyzer** | `toolkit/nilm_analyzer/` | Data validation and analysis |

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Install Toolkits (Editable Mode)

```bash
pip install -e ./toolkit/nilmtk
pip install -e ./toolkit/nilmtk-contrib
pip install -e ./toolkit/torch-nilm
pip install -e ./toolkit/deep-nilmtk-v1
pip install -e ./toolkit/NeuralNILM_Pytorch
pip install -e ./toolkit/nilm_analyzer
```

### 3. Start Development

```bash
cursor-notebook-mcp --transport streamable-http --allow-root "C:\Users\Raymond Tie\Desktop\NILM" --host 127.0.0.1 --port 8080
```

## 📁 Project Structure

```
NILM/
├── docs/                          # 📚 Comprehensive documentation
├── toolkit/                       # 🛠️ All NILM toolkits
│   ├── nilmtk/                   # Core NILMTK toolkit
│   ├── nilmtk-contrib/           # Advanced algorithms
│   ├── torch-nilm/               # PyTorch-based toolkit
│   ├── deep-nilmtk-v1/           # Deep learning framework
│   ├── NeuralNILM_Pytorch/       # Neural implementations
│   └── nilm_analyzer/            # Analysis toolkit
├── notebooks/                     # 📓 Jupyter notebooks
├── datasets/                      # 📊 NILM datasets
├── src/                          # 💻 Your project code
├── requirements.txt              # 📦 Dependencies
└── README.md                     # 📖 This file
```

## 🎯 Key Features

- **Complete Toolkit Collection**: All major NILM toolkits in one place
- **Comprehensive Documentation**: Detailed guides for AI development
- **Easy Integration**: Pre-configured for seamless toolkit interaction
- **Cross-Toolkit Support**: Functions that work across multiple toolkits
- **Production Ready**: Best practices and implementation patterns

## 🤖 AI Development Support

This project is specifically designed to help AI systems:
- **Discover Functions**: Easy identification of existing capabilities
- **Reuse Code**: Avoid reinventing common NILM functionality
- **Integrate Toolkits**: Combine functions across different toolkits
- **Follow Best Practices**: Leverage proven implementation patterns

---

**📖 For detailed information, start with the [Documentation](./docs/README.md)**
