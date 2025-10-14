# Model Testing Guide

This guide will help you test your trained NILM model step by step.

## Files Created

1. **`test_model_loading.py`** - Ultra-simple test that just loads your model
2. **`simple_model_test.py`** - Complete step-by-step testing pipeline
3. **`MODEL_TESTING_GUIDE.md`** - This guide

## Quick Start

### Step 1: Test Model Loading (Simplest)

```bash
python test_model_loading.py
```

This will:
- Load your `model_train.pth` file
- Create the model architecture
- Load the trained weights
- Test with dummy data
- Verify everything works

### Step 2: Complete Test with Real Data

```bash
python simple_model_test.py
```

This will:
- Load your model
- Load one day of real data from UK-DALE
- Make predictions on test samples
- Show visualizations
- Calculate performance metrics

## What Each Script Does

### `test_model_loading.py` (Ultra-Simple)

**Purpose**: Just verify your model can be loaded and works

**What it tests**:
- ✅ Model file exists
- ✅ Model can be loaded
- ✅ Model architecture is correct
- ✅ Model can make predictions
- ✅ Output format is correct

**Output**: Simple success/failure message

### `simple_model_test.py` (Complete Pipeline)

**Purpose**: Full testing with real data and metrics

**What it does**:
1. **Step 1**: Load the trained model
2. **Step 2**: Load one day of real UK-DALE data
3. **Step 3**: Make predictions on test samples
4. **Step 4**: Visualize results (plots)
5. **Step 5**: Calculate performance metrics

**Output**: 
- Visual plots showing predictions vs actual
- Performance metrics (MAE, RMSE, Energy Error)
- Detailed analysis

## Understanding the Results

### Model Loading Test
- If successful: "SUCCESS! Your model is working! 🎉"
- If failed: Error message explaining what went wrong

### Complete Test Results

**Visualizations**:
- **Input**: Total mains power (what the model sees)
- **Actual**: Real appliance power (ground truth)
- **Prediction**: Model's prediction vs actual

**Metrics**:
- **MAE**: Mean Absolute Error (average difference in watts)
- **RMSE**: Root Mean Square Error (penalizes large errors more)
- **Energy Error**: Percentage difference in total energy consumption

## Troubleshooting

### Common Issues

1. **"Model file not found"**
   - Make sure you've trained a model first
   - Run `model_train.py` to create `model_train.pth`

2. **"Not enough data"**
   - The script loads one day of data
   - Make sure your UK-DALE dataset is properly set up

3. **Import errors**
   - Make sure all required packages are installed
   - Check that `cnn_model.py` and `preprocessing.py` are in the same directory

### Expected Performance

For a well-trained model on washer dryer:
- **MAE**: 50-200W (lower is better)
- **RMSE**: 100-300W (lower is better)
- **Energy Error**: 5-20% (lower is better)

## Next Steps

Once your model is working:

1. **Improve the model**: Train for more epochs or adjust architecture
2. **Test on different appliances**: Modify `TARGET_APPLIANCES` in preprocessing
3. **Test on different time periods**: Change the date ranges
4. **Compare with other models**: Use the same test data for fair comparison

## Learning Path

1. **Start with**: `test_model_loading.py` (simplest)
2. **Then try**: `simple_model_test.py` (complete test)
3. **Understand**: The visualizations and metrics
4. **Experiment**: Modify the scripts to test different scenarios
5. **Improve**: Use the results to improve your model

## Customization

You can modify the scripts to:
- Test different appliances
- Use different time periods
- Change the number of test samples
- Add more performance metrics
- Test on different datasets

Happy testing! 🚀
