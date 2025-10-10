import torch
import numpy as np
import matplotlib.pyplot as plt
from cnn_model import NILMCNN
from preprocessing import NILMPreprocessor

def test_model_predictions():
    """Test the trained model on new data"""
    
    # Load the trained model
    model = NILMCNN(window_size=99)
    model.load_state_dict(torch.load('model_train.pth'))
    model.eval()
    
    # Load some test data
    DATASET_PATH = r"C:\Users\Raymond Tie\Desktop\NILM\datasets\ukdale.h5"
    preprocessor = NILMPreprocessor(DATASET_PATH)
    train_start, train_end, test_start, test_end = preprocessor.set_time_windows()
    
    # Load test data
    preprocessor.load_testing_data(test_start, test_end, ['washer dryer'])
    processed_data = preprocessor.create_windows_and_normalize(window_size=99, stride=1)
    
    # Get test data
    X_test = processed_data['X_test']
    y_test = processed_data['y_test']['washer dryer']
    
    # Make predictions on first 5 samples
    with torch.no_grad():
        for i in range(5):
            # Prepare input
            input_data = torch.FloatTensor(X_test[i]).unsqueeze(0).unsqueeze(0)  # (1, 1, 99)
            
            # Make prediction
            prediction = model(input_data)
            prediction = prediction.squeeze().numpy()
            
            # Get actual values
            actual = y_test[i]
            input_mains = X_test[i]
            
            # Denormalize (you'd need the stats from preprocessing)
            # For now, just show normalized values
            
            # Plot comparison
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(input_mains, 'b-', label='Mains Power (Input)')
            plt.title('Input: Total Mains Power')
            plt.ylabel('Normalized Power')
            plt.legend()
            
            plt.subplot(1, 3, 2)
            plt.plot(actual, 'g-', label='Actual Washer Dryer')
            plt.title('Actual: Washer Dryer Power')
            plt.ylabel('Normalized Power')
            plt.legend()
            
            plt.subplot(1, 3, 3)
            plt.plot(actual, 'g-', label='Actual', alpha=0.7)
            plt.plot(prediction, 'r--', label='Predicted', linewidth=2)
            plt.title('Prediction vs Actual')
            plt.ylabel('Normalized Power')
            plt.legend()
            
            plt.tight_layout()
            plt.show()
            
            # Calculate error
            mse = np.mean((actual - prediction) ** 2)
            print(f"Sample {i+1} - MSE: {mse:.4f}")
            print(f"Actual range: [{actual.min():.3f}, {actual.max():.3f}]")
            print(f"Predicted range: [{prediction.min():.3f}, {prediction.max():.3f}]")
            print("-" * 50)

if __name__ == "__main__":
    test_model_predictions()
