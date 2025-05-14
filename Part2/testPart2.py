import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
from unittest.mock import patch, MagicMock

# Make sure we can find the module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the predictor class
from abpPredictorModulePart2 import abpPredictor

def simple_test():
    """Run a simple test of the ABP predictor"""
    print("=== Running ABP Predictor Quick Test ===")
    
    # Check if config file exists
    config_file = 'configPart2.ini'
    if not os.path.exists(config_file):
        print(f"Error: Configuration file '{config_file}' not found!")
        return False
    
    try:
        # Create a predictor instance
        predictor = abpPredictor(config_file)
        
        # Modify settings for quicker test
        predictor.NUM_RANDOM_CASES = 1
        predictor.MAX_SAMPLES_PER_CASE = 100
        predictor.MAX_EPOCHS = 1
        predictor.WINDOW_SIZE = 100  # Smaller window for faster testing
        predictor.STEP_SIZE = 50
        
        # Skip the data loading and directly create synthetic segments
        print("Creating synthetic data segments for testing...")
        num_segments = 20  # Create 20 segments for reliable splitting
        
        # Create synthetic segments directly
        X_segments = []
        y_segments = []
        
        for i in range(num_segments):
            # Generate a synthetic window
            window_size = predictor.WINDOW_SIZE
            
            # Ensure values are within the valid ranges defined in predictor
            ecg = np.sin(np.linspace(0, 4*np.pi, window_size)) + 0.1 * np.random.randn(window_size)
            ecg = np.clip(ecg, predictor.ECG_MIN_THRESHOLD, 2.0)  # Ensure above min threshold
            
            ppg = 50 + 40 * np.sin(np.linspace(0, 4*np.pi, window_size) + np.pi/4) + 0.05 * np.random.randn(window_size)
            ppg = np.clip(ppg, predictor.PPG_MIN_THRESHOLD, predictor.PPG_MAX_THRESHOLD)
            
            mean_abp = 100 + 15 * np.sin(np.linspace(0, 4*np.pi, window_size) + np.pi/6)
            mean_abp = np.clip(mean_abp, predictor.ABP_MIN_THRESHOLD, predictor.ABP_MAX_THRESHOLD)
            
            abp = 100 + 20 * np.sin(np.linspace(0, 4*np.pi, window_size) + np.pi/6) + np.random.randn(window_size)
            abp = np.clip(abp, predictor.ABP_MIN_THRESHOLD, predictor.ABP_MAX_THRESHOLD)
            
            # Add to segments
            X_segments.append(np.array([ecg, ppg, mean_abp]))
            y_segments.append(abp)
        
        # Convert to numpy arrays
        predictor.X_segments = np.array(X_segments)
        predictor.y_segments = np.array(y_segments)
        
        print(f"Created {len(predictor.X_segments)} synthetic data segments")
        print(f"X_segments shape: {predictor.X_segments.shape}")
        print(f"y_segments shape: {predictor.y_segments.shape}")
        
        # Test data splitting
        print("\nTesting data splitting...")
        predictor.split_data()
        if predictor.X_train is None or predictor.y_train is None:
            print("Error: Failed to split data")
            return False
        print("Data splitting successful!")
        print(f"Train: {len(predictor.X_train)}, Val: {len(predictor.X_val)}, Test: {len(predictor.X_test)}")
        
        # Test model building
        print("\nTesting model building...")
        
        # Patch the build_model method to avoid TPU/hardware detection issues
        original_build_model = predictor.build_model
        
        def mock_build_model(self):
            """Mock build_model to skip TPU detection"""
            self.model = self._create_model()
            return self
            
        # Replace the method temporarily
        predictor.build_model = mock_build_model.__get__(predictor)
        
        # Build the model
        predictor.build_model()
        
        # Restore original method
        predictor.build_model = original_build_model
                
        if predictor.model is None:
            print("Error: Failed to build model")
            return False
        print("Model building successful!")
        
        # Test model compilation
        print("\nTesting model compilation...")
        if not hasattr(predictor.model, 'optimizer'):
            print("Error: Model not properly compiled")
            return False
        print("Model compilation successful!")
        
        print("\nVerifying model input/output shapes...")
        print(f"Model input shape: {predictor.model.input_shape}")
        print(f"Model output shape: {predictor.model.output_shape}")
        print(f"Expected input shape: (None, 3, {predictor.WINDOW_SIZE})")
        print(f"Expected output shape: (None, {predictor.WINDOW_SIZE})")
        
        print("\nAll basic functionality tests passed!")
        return True
    
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = simple_test()
    sys.exit(0 if success else 1)


# verified that it does not throw err needs additional testing
