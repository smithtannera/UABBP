"""
Main script for ABP prediction
"""
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import os

# Import custom modules
from config import Config
from data_processor import DataProcessor
from model import ABPModel


def main():
    """
    Main function to run the ABP prediction pipeline
    """
    print("Starting ABP Prediction Pipeline")
    start_time = time.time()
    
    # Initialize configuration
    config = Config()
    print(f"Using {config.NUM_SAMPLES} samples")
    
    # Set seed for reproducibility
    np.random.seed(config.RANDOM_SEED)
    tf.random.set_seed(config.RANDOM_SEED)
    
    # Process data
    print("Loading and processing data...")
    data_processor = DataProcessor(config)
    data_processor.load_data()
    data_processor.plot_samples(num_samples=min(3, len(data_processor.X)))
    data_processor.segment_data()
    data_processor.split_data()
    
    # Build and train model
    print("Building and training model...")
    input_shape = (data_processor.X_train.shape[1], data_processor.X_train.shape[2])
    abp_model = ABPModel(config)
    abp_model.build_model(input_shape)
    abp_model.model.summary()
    
    # Train the model
    abp_model.train(
        data_processor.X_train, 
        data_processor.y_mean_train,
        data_processor.y_sys_train,
        data_processor.y_dia_train,
        data_processor.X_val,
        data_processor.y_mean_val,
        data_processor.y_sys_val,
        data_processor.y_dia_val
    )
    
    # Plot training history
    abp_model.plot_training_history()
    
    # Evaluate the model
    print("Evaluating model...")
    abp_model.evaluate(
        data_processor.X_test,
        data_processor.y_mean_test,
        data_processor.y_sys_test,
        data_processor.y_dia_test
    )
    
    # Plot predictions
    abp_model.plot_predictions(num_samples=5)
    
    # Calculate total runtime
    total_time = time.time() - start_time
    print(f"Total runtime: {total_time/60:.2f} minutes")


if __name__ == "__main__":
    main()
