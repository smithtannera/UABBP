"""
Configuration settings for ABP prediction model
"""

class Config:
    # Data collection parameters
    NUM_SAMPLES = 20  # Increased from 2 to 20 samples
    SAMPLE_WINDOW_SIZE = 2000  # Window size for signal sampling
    STEP_SIZE = 150000  # 5 minute downsampling
    
    # Segmentation parameters
    WINDOW_SIZE = 4000  # Size of segments for training
    STEP = 3000  # Step size for segmentation
    MAX_SEGMENT_RANGE = 175000  # Maximum range to consider for segmentation
    
    # Signal filtering parameters
    ECG_MIN_THRESHOLD = -1.5
    PPG_MAX_THRESHOLD = 100
    PPG_MIN_THRESHOLD = 0
    BP_MAX_THRESHOLD = 200
    BP_MIN_THRESHOLD = 20
    
    # Model parameters
    LSTM_UNITS = 64
    DROPOUT_RATE = 0.2
    RECURRENT_DROPOUT = 0.2
    DENSE_UNITS = 64
    OUTPUT_DROPOUT = 0.5
    L2_REG = 0.01
    
    # Training parameters
    BATCH_SIZE = 64
    MAX_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 5
    VALIDATION_SPLIT = 0.2
    TEST_SPLIT = 0.2
    RANDOM_SEED = 42
