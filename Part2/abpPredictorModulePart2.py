#####            Importing the Libraries            #####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import vitaldb
import math
import configparser
import os
import random
from tensorflow.keras.preprocessing.sequence import pad_sequences

class abpPredictor:
    def __init__(self, config_file='configPart2.ini'):
        """
        Initialize the ABP predictor by loading configuration from the INI file.
        
        Args:
            config_file: Path to the configuration INI file
        """
        # Load configuration from INI file
        self._load_config(config_file)
        
        # Initialize data structures
        self.X = []
        self.y = []
        self.X_segments = None
        self.y_segments = None
        self.X_train = None
        self.X_val = None
        self.X_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.model = None
        self.history = None
        self.y_pred = None
    
    def _load_config(self, config_file):
        """Load configuration parameters from INI file"""
        # Check if the config file exists
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Configuration file '{config_file}' not found!")
        
        # Parse the configuration file
        parser = configparser.ConfigParser()
        parser.read(config_file)
        
        # Data loading parameters
        self.NUM_RANDOM_CASES = parser.getint('dataLoadingParams', 'numrandomcases')
        self.SAMPLING_RATE = parser.getint('dataLoadingParams', 'samplingrate')
        
        # Signal processing parameters
        self.WINDOW_SIZE = parser.getint('signalProcessingParams', 'windowsize')
        self.STEP_SIZE = parser.getint('signalProcessingParams', 'stepsize')
        self.MAX_SAMPLES_PER_CASE = 175000  # Default value, not in INI file
        
        # ART downsampling parameters
        self.DOWNSAMPLE_WINDOW = parser.getint('downsampling', 'downwindow')
        self.DOWNSAMPLE_STRIDE = parser.getint('downsampling', 'downsampleskip')
        
        # Data filtering thresholds - Using default values, not in INI file
        self.ECG_MIN_THRESHOLD = -1.5
        self.PPG_MIN_THRESHOLD = 0
        self.PPG_MAX_THRESHOLD = 100
        self.ABP_MIN_THRESHOLD = 20
        self.ABP_MAX_THRESHOLD = 200
        
        # Training parameters
        self.BATCH_SIZE = parser.getint('trainingParams', 'batchsize')
        self.MAX_EPOCHS = parser.getint('trainingParams', 'maxepoch')
        self.EARLY_STOPPING_PATIENCE = parser.getint('trainingParams', 'earlystoppingpatience')
        
        # Data splitting parameters
        self.TEST_SIZE = parser.getfloat('dataSplittingParams', 'testsize')
        self.VAL_SIZE = parser.getfloat('dataSplittingParams', 'valsize')
        self.RANDOM_STATE = parser.getint('dataSplittingParams', 'randomstate')
        
        print(f"Configuration loaded from {config_file}")
    
    def load_data(self):
        """Load cases from vitaldb and prepare input and target arrays"""
        caseids = vitaldb.find_cases(['ECG_II','PLETH', 'ART'])
        print(f"Found {len(caseids)} cases")
        
        self.X = []
        self.y = []
        
        # Select random cases
        random_cases = set([np.random.randint(0, len(caseids)) for _ in range(0, self.NUM_RANDOM_CASES)])
        
        for i in tqdm(random_cases):
            print(f"\nCase:{i}")
            vals = vitaldb.load_case(caseids[i], ['ECG_II','PLETH','ART'], 1/self.SAMPLING_RATE)
            try:
                ecg = vals[:,0]
                ppg = vals[:,1]
                art = vals[:,2]
                del vals
                
                # Remove NaN values
                na_indices = self._find_nan_indices(ecg, ppg, art)
                ecg = np.delete(ecg, na_indices)
                ppg = np.delete(ppg, na_indices)
                art = np.delete(art, na_indices)
                
                # Downsample arterial pressure
                art_slide = np.mean(sliding_window_view(art, self.DOWNSAMPLE_WINDOW)[::self.DOWNSAMPLE_STRIDE], axis=1)
                dup_art_windows = np.repeat(art_slide, self.DOWNSAMPLE_STRIDE)[0:len(art)]
                del art_slide, na_indices
                
                if (len(ecg)==len(art)) & (len(ppg)==len(art)) & (len(ecg)==len(dup_art_windows)):
                    self.X.append(np.array([ecg, ppg, dup_art_windows]))
                    self.y.append(art)
            except:
                pass
        
        print(f"Successfully loaded {len(self.X)} cases")
        return self
    
    def _find_nan_indices(self, ecg, ppg, art):
        """Helper method to find indices with NaN values"""
        na_indices = np.append(np.argwhere(np.isnan(ecg)), np.argwhere(np.isnan(ppg)), axis=0)
        na_indices = np.append(na_indices, np.argwhere(np.isnan(art)), axis=0)
        return np.unique(na_indices, axis=0)
    
    def visualize_raw_data(self):
        """Visualize the raw data before segmentation"""
        if len(self.X) < 1:
            print("No data to visualize")
            return self
            
        # Full signal visualization
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 5))
        for i in range(min(2, len(self.X))):
            ax[0, i].plot(self.X[i][0], label='ECG')
            ax[0, i].legend()
            ax[1, i].plot(self.X[i][1], label='PPG')
            ax[1, i].legend()
            ax[2, i].plot(self.y[i].squeeze(), label='ABP', color='green')
            ax[2, i].legend()
        plt.show()
        
        # Zoomed visualization
        fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 5))
        for i in range(min(2, len(self.X))):
            ax[0, i].plot(self.X[i][0][200000:201000], label='ECG')
            ax[0, i].legend()
            ax[1, i].plot(self.X[i][1][200000:201000], label='PPG')
            ax[1, i].legend()
            ax[2, i].plot(self.y[i].squeeze()[200000:201000], label='ABP', color='green')
            ax[2, i].legend()
        plt.show()
        
        # Combined visualization with mean ABP
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 5))
        for i in range(min(2, len(self.X))):
            ax[0, i].plot(self.X[i][0], label='ECG')
            ax[0, i].plot(self.X[i][1], label='PPG')
            ax[0, i].plot(self.X[i][2], label='Mean ABP')
            ax[0, i].legend()
            ax[1, i].plot(self.y[i].squeeze(), label='ABP', color='green')
            ax[1, i].legend()
        plt.show()
        
        # Detailed zoomed view
        fig, ax = plt.subplots(nrows=4, ncols=1, figsize=(20, 5))
        ax[0].plot(self.X[0][0][200000:201000], label='ECG')
        ax[0].legend()
        ax[1].plot(self.X[0][1][200000:201000], label='PPG')
        ax[1].legend()
        ax[2].plot(self.X[0][2][200000:201000], label='Mean ABP', color='orange')
        ax[2].legend()
        ax[3].plot(self.y[0].squeeze()[200000:201000], label='ABP', color='green')
        ax[3].legend()
        plt.show()
        
        return self
    
    def segment_data(self):
        """Segment data into windows for training"""
        X_segments = []
        y_segments = []
        
        for i in range(len(self.y)):
            X_sample = self.X[i]
            y_sample = self.y[i]
            
            # Process each window
            for j in range(0, min(self.MAX_SAMPLES_PER_CASE, len(y_sample)-self.WINDOW_SIZE+1), self.STEP_SIZE):
                X_window = X_sample[:, j:j+self.WINDOW_SIZE]
                y_window = y_sample[j:j+self.WINDOW_SIZE]
                
                # Check for anomalies in the data
                if self._is_valid_window(X_window, y_window):
                    # Pad sequences if needed
                    if X_window.shape[1] < self.WINDOW_SIZE:
                        X_window = np.pad(X_window, ((0, 0), (0, self.WINDOW_SIZE - X_window.shape[1])), mode='constant')
                        y_window = np.pad(y_window, (0, self.WINDOW_SIZE - len(y_window)), mode='constant')
                    
                    X_segments.append(X_window)
                    y_segments.append(y_window)
        
        self.X_segments = np.asarray(X_segments)
        self.y_segments = np.asarray(y_segments)
        
        print("X_segments shape:", self.X_segments.shape)
        print("y_segments shape:", self.y_segments.shape)
        
        # Clean up memory
        del self.X
        del self.y
        
        return self
    
    def _is_valid_window(self, X_window, y_window):
        """Check if a window contains valid data"""
        # Check for constant ECG values
        if np.all(X_window[0] == X_window[0][0]):
            return False
            
        # Check for anomalous values in signals
        if (np.any(X_window[0] < self.ECG_MIN_THRESHOLD) or 
            np.any(X_window[1] > self.PPG_MAX_THRESHOLD) or 
            np.any(X_window[1] < self.PPG_MIN_THRESHOLD) or 
            np.any(X_window[2] > self.ABP_MAX_THRESHOLD) or 
            np.any(X_window[2] < self.ABP_MIN_THRESHOLD)):
            return False
            
        # Check for constant or anomalous ABP values
        if (np.all(y_window == y_window[0]) or 
            np.any(y_window > self.ABP_MAX_THRESHOLD) or 
            np.any(y_window < self.ABP_MIN_THRESHOLD)):
            return False
            
        return True
    
    def split_data(self):
        """Split data into train, validation and test sets"""
        # Split into train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X_segments, self.y_segments, 
            test_size=self.TEST_SIZE, 
            random_state=self.RANDOM_STATE, 
            shuffle=True
        )
        
        # Split train into train and validation
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X_train, self.y_train, 
            test_size=self.VAL_SIZE, 
            random_state=self.RANDOM_STATE, 
            shuffle=True
        )
        
        print(f"Train: {len(self.X_train)}, Validation: {len(self.X_val)}, Test: {len(self.X_test)}")
        
        # Free memory
        del self.X_segments
        del self.y_segments
        
        return self
    
    def visualize_test_data(self):
        """Visualize segmented test data"""
        # Plot multiple samples
        num_samples = min(5, len(self.X_test))
        fig, ax = plt.subplots(nrows=4, ncols=num_samples, figsize=(20, 10))
        
        for i in range(num_samples):
            # Handle the case when only one sample is plotted
            if num_samples == 1:
                ax_col = [ax[0], ax[1], ax[2], ax[3]]  # List of axes
            else:
                ax_col = [ax[0, i], ax[1, i], ax[2, i], ax[3, i]]
                
            # The data is structured as [batch, channel, time]
            ax_col[0].plot(self.X_test[i][0], label='ECG')  # First channel is ECG
            ax_col[0].legend()
            ax_col[1].plot(self.X_test[i][1], label='PPG')  # Second channel is PPG
            ax_col[1].legend()
            ax_col[2].plot(self.X_test[i][2], label='Mean ABP')  # Third channel is Mean ABP
            ax_col[2].legend()
            ax_col[3].plot(self.y_test[i].squeeze(), label='ABP', color='green')
            ax_col[3].legend()
        plt.show()
        
        # Plot a single sample
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 4))
        i = min(0, len(self.X_test)-1)  # Sample index (ensuring it's within bounds)
        ax[0].plot(self.X_test[i][0], label='ECG')
        ax[0].legend()
        ax[1].plot(self.X_test[i][1], label='PPG')
        ax[1].legend()
        ax[2].plot(self.y_test[i].squeeze(), label='ABP', color='green')
        ax[2].legend()
        plt.show()
        
        return self
    
    def build_model(self):
        """Build the LSTM model"""
        try:
            # Attempt to use TPU if available
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            
            with strategy.scope():
                self.model = self._create_model()
        except:
            # Fall back to GPU/CPU
            try:
                with tf.device('/device:GPU:0'):
                    self.model = self._create_model()
            except:
                # Fall back to CPU
                self.model = self._create_model()
        
        self.model.summary()
        return self
    
    def _create_model(self):
        """Create the LSTM model architecture"""
        model = models.Sequential([
            layers.Input(shape=(3, self.X_train.shape[2])),
            layers.LSTM(64, return_sequences=True, dropout=0.2, recurrent_dropout=0.2,
                       kernel_regularizer=regularizers.l2(0.01)),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Flatten(),
            layers.Dense(self.X_train.shape[2], activation='linear')
        ])
        
        model.compile(optimizer=optimizers.Adam(clipnorm=1.0), loss='mse')
        return model
    
    def train_model(self):
        """Train the model"""
        if self.model is None:
            self.build_model()
            
        # Early stopping callback
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=self.EARLY_STOPPING_PATIENCE, 
            verbose=1, 
            mode='min'
        )
        
        # Train the model
        self.history = self.model.fit(
            self.X_train, self.y_train, 
            epochs=self.MAX_EPOCHS, 
            steps_per_epoch=len(self.X_train)//self.BATCH_SIZE, 
            batch_size=self.BATCH_SIZE,
            validation_data=(self.X_val, self.y_val), 
            validation_steps=len(self.X_val)//self.BATCH_SIZE,
            callbacks=[early_stopping]
        )
        
        return self
    
    def evaluate_model(self):
        """Evaluate the model on test data"""
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        test_loss = self.model.evaluate(self.X_test, self.y_test)
        overall_accuracy = 1 - tf.reduce_mean(tf.abs(self.y_pred - self.y_test) / self.y_test)
        rmse = math.sqrt(mean_squared_error(self.y_test, self.y_pred))
        mape = self._calculate_mape(self.y_test, self.y_pred)
        
        # Print metrics
        print('Test loss:', test_loss)
        print('Overall accuracy:', overall_accuracy.numpy())
        print('Root Mean Square Error:', rmse)
        print('MAPE:', mape, '%')
        
        return self
    
    def _calculate_mape(self, actual, predicted):
        """Calculate Mean Absolute Percentage Error"""
        if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
            actual, predicted = np.array(actual), np.array(predicted)
        return round(np.mean(np.abs((actual - predicted) / actual)) * 100, 2)
    
    def plot_training_history(self):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.history.history['loss'], label='Training loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        return self
    
    def visualize_predictions(self):
        """Visualize model predictions against ground truth"""
        # Plot multiple samples
        num_samples = min(5, len(self.X_test))
        fig, ax = plt.subplots(nrows=3, ncols=num_samples, figsize=(20, 10))
        
        for j in range(num_samples):
            i = np.random.randint(0, self.X_test.shape[0])
            
            # Handle the case when only one sample is plotted
            if num_samples == 1:
                ax_col = [ax[0], ax[1], ax[2]]  # List of axes
            else:
                ax_col = [ax[0, j], ax[1, j], ax[2, j]]
                
            ax_col[0].plot(self.X_test[i][0], label='ECG')  # First channel is ECG
            ax_col[0].legend()
            ax_col[1].plot(self.X_test[i][1], label='PPG')  # Second channel is PPG
            ax_col[1].legend()
            ax_col[2].plot(self.y_test[i].squeeze(), label='Target ABP')
            ax_col[2].plot(self.y_pred[i].squeeze(), label='Predicted ABP')
            ax_col[2].legend()
        plt.show()
        
        # Plot a single sample in detail
        fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(20, 5))
        i = min(0, len(self.X_test)-1)  # Sample index (ensuring it's within bounds)
        ax[0].plot(self.X_test[i][0], label='ECG')
        ax[1].plot(self.X_test[i][1], label='PPG')
        ax[0].legend()
        ax[1].legend()
        ax[2].plot(self.y_test[i].squeeze(), label='Target ABP', color='green')
        ax[2].plot(self.y_pred[i].squeeze(), label='Predicted ABP', color='red')
        ax[2].legend()
        plt.show()
        
        return self
    
    def run_pipeline(self):
        """Run the complete analysis pipeline"""
        return (self
                .load_data()
                .visualize_raw_data()
                .segment_data()
                .split_data()
                .visualize_test_data()
                .build_model()
                .train_model()
                .evaluate_model()
                .plot_training_history()
                .visualize_predictions())
