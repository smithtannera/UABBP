"""
Data processing utilities for ABP prediction
"""
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import vitaldb
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.X = []
        self.y_mean = []
        self.y_systolic = []
        self.y_diastolic = []
        
    def find_blood_pressure_values(self, bp_window):
        """
        Extract systolic and diastolic values from a blood pressure window
        """
        if len(bp_window) == 0:
            return 0, 0
            
        # Get systolic (max) and diastolic (min) values
        systolic = np.max(bp_window)
        diastolic = np.min(bp_window)
        
        return systolic, diastolic
        
    def load_data(self):
        """
        Load and process data from VitalDB
        """
        caseids = vitaldb.find_cases(['ECG_II', 'PLETH', 'ART'])
        
        # Use random.sample for selecting random cases without replacement
        import random
        random.seed(self.config.RANDOM_SEED)
        selected_indices = random.sample(range(len(caseids)), min(self.config.NUM_SAMPLES, len(caseids)))
        
        for i in tqdm(selected_indices):
            print(f"\nCase: {i}")
            vals = vitaldb.load_case(caseids[i], ['ECG_II', 'PLETH', 'ART'], 1/100)
            try:
                ecg = vals[:, 0]
                ppg = vals[:, 1]
                art = vals[:, 2]
                del vals
                
                # Clean NaN values
                na_indices = np.append(np.argwhere(np.isnan(ecg)), np.argwhere(np.isnan(ppg)), axis=0)
                na_indices = np.append(na_indices, np.argwhere(np.isnan(art)), axis=0)
                na_indices = np.unique(na_indices, axis=0)
                ecg = np.delete(ecg, na_indices)
                ppg = np.delete(ppg, na_indices)
                art = np.delete(art, na_indices)
                
                # Calculate mean arterial pressure using sliding window
                art_windows = sliding_window_view(art, self.config.SAMPLE_WINDOW_SIZE)[::self.config.STEP_SIZE]
                art_mean = np.mean(art_windows, axis=1)
                
                # Calculate systolic and diastolic values for each window
                art_systolic = np.max(art_windows, axis=1)
                art_diastolic = np.min(art_windows, axis=1)
                
                # Duplicate values to match original length
                dup_art_mean = np.repeat(art_mean, self.config.STEP_SIZE + self.config.SAMPLE_WINDOW_SIZE - 1)[0:len(art)]
                dup_art_systolic = np.repeat(art_systolic, self.config.STEP_SIZE + self.config.SAMPLE_WINDOW_SIZE - 1)[0:len(art)]
                dup_art_diastolic = np.repeat(art_diastolic, self.config.STEP_SIZE + self.config.SAMPLE_WINDOW_SIZE - 1)[0:len(art)]
                
                del art_mean, art_systolic, art_diastolic, na_indices
                
                if (len(ecg) == len(art)) and (len(ppg) == len(art)) and (len(ecg) == len(dup_art_mean)):
                    self.X.append(np.array([ecg, ppg, dup_art_mean, dup_art_systolic, dup_art_diastolic]))
                    self.y_mean.append(art)
                    # Store original arterial waveform for both systolic and diastolic 
                    # as we'll extract actual values during segmentation
                    self.y_systolic.append(art)
                    self.y_diastolic.append(art)
            except Exception as e:
                print(f"Error processing case {i}: {e}")
                pass
                
        print(f"Loaded {len(self.X)} cases successfully")
        return self
        
    def plot_samples(self, num_samples=2):
        """
        Plot samples of the loaded data
        """
        if len(self.X) < num_samples:
            num_samples = len(self.X)
            
        fig, ax = plt.subplots(nrows=3, ncols=num_samples, figsize=(20, 5))
        
        for i in range(num_samples):
            ax[0, i].plot(self.X[i][0][100000:101000], label='ECG')
            ax[0, i].legend()
            ax[1, i].plot(self.X[i][1][100000:101000], label='PPG')
            ax[1, i].legend()
            ax[2, i].plot(self.X[i][2][100000:101000], label='Mean ABP', color='green')
            ax[2, i].plot(self.X[i][3][100000:101000], label='Systolic', color='red')
            ax[2, i].plot(self.X[i][4][100000:101000], label='Diastolic', color='blue')
            ax[2, i].plot(self.y_mean[i][100000:101000], label='ABP', color='orange')
            ax[2, i].legend()
        
        plt.show()
        return self
    
    def segment_data(self):
        """
        Segment data into windows for training
        """
        X_segments = []
        y_mean_segments = []
        y_systolic_segments = []
        y_diastolic_segments = []
        
        for i in range(len(self.y_mean)):
            X_sample = self.X[i]
            y_mean_sample = self.y_mean[i]
            y_systolic_sample = self.y_systolic[i]
            y_diastolic_sample = self.y_diastolic[i]
            
            for j in range(0, min(self.config.MAX_SEGMENT_RANGE, len(y_mean_sample) - self.config.WINDOW_SIZE), self.config.STEP):
                X_window = X_sample[:, j:j+self.config.WINDOW_SIZE]
                y_mean_window = y_mean_sample[j:j+self.config.WINDOW_SIZE]
                y_systolic_window = y_systolic_sample[j:j+self.config.WINDOW_SIZE]
                y_diastolic_window = y_diastolic_sample[j:j+self.config.WINDOW_SIZE]
                
                # Skip windows with constant or out-of-range values
                if (np.all(X_window[0] == X_window[0][0])) or \
                   (np.any(X_window[0] < self.config.ECG_MIN_THRESHOLD)) or \
                   (np.any(X_window[1] > self.config.PPG_MAX_THRESHOLD)) or \
                   (np.any(X_window[1] < self.config.PPG_MIN_THRESHOLD)) or \
                   (np.any(X_window[2] > self.config.BP_MAX_THRESHOLD)) or \
                   (np.any(X_window[2] < self.config.BP_MIN_THRESHOLD)) or \
                   (np.all(y_mean_window == y_mean_window[0])) or \
                   (np.any(y_mean_window > self.config.BP_MAX_THRESHOLD)) or \
                   (np.any(y_mean_window < self.config.BP_MIN_THRESHOLD)):
                    pass
                else:
                    # Pad sequences if needed
                    if X_window.shape[1] < self.config.WINDOW_SIZE:
                        pad_len = self.config.WINDOW_SIZE - X_window.shape[1]
                        X_window = np.pad(X_window, ((0, 0), (0, pad_len)), mode='constant')
                        y_mean_window = np.pad(y_mean_window, (0, pad_len), mode='constant')
                        y_systolic_window = np.pad(y_systolic_window, (0, pad_len), mode='constant')
                        y_diastolic_window = np.pad(y_diastolic_window, (0, pad_len), mode='constant')
                    
                    # Calculate systolic and diastolic for this window
                    systolic = np.max(y_systolic_window)
                    diastolic = np.min(y_diastolic_window)
                    
                    # Append segments
                    X_segments.append(X_window[:3])  # Only use ECG, PPG, and mean ABP as input
                    y_mean_segments.append(y_mean_window)
                    y_systolic_segments.append(systolic)  # Store scalar systolic value
                    y_diastolic_segments.append(diastolic)  # Store scalar diastolic value
                    
                    del X_window, y_mean_window, y_systolic_window, y_diastolic_window
        
        # Convert to numpy arrays
        self.X_segments = np.asarray(X_segments)
        self.y_mean_segments = np.asarray(y_mean_segments)
        self.y_systolic_segments = np.asarray(y_systolic_segments)
        self.y_diastolic_segments = np.asarray(y_diastolic_segments)
        
        print("X_segments shape:", self.X_segments.shape)
        print("y_mean_segments shape:", self.y_mean_segments.shape)
        print("y_systolic_segments shape:", self.y_systolic_segments.shape)
        print("y_diastolic_segments shape:", self.y_diastolic_segments.shape)
        
        del self.X, self.y_mean, self.y_systolic, self.y_diastolic
        return self
    
    def split_data(self):
        """
        Split data into train, validation, and test sets
        """
        # First split into train+val and test
        X_train_val, X_test, y_mean_train_val, y_mean_test, y_sys_train_val, y_sys_test, y_dia_train_val, y_dia_test = train_test_split(
            self.X_segments, self.y_mean_segments, self.y_systolic_segments, self.y_diastolic_segments, 
            test_size=self.config.TEST_SPLIT, random_state=self.config.RANDOM_SEED, shuffle=True
        )
        
        # Then split train+val into train and val
        X_train, X_val, y_mean_train, y_mean_val, y_sys_train, y_sys_val, y_dia_train, y_dia_val = train_test_split(
            X_train_val, y_mean_train_val, y_sys_train_val, y_dia_train_val,
            test_size=self.config.VALIDATION_SPLIT, random_state=self.config.RANDOM_SEED, shuffle=True
        )
        
        # Clean up intermediate variables
        del self.X_segments, self.y_mean_segments, self.y_systolic_segments, self.y_diastolic_segments
        del X_train_val, y_mean_train_val, y_sys_train_val, y_dia_train_val
        
        # Store the splits
        self.X_train, self.X_val, self.X_test = X_train, X_val, X_test
        self.y_mean_train, self.y_mean_val, self.y_mean_test = y_mean_train, y_mean_val, y_mean_test
        self.y_sys_train, self.y_sys_val, self.y_sys_test = y_sys_train, y_sys_val, y_sys_test
        self.y_dia_train, self.y_dia_val, self.y_dia_test = y_dia_train, y_dia_val, y_dia_test
        
        print(f"Training set: {len(self.X_train)} samples")
        print(f"Validation set: {len(self.X_val)} samples")
        print(f"Test set: {len(self.X_test)} samples")
        
        return self
