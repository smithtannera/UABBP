"""
Model definition for ABP prediction
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers, optimizers
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.metrics import mean_squared_error


class ABPModel:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.history = None
    
    def build_model(self, input_shape):
        """
        Build the model architecture with Bidirectional LSTM
        """
        try:
            # Check for TPU availability
            tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(tpu)
            tf.tpu.experimental.initialize_tpu_system(tpu)
            strategy = tf.distribute.TPUStrategy(tpu)
            print("Using TPU strategy")
            
            # Define model in TPU strategy scope
            with strategy.scope():
                self.model = self._create_model(input_shape)
        except:
            # Fall back to GPU if TPU not available
            print("TPU not available, falling back to GPU/CPU")
            with tf.device('/device:GPU:0'):
                self.model = self._create_model(input_shape)
                
        return self
    
    def _create_model(self, input_shape):
        """
        Create the model architecture with Bidirectional LSTM
        """
        # Main model for arterial waveform prediction
        inputs = layers.Input(shape=input_shape)
        
        # Bidirectional LSTM layers (replacing standard LSTM)
        x = layers.Bidirectional(
            layers.LSTM(
                self.config.LSTM_UNITS, 
                return_sequences=True,
                dropout=self.config.DROPOUT_RATE,
                recurrent_dropout=self.config.RECURRENT_DROPOUT,
                kernel_regularizer=regularizers.l2(self.config.L2_REG)
            )
        )(inputs)
        
        # Dense layer with activation
        x = layers.Dense(self.config.DENSE_UNITS, activation='relu')(x)
        x = layers.Dropout(self.config.OUTPUT_DROPOUT)(x)
        x = layers.Flatten()(x)
        
        # Output for arterial waveform
        waveform_output = layers.Dense(input_shape[1], activation='linear', name='waveform_output')(x)
        
        # Additional outputs for systolic and diastolic pressures
        systolic_output = layers.Dense(1, activation='linear', name='systolic_output')(x)
        diastolic_output = layers.Dense(1, activation='linear', name='diastolic_output')(x)
        
        # Create multi-output model
        model = models.Model(
            inputs=inputs, 
            outputs=[waveform_output, systolic_output, diastolic_output]
        )
        
        # Compile with different loss weights
        model.compile(
            optimizer=optimizers.Adam(clipnorm=1.0),
            loss={
                'waveform_output': 'mse',
                'systolic_output': 'mse',
                'diastolic_output': 'mse'
            },
            loss_weights={
                'waveform_output': 1.0,
                'systolic_output': 0.5,
                'diastolic_output': 0.5
            }
        )
        
        return model
    
    def train(self, X_train, y_train_wave, y_train_sys, y_train_dia, 
              X_val, y_val_wave, y_val_sys, y_val_dia):
        """
        Train the model
        """
        # Set up early stopping
        early_stopping = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=self.config.EARLY_STOPPING_PATIENCE,
            verbose=1,
            mode='min'
        )
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // self.config.BATCH_SIZE
        validation_steps = len(X_val) // self.config.BATCH_SIZE
        
        # Train the model
        self.history = self.model.fit(
            X_train,
            {
                'waveform_output': y_train_wave,
                'systolic_output': y_train_sys,
                'diastolic_output': y_train_dia
            },
            epochs=self.config.MAX_EPOCHS,
            batch_size=self.config.BATCH_SIZE,
            steps_per_epoch=steps_per_epoch,
            validation_data=(
                X_val,
                {
                    'waveform_output': y_val_wave,
                    'systolic_output': y_val_sys,
                    'diastolic_output': y_val_dia
                }
            ),
            validation_steps=validation_steps,
            callbacks=[early_stopping]
        )
        
        return self
    
    def evaluate(self, X_test, y_test_wave, y_test_sys, y_test_dia):
        """
        Evaluate the model
        """
        # Make predictions
        y_pred_wave, y_pred_sys, y_pred_dia = self.model.predict(X_test)
        
        # Calculate metrics for waveform
        test_loss = self.model.evaluate(
            X_test,
            {
                'waveform_output': y_test_wave,
                'systolic_output': y_test_sys,
                'diastolic_output': y_test_dia
            },
            verbose=0
        )
        
        # Calculate accuracy between target and predicted waveforms
        waveform_accuracy = 1 - np.mean(np.abs(y_pred_wave - y_test_wave) / np.clip(np.abs(y_test_wave), 1e-7, None))
        
        # Calculate RMSE for waveform
        waveform_rmse = math.sqrt(mean_squared_error(y_test_wave, y_pred_wave))
        
        # Calculate MAPE for waveform
        waveform_mape = self._calculate_mape(y_test_wave, y_pred_wave)
        
        # Calculate metrics for systolic and diastolic
        sys_rmse = math.sqrt(mean_squared_error(y_test_sys, y_pred_sys))
        dia_rmse = math.sqrt(mean_squared_error(y_test_dia, y_pred_dia))
        
        sys_mape = self._calculate_mape(y_test_sys, y_pred_sys)
        dia_mape = self._calculate_mape(y_test_dia, y_pred_dia)
        
        # Print evaluation results
        print("Test loss:", test_loss[0])
        print("Waveform accuracy:", waveform_accuracy)
        print("Waveform RMSE:", waveform_rmse)
        print("Waveform MAPE:", waveform_mape, "%")
        print("Systolic RMSE:", sys_rmse)
        print("Systolic MAPE:", sys_mape, "%")
        print("Diastolic RMSE:", dia_rmse)
        print("Diastolic MAPE:", dia_mape, "%")
        
        # Store predictions for plotting
        self.y_test_wave = y_test_wave
        self.y_pred_wave = y_pred_wave
        self.y_test_sys = y_test_sys
        self.y_pred_sys = y_pred_sys
        self.y_test_dia = y_test_dia
        self.y_pred_dia = y_pred_dia
        self.X_test = X_test
        
        return self
    
    def _calculate_mape(self, actual, predicted) -> float:
        """
        Calculate Mean Absolute Percentage Error
        """
        # Convert to numpy arrays
        if not all([isinstance(actual, np.ndarray), isinstance(predicted, np.ndarray)]):
            actual, predicted = np.array(actual), np.array(predicted)
        
        # Avoid division by zero
        mask = actual != 0
        return round(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100, 2)
    
    def plot_training_history(self):
        """
        Plot training and validation loss
        """
        plt.figure(figsize=(12, 6))
        
        # Plot the main loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history.history['loss'], label='Training loss')
        plt.plot(self.history.history['val_loss'], label='Validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Total Loss')
        plt.legend()
        plt.title('Total Loss Over Time')
        
        # Plot component losses if available
        if 'waveform_output_loss' in self.history.history:
            plt.subplot(1, 2, 2)
            plt.plot(self.history.history['waveform_output_loss'], label='Waveform loss')
            plt.plot(self.history.history['systolic_output_loss'], label='Systolic loss')
            plt.plot(self.history.history['diastolic_output_loss'], label='Diastolic loss')
            plt.xlabel('Epoch')
            plt.ylabel('Component Loss')
            plt.legend()
            plt.title('Component Losses Over Time')
        
        plt.tight_layout()
        plt.show()
        
        return self
    
    def plot_predictions(self, num_samples=5):
        """
        Plot predictions against actual values
        """
        # Select random samples
        import random
        random.seed(self.config.RANDOM_SEED)
        sample_indices = random.sample(range(self.X_test.shape[0]), min(num_samples, self.X_test.shape[0]))
        
        # Create figure with subplots
        fig, axes = plt.subplots(3, num_samples, figsize=(20, 10))
        
        for j, i in enumerate(sample_indices):
            # Plot ECG and PPG for context
            axes[0, j].plot(self.X_test[i, 0][100:1000], label='ECG')
            axes[0, j].set_title(f'Sample {j+1}')
            axes[0, j].legend()
            
            axes[1, j].plot(self.X_test[i, 1][100:1000], label='PPG')
            axes[1, j].legend()
            
            # Plot arterial blood pressure waveform
            axes[2, j].plot(self.y_test_wave[i][100:1000], label='Target ABP', alpha=0.7)
            axes[2, j].plot(self.y_pred_wave[i][100:1000], label='Predicted ABP', alpha=0.7)
            
            # Add systolic/diastolic markers
            axes[2, j].axhline(y=self.y_test_sys[i], color='r', linestyle='--', alpha=0.5, label=f'True Sys: {self.y_test_sys[i]:.1f}')
            axes[2, j].axhline(y=self.y_test_dia[i], color='b', linestyle='--', alpha=0.5, label=f'True Dia: {self.y_test_dia[i]:.1f}')
            axes[2, j].axhline(y=self.y_pred_sys[i], color='r', linestyle=':', alpha=0.5, label=f'Pred Sys: {self.y_pred_sys[i][0]:.1f}')
            axes[2, j].axhline(y=self.y_pred_dia[i], color='b', linestyle=':', alpha=0.5, label=f'Pred Dia: {self.y_pred_dia[i][0]:.1f}')
            
            axes[2, j].set_ylim(self.config.BP_MIN_THRESHOLD, self.config.BP_MAX_THRESHOLD)
            axes[2, j].legend(loc='upper right', fontsize='x-small')
        
        plt.tight_layout()
        plt.show()
        
        return self
