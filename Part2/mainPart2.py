#####            Importing the Libraries            #####
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers , models, regularizers, optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm
import vitaldb
import math
import configparser
from tensorflow.keras.preprocessing.sequence import pad_sequences
from abpPredictorModulePart2 import abpPredictor


def main():
    
    # Create an instance of the abpPredictor class
    # and run the full sequence.
    abp_predictor = abpPredictor()
    abp_predictor.run_pipeline()



if __name__ == "__main__":
    
    # Run the main function
    main()    


# def readConfig():
    
#     config = configparser.ConfigParser()

#     # Read the configuration file in same directory
#     config.read('configPart2.ini')

#     # Access values from the configuration file
#     numRandomCases = config.get('dataLoadingParams','numRandomCases')
#     samplingRate = config.get('dataLoadingParams','samplingRate')
    
#     windowSize = config.get('signalProcessingParams','windowSize')
#     stepSize = config.get('signalProcessingParams','stepSize')
   
#     downWindow = config.get('downsampling','downWindow')
#     downsampleSkip = config.get('downsampling','downsampleSkip')
    
#     batchSize = config.get('trainingParams','batchSize')
#     maxEpoch = config.get('trainingParams','maxEpoch')
#     earlyStoppingPatience = config.get('trainingParams','earlyStoppingPatience')

#     testSize = config.get('dataSplittingParams', 'testSize')
#     valSize = config.get('dataSplittingParams', 'valSize')
#     randomState = config.get('dataSplittingParams', 'randomState')

#     # Return a dictionary with the retrieved values
#     config_values = {
#         'numRandomCases': numRandomCases,
#         'samplingRate': samplingRate,
#         'windowSize': windowSize,
#         'stepSize': stepSize,
#         'downWindow': downWindow,
#         'downsampleSkip': downsampleSkip,
#         'batchSize': batchSize,
#         'maxEpoch': maxEpoch,
#         'earlyStoppingPatience': earlyStoppingPatience,
#         'testSize': testSize,
#         'valSize': valSize,
#         'randomState': randomState
#     }

#     return config_values
