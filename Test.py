# -*- coding: utf-8 -*-
"""
@github: https://github.com/yi-ting-wu/Multi-ion-Sensing-Model

@author: Yi-Ting Wu
"""

import numpy as np
import pickle
import keras
from keras.models import load_model


class Testing:
    
    def __init__(self, size, num, name):
        self.size = size
        self.num = num
        self.name = name
        print('Size: ({}, {})'.format(self.size, self.size))
        print('Number: ', self.num)
        print('Loading: ', name)    
    
    def load_data(self):
        """
        Testing set defaults to the data and labels in the same file
        """    
        # Load data
        test = np.load(self.name)        
        
        # Split label
        x_test = np.delete(test,[0, 1, 2, 3], axis=1)
        y_test = test[:, 0:4]
        
        # Z-score standardization
        std = pickle.load(open('std_scaler.pkl', 'rb')) # Load scaler 
        x_test_std = std.transform(x_test)
        
        # Reshape
        x_test_std = x_test_std.reshape(self.num, self.size, self.size)      
        x_test_std = x_test_std.reshape(x_test_std.shape[0], self.size, self.size, 1).astype('float32') 
        
        return x_test_std, y_test

    def test_model(self, name):
        # Input data
        x_test_std, y_test = Testing.load_data(self)
        
        # Load 
        model = load_model(name)
        print(model.summary())
        
        # Infer
        y_pred = model.predict(x_test_std)
        
        # Save
        np.save('result.npy', y_pred) 

        return y_pred