# -*- coding: utf-8 -*-
"""
@github: https://github.com/yi-ting-wu/Multi-ion-Sensing-Model

@author: Yi-Ting Wu
"""

import CNN_model
import numpy as np
import pickle
import os
import keras
from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler


class Training:
    
    def __init__(self, size, num, name):
        self.size = size
        self.num = num
        self.name = name
        print('Size: ({}, {})'.format(self.size, self.size))
        print('Number: ', self.num)
        print('Loading: ', name)
    
    def load_data(self):
        """
        Training set defaults to the data and labels in the same file
        """    
        # Load data
        train = np.load(self.name)

        # Split label
        x_train = np.delete(train,[0, 1, 2, 3], axis=1)
        y_train = train[:, 0:4]

        # Z-score standardization
        std = preprocessing.StandardScaler().fit(x_train)
        x_train_std = std.transform(x_train)    
        pickle.dump(std, open('std_scaler.pkl','wb')) # Save scaler 
    
        # Reshape
        x_train_std = x_train_std.reshape(self.num, self.size, self.size)
        x_train_std = x_train_std.reshape(x_train_std.shape[0], self.size, self.size, 1).astype('float32') 

        return x_train_std, y_train

    def gpu(self):
        os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
        print ('device = GPU\n')

    def train_model(self, epoch_num=500, batch_size=32):
        """
        The epoch number and batch size default to '500' and '32'
        and 
        split '30%' as a validation set
        """
        # Input data
        x_train_std, y_train = Training.load_data(self)
      
        # Build 
        input_shape = (self.size, self.size, 1)
        model = CNN_model.CNN(input_shape)
        
        # Train
        model.fit(x=x_train_std, 
                  y=y_train, 
                  validation_split=0.3,
                  epochs=epoch_num, 
                  batch_size=batch_size, 
                  shuffle=True, verbose=1)

        # Save
        model.save('model.h5')
        
        return model