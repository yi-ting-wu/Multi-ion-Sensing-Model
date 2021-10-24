# -*- coding: utf-8 -*-
"""
@github: https://github.com/yi-ting-wu/Multi-ion-Sensing-Model

@author: Yi-Ting Wu
"""

from keras.layers import Input, Dense, Activation, Conv2D, MaxPool2D, regularizers, Flatten, GlobalAvgPool2D, Dropout
from keras.models import Model
from keras import optimizers


def CNN (input_shape):

    data_in = Input(shape=input_shape)

    # Create Conv layer_1
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(data_in)
    x = Activation('relu')(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Create Conv layer_2
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)
    
    # Create Conv layer_3
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Activation('relu')(x)
    x = Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='same', 
               kernel_regularizer=regularizers.l2(0.0001))(x)
    x = Activation('relu')(x)
    x = MaxPool2D(pool_size=(2, 2))(x)

    # Create GAP layer
    x = GlobalAvgPool2D()(x)
    
    # Create Dropout layer
    x = Dropout(0.5)(x)
    
    # Create Dense layer
    x = Dense(64, activation='relu', kernel_initializer='uniform')(x)
    
    # Create Dropout layer
    x = Dropout(0.3)(x)
    
    # Create Output layer
    data_out = Dense(4, kernel_initializer='uniform')(x)
    
    # Summary 
    model = Model(inputs=data_in, outputs=data_out)
    model.summary()
    
    # Optimizer parameter
    adam = optimizers.adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=True)
    model.compile(loss='mse', optimizer=adam) 

    return model