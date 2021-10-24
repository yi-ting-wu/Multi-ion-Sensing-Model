[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# An Augmented-FET Sensor for Multi-ion Recognition with Machine Learning

A [Keras](https://keras.io/) implementation of An Augmented-FET Sensor for Multi-ion Recognition with Machine Learning. 

This implementation of an algorithm developed by the [
NTU CMOS Biotechnology Lab](https://sites.google.com/site/cmosbiotechnology/home).


&nbsp;

## Requirements
- Python 3.7.9
- Keras 2.2.4
- numpy, sklearn, pickle
- ...

&nbsp;

## Modeling 
- The script of modeling is `CNN_model.py` :

    ```python
    from CNN_model import CNN
    
    model = CNN(input_shape)
    ```

- Note:
  * Please confirm a script is on the same path before importing the module.
  * The **input_shape** for the paper is **(11, 11, 1)**.

&nbsp;

## Training 
- The script of training is `Train.py` :

    ```python
    from Train import Training
    
    t = Training(size, number, name)
    
    t.train_model()
    ```

- The **standardization scaler** and **pre-trained model** can be obtained :

    ```
    std_scaler.pkl
    model.h5
    ```

- If you want to use **GPU** for training :

  * The study used **AMD GPU(RX570)** to train the model, 
  
    but the setting is complicated, hence, it's recommended to use CPU or Nvidia GPU.
    
  * The following is a tutorial using **AMD GPU** to train the model :
  
    ```python
    pip install plaidml-keras plaidbench
    ```
    ```python
    plaidml-setup
    ```
    Choose which accelerator you'd like to use.
    
    **PlaidML** has to be running under `keras==2.2.4`.
    ```python
    from Train import Training
    
    t = Training(size, number, name)
    
    t.gpu() 
    
    t.train_model()
    ```  

- Note:
  * Please confirm all scripts are on the same path before importing the module.
  * The **size**, **number**, and **name** for the paper are **11**, **7650**, and **'train.npy'**.

&nbsp;

## Testing 
- The script of testing is `Test.py` :

    ```python
    from Test import Testing
    
    t = Testing(size, number, set_name)
    
    t.test_model(model_name)
    ```

- The **inference result** can be obtained :

    ```
    result.npy
    ```

- Note:
  * Please confirm all scripts, the standardization scaler, and the pre-trained model are on the same path 
    
    before importing the module.
  * The **size**, **number**, and **set_name** for the paper are **11**, **1680**, and **'test.npy'**.
  * The **model_name** for the paper is **'model.h5'**.

&nbsp;


## Datasets
The data that support the findings of this study are available from the corresponding author of the paper upon reasonable request.
