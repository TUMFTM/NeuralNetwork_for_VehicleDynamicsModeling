# Neural Network for Vehicle Dynamics Modeling

This repository provides a neural network training algorithm which is able to substitute a physics-based single-track ("bicycle") model for vehicle dynamics simulation.

## Introduction

Autonomous vehicles have to meet high safety standards
in order to be commercially viable. Before real-world
testing of an autonomous vehicle, extensive simulation is required
to verify software functionality and to detect unexpected behavior.
Until now, vehicle
dynamics simulation/ estimation has mostly been performed with physics-based
models. Whereas these models allow specific effects to
be implemented, accurate models need a variety of parameters.
Their identification requires costly resources, e.g., expensive test
facilities. Machine learning models enable new approaches to
perform these modeling tasks without the necessity of identifying
parameters. Neural networks can be trained with recorded vehicle
data to represent the vehicle’s dynamic behavior.

Our model is trained to predict the vehicle state of the next (simulation) timestep by using the current vehicle control input and the current vehicle state plus the vehicle states of the past four timesteps.

![overview vehicle dynamics model](/resources/overview.png)

## List of components
* `helper_funcs_NN`: This package contains helper functions used in several other functions when running the main script for neural network training.
* `inputs`: This folder contains pre-trained neural network models and/ or vehicle sensor data to train a neural network (or re-train an existing one).
* `outputs`: This folder contains the simulation results, figures, the trained neural network, a backup of the used parameters and information on how the input/ output data was scaled before and re-scaled after training.
* `params`: This folder contains a parameter file to manage all settings.
* `src`: This folder contains the algorithms necessary to train and validate the neural network.
* `visualization`: This folder contains algorithms to visualize the training and validation process of the neural network.
* `main_NN_Vehicle_dynamics.py`: This script starts the training and validation process.


## Dependencies
Use the provided `requirements.txt` to install all required modules. Run:\
``pip3 install -r /path/to/requirements.txt``

The code is developed and tested with Python 3.7.


## Input data information
The input data to train and test the neural network has to be located in `/inputs/trainingdata/` as `.csv` files.

The `.csv` files contain the following information (one per column):
* `vx_mps`: longitudinal velocity of vehicle CoG, meter/second,
* `vy_mps`: lateral velocity of vehicle CoG, meter/second,
* `dpsi_radps`: yaw rate, rad/second, vehicle turning left is positive
* `ax_mps2`: longitudinal acceleration, meter/second²
* `ay_mps2`: lateral acceleration, meter/second²
* `deltawheel_rad`: steering angle of both front wheels (averaged), rad
* `TwheelRL_Nm`: wheel torque at rear left, Nm
* `TwheelRR_Nm`: wheel torque at rear right, Nm
* `pBrakeF_bar`: brake pressure at front, bar
* `pBrakeR_bar`: brake pressure at rear, bar

**Note:**\
The provided data is exemplary and generated from an existing [vehicle dynamics simulation](https://github.com/TUMFTM/sim_vehicle_dynamics). For the application described in the paper, we did use real vehicle sensor data instead.

### Training data
The training data files contain vehicle data which is used to train the neural network in order to model the vehicle's dynamic behavior. All training data filenames must start with ``data_to_train`` (e.g. "data_to_train_run1.csv"). All files located in this folder and starting with this prefix are then used for training the neural network.
The data format is described above.

### Test data
The test data file contains vehicle data which is used to test the trained neural network. The vehicle sensor data provided in this file is compared against the neural network's vehicle state prediction when receiving the same inputs as the real vehicle does.
The vehicle input from the provided file (steering angle, torque and brake pressure) is applied to the neural network and the output (vehicle state) is compared to the actual vehicle state from the provided test file.

The name of the test data file is ``data_to_run.csv``.
The data format is equal to the ``data_to_train`` files and is described above.


## Running the code:

The training process of the neural network has two different modes (switch via parameter settings):
* Mode 1 --> Feedforward Model
* Mode 2 --> Recurrent Model (GRU, LSTM, ...)

### Training a new NN model
Following steps are necessary to run the training process:
1. Open `/params/parameters.toml` and set parameters (you can find good hyperparameters to start with in the instructions below).
2. Set parameter ``model_mode`` in section ``NeuralNetwork_Settings`` to the Neural Network type which should be used (0 --> No training, 1 --> Feedforward, 2--> Recurrent).
3. Set optimizer parameters in section ``NeuralNetwork_Settings.Optimizer``.
4. Optional: Change the NN model architecture in ``src/neural_network_fcn.py``, e.g. add a layer.
5. Run ``main_NN_Vehicle_dynamics.py``.
6. The results will be saved in the ``\outputs`` folder.

**Note:**\
You can train the NN and subsequently test it at once. Set both parameters ``model_mode`` and ``run_mode`` to 1 or 2, respectively.

### Retrain an existing NN model
An already existing model can be retrained on new training data. Therefore, an existing model has to be provided in ``/inputs/trainedmodels/``.

1. Set parameter ``model_mode`` in section ``NeuralNetwork_Settings`` to 1 or 2, respectively.
2. Set parameter ``bool_load_existingmodel`` in section ``General`` to True.
3. Copy new training data into ``/inputs/trainingdata/`` with the above mentioned naming.
4. Run ``main_NN_Vehicle_dynamics.py``.
5. The results will be saved in the ``\outputs`` folder.

### Run a test against real vehicle data
The test mode can be run independently of the model training. Therefore, an already trained model has to be provided in ``/inputs/trainedmodels/``.
Copy both files ``keras_model.h5`` and ``scaler.plk`` into ``/inputs/trainedmodels/``. The filenames must be retained.
These files are generated as a result of the training process and are save in ``/outputs``.

1. Set parameter ``model_mode`` in section ``NeuralNetwork_Settings`` to 0 and ``run_mode`` to 1 or 2, respectively (depending on which NN should be used for testing: 1 -> Feedforward, 2 -> Recurrent).
2. Adjust parameters in section ``Test``.
3. Copy a test data file into ``/inputs/trainingdata/`` with the above mentioned naming.
4. Run ``main_NN_Vehicle_dynamics.py``.
5. The results will be saved in the ``\outputs`` folder.


## Some hints on how you could start with your own neural network

In general:
*  Begin with a small network size. One layer, 10 neurons for example. If the network does not deliver any results, than a fundamental problem may exist.
*  Slowly increase the complexity
*  If not sure, start with the feedforward model. It is easier to train.

### Possible starting parameter

*  Optimizer Mode = 4 (``Nesterov Adam`` showed the best results)
*  ``Leaky Relu`` for feedforward network, ``tanh`` for recurrent networks
*  Linear output activation  
*  2 hidden layers
*  10 neurons per layer
*  Small learning rate (lr=0.0001)
*  Batch_Size = 128
*  Validation split = 0.25
*  Standard Scaler

### Weblinks to search for suitable hyperparameters

*  [Hyper-parameters in Action! Part II — Weight Initializers](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
*  [Activation Functions Explained - GELU, SELU, ELU, ReLU and more](https://mlfromscratch.com/activation-functions-explained/#/)
*  [Tips for Training Recurrent Neural Networks](https://danijar.com/tips-for-training-recurrent-neural-networks/)
*  [37 Reasons why your Neural Network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)


## References

L. Hermansdorfer, R. Trauth, J. Betz and M. Lienkamp, "End-to-End Neural Network for Vehicle Dynamics Modeling," 2020 6th IEEE Congress on Information Science and Technology (CiSt), Agadir - Essaouira, Morocco, 2020, pp. 407-412, doi: 10.1109/CiSt49399.2021.9357196.

Contact:
* [Leonhard Hermansdorfer](mailto:leo.hermansdorfer@tum.de)
* [Rainer Trauth](mailto:trauth@ftm.mw.tum.de)

Please cite as:
```
@inproceedings{hermansdorfer2020,
booktitle={2020 6th IEEE Congress on Information Science and Technology (CiSt)},
title={End-to-End Neural Network for Vehicle Dynamics Modeling},
author={Hermansdorfer, Leonhard and Trauth, Rainer and Betz, Johannes and Lienkamp, Markus},
year={2020},
pages={407-412},
doi={10.1109/CiSt49399.2021.9357196}}
```
