# Neural Network for Vehicle Dynamics Modeling

## Introduction

This repository provides a neural network training algorithm which is able to substitute a physics-based single-track ("bicycle") model for vehicle dynamics simulation.

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
data to represent the vehicle’s dynamic behavior.\
We present a neural network architecture that aims to
replace a single-track model for vehicle dynamics modeling.

The model is trained to predict the vehicle state of the next (simulation) timestep by using the current vehicle input together with the current vehicle state plus the vehicle states of the past last four timesteps.

The vehicle state consists of:
* Longitudinal and lateral velocity (in m/s)
* Yaw rate (in rad/s)
* Longitudinal and lateral acceleration (in m/s²)

The vehicle input consists of:
* Steering angle (in rad)
* Torque of rear left and right wheel (in Nm)
* brake pressure at front and rear (in bar)

The training process of the neural network has two different modes (switch via parameter settings):
* Mode 1 --> Feedforward Model
* Mode 2 --> Recurrent Model (GRU, LSTM, ...)


## List of components
* `helper_funcs_NN`: This package contains helper functions used in several other functions when running the main script for neural network training.
* `inputs`: This folder contains pre-trained neural network models and/ or vehicle sensor data to train a neural network (or re-train an existing one).
* `outputs`: This folder contains the the simulation results, figures, the trained neural network, a backup of the used parameters and information on how the input/ output data was scaled before and re-scaled after training.
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

**Please Note:**\
The provided data is exemplary and generated from an existing [vehicle dynamics simulation](https://github.com/TUMFTM/sim_vehicle_dynamics). For the application described in the paper, we did use real vehicle sensor data instead.

### Training data
The training data files contain vehicle data which is used to train the neural network in order to model the vehicle's dynamic behavior. All training data filenames must start with "data_to_train" (e.g. "data_to_train_run1.csv"). All files located in this folder and starting with this prefix are then used for training the neural network.
The data format is described above.

### Test data 
The test data file contains vehicle data which is used to test the trained neural network. The vehicle sensor data provided in this file is compared against the neural network's vehicle state prediction when receiving the same inputs (as de real vehicle does).
The vehicle input from the provided file (steering angle, torque and brake pressure) is applied to the neural network and the output (vehicle state) is compared to the actual vehicle state from the provided test file.

The name of the test data file has to be "data_to_run.csv".
The data format is equal to the "data_to_train" files and is described above.


## Running the code:
Following steps are necessary to run the training process:
1. Set the parameters in `/params/parameters.toml` (you can find good hyperparameters to start with in the instructions below)
2. Set model.mode and run.file.mode (0 --> No usage, 1 --> Feedforward, 2--> Recurrent Model)
3. Set the optimizer parameters in `/params/parameters.toml`
4. You can change the model structure in the neural.network.fcn file (This is the main file where you decide how your model looks like)
5. Set the data standardization/normalization mode in the parameters file (normalization range can be changed in the data.preparation.file)
7. If you later only want to do the simulation and not the training than turn off the model.mode with 0
8. It is possible to load an already created model (load.old.model=True)
9. The parameter scaler and the results will be saved in the outputs file


### Run Test against real vehicle data
In ``params/parameters.toml`` set ``NeuralNetwork_Settings.run_file_mode`` to 1 or 2 (depending on which NN should be used for testing: 1 -> Feedforward, 2 -> Recurrent)
7. It is possible to load an already created model (load.old.model=True, copy the model you want to use into the input file)


## Some hints on how you could start with your own neural network

In general:
*  Begin with a VERY small network size. One layer, 10 neurons for example. If the network does not deliver any results, than a fundamental problem may exist.
*  Then slowly try to increase the complexity
*  If not sure, start with the feedforward model. It is much easier to train.

### Possible starting parameters:

*  Optimizer Mode = 4 (Nesterov Adam showed the best results)
*  `Leaky Relu` for feedforward network, `tanh` for recurrent networks
*  Output activation has to be linear
*  Use 2 hidden layers
*  Use 10 neurons per layer
*  Use small learning rate (lr=0.0001)
*  Batch_Size = 128
*  Validation split = 0.2
*  Use Standard Scaler
*  For recurrent networks, it is usually better to use GRU than LSTM

### Weblinks to search for good Hyperparameters

*  [Hyper-parameters in Action! Part II — Weight Initializers](https://towardsdatascience.com/hyper-parameters-in-action-part-ii-weight-initializers-35aee1a28404)
*  [Activation Functions Explained - GELU, SELU, ELU, ReLU and more](https://mlfromscratch.com/activation-functions-explained/#/)
*  [Tips for Training Recurrent Neural Networks](https://danijar.com/tips-for-training-recurrent-neural-networks/)
*  [37 Reasons why your Neural Network is not working](https://blog.slavv.com/37-reasons-why-your-neural-network-is-not-working-4020854bd607)


## References

L. Hermansdorfer, R. Trauth, J. Betz and M. Lienkamp, "End-to-End Neural Network for Vehicle Dynamics Modeling," 2020 6th IEEE Congress on Information Science and Technology (CiSt), Agadir - Essaouira, Morocco, 2020, pp. 407-412, doi: 10.1109/CiSt49399.2021.9357196.

Contact: [Leonhard Hermansdorfer](mailto:leo.hermansdorfer@tum.de)

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
