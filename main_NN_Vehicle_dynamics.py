import numpy as np
import sys
import random

# custom modules
import helper_funcs_NN
import src
import visualization

"""
Created by: Leonhard Hermansdorfer, Rainer Trauth
Created on: 01.04.2020

Documentation
main script to run neural network training
"""


random.seed(7)
np.random.seed(7)

# ----------------------------------------------------------------------------------------------------------------------
# Manage Paths ---------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# create a dictionary which contains paths to all relevant folders and files
path_dict = helper_funcs_NN.src.manage_paths.manage_paths()


# ----------------------------------------------------------------------------------------------------------------------
# Read Parameters ------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# create a dictionary which contains all parameters
params_dict = helper_funcs_NN.src.handle_params.handle_params(path_dict=path_dict)


# ----------------------------------------------------------------------------------------------------------------------
# Training of the Neural Network ---------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# THIS IS THE START (CONFIGURATION) OF THE PROGRAM
if params_dict['NeuralNetwork_Settings']['model_mode'] == 1:
    src.train_neuralnetwork.train_neuralnetwork(path_dict=path_dict,
                                                params_dict=params_dict,
                                                nn_mode='feedforward')

if params_dict['NeuralNetwork_Settings']['model_mode'] == 2:
    src.train_neuralnetwork.train_neuralnetwork(path_dict=path_dict,
                                                params_dict=params_dict,
                                                nn_mode="recurrent")


# ----------------------------------------------------------------------------------------------------------------------
# Evaluation of the Neural Network -------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

# exit python if evaluation is disabled (NeuralNetwork_Settings.run_file_mode == 0)
if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 0:
    sys.exit('SYSTEM EXIT: exit due to run_file_mode is set to zero to avoid testing the neural network against '
             + 'vehicle sensor data')

for i_count in range(0, params_dict['Test']['n_test']):

    idx_start = params_dict['Test']['run_timestart'] + i_count * params_dict['Test']['iteration_step']

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        print('STARTING RUN FEEDFORWARD NETWORK')

        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="feedforward")

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        print('STARTING RUN RECURRENT NETWORK')

        src.run_neuralnetwork.run_nn(path_dict=path_dict,
                                     params_dict=params_dict,
                                     startpoint=idx_start,
                                     counter=i_count,
                                     nn_mode="recurrent")

    # save and plot results (if activated in parameter file)
    visualization.plot_results.plot_run(path_dict=path_dict,
                                        params_dict=params_dict,
                                        counter=i_count,
                                        start=idx_start)
