import numpy as np
import sys
from tqdm import tqdm
import os.path
from tensorflow import keras

# custom modules
import src

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""

# SET FLOATING POINT PRECISION
np.set_printoptions(formatter={'float': lambda x: "{0:0.16f}".format(x)})


def run_nn_feedforward(path_dict: dict,
                       params_dict: dict,
                       startpoint: float,
                       counter: int):
    """Runs the feedforward neural network to test its predictions against actual vehicle data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param startpoint:          row index where to start using provided test data
    :type startpoint: float
    :param counter:             number of current testing loop (only used for naming of output files)
    :type counter: int
    """

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']
        path2model = path_dict['filepath2inputs_trainedmodel_ff']

    else:
        path2scaler = path_dict['filepath2scaler_save']
        path2model = path_dict['filepath2results_trainedmodel_ff']

    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    if startpoint + params_dict['Test']['run_timespan'] > data.shape[0]:
        sys.exit("test dataset fully covered -> exit main script")

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # scale dataset the vanish effects of different input data quantities
    data = src.prepare_data.scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar = \
        src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint,
                                                       params_dict['Test']['run_timespan'], 0)

    # load neural network model
    model = keras.models.load_model(path2model)

    results = np.zeros((len(torqueRR_Nm) + input_timesteps, input_shape))
    new_input = np.zeros((1, input_shape * input_timesteps))

    for m in range(0, input_timesteps):
        results[m, 0:output_shape] \
            = initial[:, m * input_shape:m * input_shape + output_shape]

    for i in tqdm(range(0, len(torqueRR_Nm))):
        if i == 0:
            result_process = model.predict(initial)
            results[i + input_timesteps, 0:output_shape] = result_process

            new_input = input_conversion(params_dict,
                                         initial,
                                         input_shape,
                                         input_timesteps,
                                         result_process,
                                         steeringangle_rad,
                                         torqueRL_Nm,
                                         torqueRR_Nm,
                                         brakepresF_bar,
                                         brakepresR_bar,
                                         i)

        else:
            # start_time = time.time()
            result_process = model.predict(new_input)
            # print("--- %s seconds ---" % (time.time() - start_time))
            results[i + input_timesteps, 0:output_shape] = result_process

            new_input = input_conversion(params_dict,
                                         new_input,
                                         input_shape,
                                         input_timesteps,
                                         result_process,
                                         steeringangle_rad,
                                         torqueRL_Nm,
                                         torqueRR_Nm,
                                         brakepresF_bar,
                                         brakepresR_bar,
                                         i)

    results[:, output_shape:input_shape] = data[startpoint:startpoint + len(steeringangle_rad) + input_timesteps,
                                                output_shape:input_shape]

    results = src.prepare_data.scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    np.savetxt(os.path.join(path_dict['path2results_matfiles'], 'prediction_result' + str(counter) + '.csv'), results)


def run_nn_recurrent(path_dict: dict,
                     params_dict: dict,
                     startpoint: float,
                     counter: int):
    """Runs the recurrent neural network to test its predictions against actual vehicle data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param startpoint:          row index where to start using provided test data
    :type startpoint: float
    :param counter:             number of current testing loop (only used for naming of output files)
    :type counter: int
    """

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']
        path2model = path_dict['filepath2inputs_trainedmodel_recurr']

    else:
        path2scaler = path_dict['filepath2scaler_save']
        path2model = path_dict['filepath2results_trainedmodel_recurr']

    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    if startpoint + params_dict['Test']['run_timespan'] > data.shape[0]:
        sys.exit("test dataset fully covered -> exit main script")

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # scale dataset the vanish effects of different input data quantities
    data = src.prepare_data.scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar = \
        src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint,
                                                       params_dict['Test']['run_timespan'], 1)

    # load neural network model
    model_recurrent = keras.models.load_model(path2model)

    results = np.zeros((len(torqueRR_Nm) + input_timesteps, input_shape))
    new_input = np.zeros((1, input_timesteps, input_shape))

    results[0:input_timesteps, :] = initial[0, :, :]

    for i in tqdm(range(0, len(torqueRR_Nm))):

        if i == 0:
            result_process = model_recurrent.predict(initial)
            results[i + input_timesteps, 0:output_shape] = result_process

            new_input = input_conversion_recurrent(params_dict,
                                                   initial,
                                                   input_shape,
                                                   input_timesteps,
                                                   result_process,
                                                   steeringangle_rad,
                                                   torqueRL_Nm,
                                                   torqueRR_Nm,
                                                   brakepresF_bar,
                                                   brakepresR_bar,
                                                   i)

        else:
            result_process = model_recurrent.predict(new_input)
            results[i + input_timesteps, 0:output_shape] = result_process

            new_input = input_conversion_recurrent(params_dict,
                                                   new_input,
                                                   input_shape,
                                                   input_timesteps,
                                                   result_process,
                                                   steeringangle_rad,
                                                   torqueRL_Nm,
                                                   torqueRR_Nm,
                                                   brakepresF_bar,
                                                   brakepresR_bar,
                                                   i)

    results[:, output_shape:input_shape] = data[startpoint:startpoint + len(steeringangle_rad) + input_timesteps,
                                                output_shape:input_shape]

    results = src.prepare_data.scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    np.savetxt(os.path.join(path_dict['path2results_matfiles'], 'prediction_result_recurrent' + str(counter) + '.csv'),
               results)


def input_conversion(params_dict,
                     init,
                     input_shape,
                     input_timesteps,
                     lastresults,
                     steerangle,
                     NN_RL,
                     NN_RR,
                     brakeFF,
                     brakeRR,
                     p):

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']

    temp = np.zeros((1, input_shape * input_timesteps))
    temp[:, 0:input_shape * (input_timesteps - 1)] = init[0, input_shape:input_shape * input_timesteps]

    temp[:, input_shape * (input_timesteps - 1):input_shape * (input_timesteps - 1) + output_shape] = lastresults

    temp[:, input_shape * (input_timesteps - 1) + output_shape] = steerangle[p]
    temp[:, input_shape * (input_timesteps - 1) + output_shape + 1] = NN_RL[p]
    temp[:, input_shape * (input_timesteps - 1) + output_shape + 2] = NN_RR[p]
    temp[:, input_shape * (input_timesteps - 1) + output_shape + 3] = brakeFF[p]
    temp[:, input_shape * (input_timesteps - 1) + output_shape + 4] = brakeRR[p]

    return temp


def input_conversion_recurrent(params_dict,
                               init,
                               input_shape,
                               input_timesteps,
                               lastresults,
                               steerangle,
                               NN_RL,
                               NN_RR,
                               brakeFF,
                               brakeRR,
                               p):

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']

    temp = np.zeros((1, input_timesteps, input_shape))
    temp[0, 0:input_timesteps - 1, :] = init[0, 1:input_timesteps, :]

    temp[0, input_timesteps - 1, 0:output_shape] = lastresults
    temp[0, input_timesteps - 1, output_shape] = steerangle[p]
    temp[0, input_timesteps - 1, output_shape + 1] = NN_RL[p]
    temp[0, input_timesteps - 1, output_shape + 2] = NN_RR[p]
    temp[0, input_timesteps - 1, output_shape + 3] = brakeFF[p]
    temp[0, input_timesteps - 1, output_shape + 4] = brakeRR[p]

    return temp
