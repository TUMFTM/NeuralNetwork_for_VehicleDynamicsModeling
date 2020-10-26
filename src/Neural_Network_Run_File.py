import numpy as np
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


def run_nn(path_dict: dict,
           params_dict: dict,
           startpoint: float,
           durationspan: float,
           counter: int):
    """doc string
    """

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']
        path2model = path_dict['filepath2inputs_trainedmodel_ff']

    else:
        path2scaler = path_dict['filepath2scaler_save']
        path2model = path_dict['filepath2results_trainedmodel_ff']

    datasets = params_dict['NeuralNetwork_Settings']['datasets']  # number of independent input parameters
    timesteps = params_dict['NeuralNetwork_Settings']['timesteps']

    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    data = src.prepare_data.scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, force_RL, force_RR, brakeF, brakeR = \
        src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint, durationspan, 0)

    # model = src.load_data_for_nn.load_kerasmodel(filepath2model=path2model)

    model = keras.models.load_model(path2model)

    results = np.zeros((len(force_RR) + timesteps, params_dict['NeuralNetwork_Settings']['datasets']))

    for m in range(0, timesteps):
        results[m, 0:params_dict['NeuralNetwork_Settings']['output_shape']] \
            = initial[:, m * datasets:m * datasets + params_dict['NeuralNetwork_Settings']['output_shape']]

    new_input = np.zeros((1, datasets * timesteps))

    for i in tqdm(range(0, len(force_RR))):
        if i == 0:
            result_process = model.predict(initial)
            results[i + timesteps, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = result_process

            new_input = input_conversion(params_dict,
                                         initial,
                                         datasets,
                                         timesteps,
                                         result_process,
                                         steeringangle_rad,
                                         force_RL,
                                         force_RR,
                                         brakeF,
                                         brakeR,
                                         i)

        else:
            # start_time = time.time()
            result_process = model.predict(new_input)
            # print("--- %s seconds ---" % (time.time() - start_time))
            results[i + timesteps, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = result_process

            new_input = input_conversion(params_dict,
                                         new_input,
                                         datasets,
                                         timesteps,
                                         result_process,
                                         steeringangle_rad,
                                         force_RL,
                                         force_RR,
                                         brakeF,
                                         brakeR,
                                         i)

    results[:, params_dict['NeuralNetwork_Settings']['output_shape']:params_dict['NeuralNetwork_Settings']['datasets']]\
        = data[startpoint:startpoint + len(steeringangle_rad) + params_dict['NeuralNetwork_Settings']['timesteps'],
               params_dict['NeuralNetwork_Settings']['output_shape']:params_dict['NeuralNetwork_Settings']['datasets']]

    results = src.prepare_data.scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    np.savetxt(os.path.join(path_dict['path2results_matfiles'], 'prediction_result' + str(counter) + '.csv'), results)


def run_nn_recurrent(path_dict: dict,
                     params_dict: dict,
                     startpoint: float,
                     durationspan: float,
                     counter: int):
    """[summary]
    """

    # if no model was trained, load existing model in inputs folder /inputs/trained_models
    if params_dict['NeuralNetwork_Settings']['model_mode'] == 0:
        path2scaler = path_dict['filepath2scaler_load']
        path2model = path_dict['filepath2inputs_trainedmodel_recurr']

    else:
        path2scaler = path_dict['filepath2scaler_save']
        path2model = path_dict['filepath2results_trainedmodel_recurr']

    datasets = params_dict['NeuralNetwork_Settings']['datasets']  # number of independent input parameters
    timesteps = params_dict['NeuralNetwork_Settings']['timesteps']

    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        data = np.loadtxt(fh, delimiter=',')

    data = src.prepare_data.scaler_run(path2scaler=path2scaler,
                                       params_dict=params_dict,
                                       dataset=data)

    initial, steeringangle_rad, force_rl, force_rr, brakef, braker = \
        src.prepare_data.create_dataset_separation_run(data, params_dict, startpoint, durationspan, 1)

    # model_recurrent = src.load_data_for_nn.load_kerasmodel(filepath2model=path2model)

    model_recurrent = keras.models.load_model(path2model)

    results = np.zeros((len(force_rr) + timesteps, params_dict['NeuralNetwork_Settings']['datasets']))

    results[0:params_dict['NeuralNetwork_Settings']['timesteps'], :] = initial[0, :, :]

    new_input = np.zeros((1, timesteps, datasets))

    for i in tqdm(range(0, len(force_rr))):

        if i == 0:
            result_process = model_recurrent.predict(initial)
            results[i + timesteps, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = result_process

            new_input = input_conversion_recurrent(params_dict,
                                                   initial,
                                                   datasets,
                                                   timesteps,
                                                   result_process,
                                                   steeringangle_rad,
                                                   force_rl,
                                                   force_rr,
                                                   brakef,
                                                   braker,
                                                   i)

        else:
            result_process = model_recurrent.predict(new_input)
            results[i + timesteps, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = result_process

            new_input = input_conversion_recurrent(params_dict,
                                                   new_input,
                                                   datasets,
                                                   timesteps,
                                                   result_process,
                                                   steeringangle_rad,
                                                   force_rl,
                                                   force_rr,
                                                   brakef,
                                                   braker,
                                                   i)

    results[:, params_dict['NeuralNetwork_Settings']['output_shape']:params_dict['NeuralNetwork_Settings']['datasets']]\
        = data[startpoint:startpoint + len(steeringangle_rad) + params_dict['NeuralNetwork_Settings']['timesteps'],
               params_dict['NeuralNetwork_Settings']['output_shape']:params_dict['NeuralNetwork_Settings']['datasets']]

    results = src.prepare_data.scaler_reverse(path2scaler=path2scaler,
                                              params_dict=params_dict,
                                              dataset=results)

    np.savetxt(os.path.join(path_dict['path2results_matfiles'], 'prediction_result_recurrent' + str(counter) + '.csv'),
               results)


def input_conversion(params_dict,
                     init,
                     datasets,
                     timesteps,
                     lastresults,
                     steerangle,
                     NN_RL,
                     NN_RR,
                     brakeFF,
                     brakeRR,
                     p):

    temp = np.zeros((1, datasets * timesteps))
    temp[:, 0:datasets * (timesteps - 1)] = init[0, datasets:datasets * timesteps]

    temp[:, datasets * (timesteps - 1):datasets * (timesteps - 1)
         + params_dict['NeuralNetwork_Settings']['output_shape']] = lastresults

    temp[:, datasets * (timesteps - 1) + params_dict['NeuralNetwork_Settings']['output_shape']] = steerangle[p]
    temp[:, datasets * (timesteps - 1) + params_dict['NeuralNetwork_Settings']['output_shape'] + 1] = NN_RL[p]
    temp[:, datasets * (timesteps - 1) + params_dict['NeuralNetwork_Settings']['output_shape'] + 2] = NN_RR[p]
    temp[:, datasets * (timesteps - 1) + params_dict['NeuralNetwork_Settings']['output_shape'] + 3] = brakeFF[p]
    temp[:, datasets * (timesteps - 1) + params_dict['NeuralNetwork_Settings']['output_shape'] + 4] = brakeRR[p]

    return temp


def input_conversion_recurrent(params_dict,
                               init,
                               datasets,
                               timesteps,
                               lastresults,
                               steerangle,
                               NN_RL,
                               NN_RR,
                               brakeFF,
                               brakeRR,
                               p):

    temp = np.zeros((1, timesteps, datasets))
    temp[0, 0:timesteps - 1, :] = init[0, 1:timesteps, :]

    temp[0, timesteps - 1, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = lastresults
    temp[0, timesteps - 1, params_dict['NeuralNetwork_Settings']['output_shape']] = steerangle[p]
    temp[0, timesteps - 1, params_dict['NeuralNetwork_Settings']['output_shape'] + 1] = NN_RL[p]
    temp[0, timesteps - 1, params_dict['NeuralNetwork_Settings']['output_shape'] + 2] = NN_RR[p]
    temp[0, timesteps - 1, params_dict['NeuralNetwork_Settings']['output_shape'] + 3] = brakeFF[p]
    temp[0, timesteps - 1, params_dict['NeuralNetwork_Settings']['output_shape'] + 4] = brakeRR[p]

    return temp
