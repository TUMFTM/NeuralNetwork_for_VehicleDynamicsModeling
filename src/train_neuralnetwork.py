import os.path
import shutil
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN

import src
import visualization

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def train_neuralnetwork(path_dict: dict,
                        params_dict: dict,
                        nn_mode: str) -> None:
    """Manages the training process of the neural network.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    print('SAVE SETTINGS')

    if not params_dict['General']['bool_load_existingmodel']:
        shutil.copyfile('params/parameters.toml',
                        os.path.join(path_dict['path2results'], 'settings_' + nn_mode + '.toml'))

    print('LOAD AND SCALE DATA')

    data = src.load_data_for_nn.load_data(path2inputs_trainingdata=path_dict['path2inputs_trainingdata'],
                                          filename_trainingdata=path_dict['filename_trainingdata'])

    # prepare training data for specific neural network architecture

    train_data, val_data = src.prepare_data.create_dataset_separation(path_dict=path_dict,
                                                                      params_dict=params_dict,
                                                                      data=data,
                                                                      nn_mode=nn_mode)

    if nn_mode == "feedforward":

        monitor = params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']

        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_ff']

        min_delta = 0.000005

    elif nn_mode == "recurrent":

        monitor = 'val_' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']

        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_recurr']

        min_delta = 0.000001

    # CREATE CALLBACKS
    es = EarlyStopping(monitor=monitor,
                       mode='min',
                       verbose=1,
                       patience=params_dict['NeuralNetwork_Settings']['earlystopping_patience'])

    mc = ModelCheckpoint(filepath=filepath2results_trainedmodel,
                         monitor=monitor,
                         mode='min',
                         verbose=1,
                         save_best_only=True)

    reduce_lr_loss = ReduceLROnPlateau(monitor=monitor,
                                       factor=params_dict['NeuralNetwork_Settings']['reduceLR_factor'],
                                       patience=params_dict['NeuralNetwork_Settings']['patience_LR'],
                                       verbose=1,
                                       mode='min',
                                       min_delta=min_delta)

    Nan = TerminateOnNaN()

    # Create a new neural network model or loads an existing one.
    model = src.neural_network_fcn.create_nnmodel(path_dict=path_dict,
                                                  params_dict=params_dict,
                                                  nn_mode=nn_mode)

    history_mod = model.fit(x=train_data[0],
                            y=train_data[1],
                            batch_size=params_dict['NeuralNetwork_Settings']['batch_size'],
                            validation_data=(val_data[0], val_data[1]),
                            epochs=params_dict['NeuralNetwork_Settings']['epochs'],
                            verbose=1,
                            shuffle=True,
                            callbacks=[reduce_lr_loss, es, mc, Nan],
                            use_multiprocessing=True)

    print(history_mod.history.keys())

    if params_dict['General']['plot_mse']:
        print('PLOT MSE CURVE')

        visualization.plot_results.plot_mse(path_dict=path_dict,
                                            params_dict=params_dict,
                                            histories=history_mod)
