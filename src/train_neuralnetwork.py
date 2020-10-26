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
    """[summary]

    :param path_dict: [description]
    :type path_dict: dict
    :param params_dict: [description]
    :type params_dict: dict
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    print('SAVE SETTINGS')

    if not params_dict['General']['bool_load_existingmodel']:
        shutil.copyfile('params/parameters.toml',
                        os.path.join(path_dict['path2results'], 'settings_' + nn_mode + '.toml'))
        shutil.copyfile('src/neural_network_fcn.py',
                        os.path.join(path_dict['path2results'], 'settings_' + nn_mode + '_function.txt'))

    print('LOAD AND SCALE DATA')

    data = src.load_data_for_nn.load_data(path2inputs_trainingdata=path_dict['path2inputs_trainingdata'],
                                          filename_trainingdata=path_dict['filename_trainingdata'])

    # prepare training data for specific neural network archtecture
    if nn_mode == "feedforward":

        train_data, val_data = src.prepare_data.create_dataset_separation(path_dict=path_dict,
                                                                          params_dict=params_dict,
                                                                          datas=data)

        monitor = params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']

        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_ff']

        min_delta = 0.000005

    elif nn_mode == "recurrent":

        train_data, val_data = src.prepare_data.create_dataset_separation_recurrent(path_dict=path_dict,
                                                                                    params_dict=params_dict,
                                                                                    datas=data)

        monitor = 'val_' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']

        filepath2results_trainedmodel = path_dict['filepath2results_trainedmodel_recurr']

        min_delta = 0.000001

    # CREATE CALLBACKS
    es = EarlyStopping(monitor=monitor,
                       mode='min',
                       verbose=1,
                       patience=80)

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
