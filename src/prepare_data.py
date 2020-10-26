import numpy as np
import os.path
import pickle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from joblib import dump, load

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def scaler(path_dict: dict,
           params_dict: dict,
           dataset):
    """doc string
    """

    if params_dict['General']['use_old_transformation']:

        if params_dict['General']['scaler_mode'] == 2:

            with open(os.path.join(path_dict['path2results'], 'scaler_tanh'), 'rb') as f:
                m, std = pickle.load(f)
                dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

        else:
            print('USE OLD TRANSFORMATION')
            scalers = load(path_dict['filepath2scaler_load'])
            dataset_out = scalers.transform(dataset)

    else:

        if params_dict['General']['scaler_mode'] == 0:
            print('USE STANDARD SCALER')
            scalers = StandardScaler()  # with_mean=True, with_std=True
            scalers = scalers.fit(dataset)
            dataset_out = scalers.transform(dataset)

        if params_dict['General']['scaler_mode'] == 1:
            print('USE MINMAX SCALER')
            scalers = MinMaxScaler(feature_range=(-1, 1))
            scalers = scalers.fit(dataset)
            dataset_out = scalers.transform(dataset)

        if params_dict['General']['scaler_mode'] == 2:
            m = np.mean(dataset, axis=0)
            std = np.std(dataset, axis=0)
            dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

        if params_dict['General']['save_scaling']:

            if params_dict['General']['scaler_mode'] == 2:

                with open(os.path.join(path_dict['path2results'], 'scaler_tanh'), 'wb') as f:
                    pickle.dump([m, std], f)

            else:
                dump(scalers, path_dict['filepath2scaler_save'])

    return dataset_out


def scaler_run(path2scaler: str,
               params_dict: dict,
               dataset):
    """doc string
    """

    if params_dict['General']['scaler_mode'] == 2:

        with open('outputs/scaler_tanh', 'rb') as f:
            m, std = pickle.load(f)
            dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

    else:
        scalers = load(path2scaler)
        dataset_out = scalers.transform(dataset)

    return dataset_out


def scaler_reverse(path2scaler: str,
                   params_dict: dict,
                   dataset):
    """doc string
    """

    print('TRANSFORM RESULT WITH SCALER TO PHYSICAL QUANTITIES')

    if params_dict['General']['scaler_mode'] == 2:

        with open('outputs/scaler_tanh', 'rb') as f:
            m, std = pickle.load(f)
            dataset_std_rev = m + 100 * std * np.arctanh(2 * dataset - 1)

    else:
        scalers = load(path2scaler)
        dataset_std_rev = scalers.inverse_transform(dataset)

    return dataset_std_rev


def create_dataset_separation_run(data_in,
                                  params_dict: dict,
                                  start,
                                  duration,
                                  mode):

    initials = data_in[start:start + params_dict['NeuralNetwork_Settings']['timesteps'], :]

    if mode == 0:
        initials = np.reshape(initials, (1,
                                         params_dict['NeuralNetwork_Settings']['timesteps']
                                         * params_dict['NeuralNetwork_Settings']['datasets']))

    if mode == 1:
        initials = np.reshape(initials, (1,
                                         params_dict['NeuralNetwork_Settings']['timesteps'],
                                         params_dict['NeuralNetwork_Settings']['datasets']))

    sta_ang = data_in[start + params_dict['NeuralNetwork_Settings']['timesteps']:start + duration,
                      params_dict['NeuralNetwork_Settings']['output_shape']]

    force_rl = data_in[start + params_dict['NeuralNetwork_Settings']['timesteps']:start + duration,
                       params_dict['NeuralNetwork_Settings']['output_shape'] + 1]

    force_rr = data_in[start + params_dict['NeuralNetwork_Settings']['timesteps']:start + duration,
                       params_dict['NeuralNetwork_Settings']['output_shape'] + 2]

    brakef = data_in[start + params_dict['NeuralNetwork_Settings']['timesteps']:start + duration,
                     params_dict['NeuralNetwork_Settings']['output_shape'] + 3]

    braker = data_in[start + params_dict['NeuralNetwork_Settings']['timesteps']:start + duration,
                     params_dict['NeuralNetwork_Settings']['output_shape'] + 4]

    return initials, sta_ang, force_rl, force_rr, brakef, braker


def extract_part(datax,
                 params_dict: dict,
                 data_infox,
                 z):

    summ = np.sum(data_infox[0:1 + z, :])
    data_part = datax[summ - data_infox[z, 0]:summ, :]
    labels_part = data_part[2 * params_dict['NeuralNetwork_Settings']['timesteps']
                            - 1::params_dict['NeuralNetwork_Settings']['timesteps'],
                            0:params_dict['NeuralNetwork_Settings']['output_shape']]

    data_part = data_part[0:len(data_part) - params_dict['NeuralNetwork_Settings']['timesteps'], :]

    return np.array(data_part), np.array(labels_part)


def create_dataset_separation_recurrent(path_dict: dict,
                                        params_dict: dict,
                                        datas: dict) -> tuple:
    """
    :param datas:
    :return:
    """
    file_counting = 0
    filepath = path_dict['path2inputs_trainingdata']

    if os.path.exists(filepath):

        for file in os.listdir(filepath):

            if file.startswith('data_to_train'):
                file_counting += 1

    lengthsum = 0

    for m in range(0, file_counting):
        lengthsum += (len(datas[m]) - params_dict['NeuralNetwork_Settings']['timesteps'])

    data_train = np.zeros((lengthsum * params_dict['NeuralNetwork_Settings']['timesteps'],
                           params_dict['NeuralNetwork_Settings']['datasets']))  # np.empty((0, params_dict['NeuralNetwork_Settings']['datasets']))

    data_labels = np.zeros((lengthsum, params_dict['NeuralNetwork_Settings']['output_shape']))
    lengthsumtwo = 0
    lengthsumtwolabels = 0

    for u in range(0, file_counting):
        data_labels[lengthsumtwolabels:lengthsumtwolabels
                    + len(datas[u])
                    - params_dict['NeuralNetwork_Settings']['timesteps']] \
            = (datas[u])[params_dict['NeuralNetwork_Settings']['timesteps']:,
                         0:params_dict['NeuralNetwork_Settings']['output_shape']]

        for pp in range(0, len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps']):
            idx = lengthsumtwo + pp * params_dict['NeuralNetwork_Settings']['timesteps']
            data_train[idx:idx + params_dict['NeuralNetwork_Settings']['timesteps'], :] \
                = (datas[u])[pp:pp + params_dict['NeuralNetwork_Settings']['timesteps'], :]

        lengthsumtwolabels += ((len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps']))
        lengthsumtwo += ((len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps'])
                         * params_dict['NeuralNetwork_Settings']['timesteps'])

    data_train = np.reshape(data_train, (len(data_train) // params_dict['NeuralNetwork_Settings']['timesteps'],
                                         params_dict['NeuralNetwork_Settings']['timesteps'],
                                         params_dict['NeuralNetwork_Settings']['datasets']))

    indices = np.arange(data_train.shape[0])

    if params_dict['General']['shuffle_mode']:
        np.random.RandomState(params_dict['General']['shuffle_number']).shuffle(indices)

    else:
        np.random.shuffle(indices)

    data_train = data_train[indices]
    data_labels = data_labels[indices]

    data_train = np.reshape(data_train, (len(data_labels) * params_dict['NeuralNetwork_Settings']['timesteps'],
                                         params_dict['NeuralNetwork_Settings']['datasets']))

    p = int(len(data_train) * (1 - params_dict['NeuralNetwork_Settings']['val_split']))
    mod = p % 5
    p = p - mod
    train_x = data_train[0:p, :]

    train_x = scaler(path_dict=path_dict,
                     params_dict=params_dict,
                     dataset=train_x)

    temp = np.zeros((len(data_labels), params_dict['NeuralNetwork_Settings']['datasets']))
    temp[:, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = data_labels

    temp = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                      params_dict=params_dict,
                      dataset=temp)

    data_labels = temp[:, 0:params_dict['NeuralNetwork_Settings']['output_shape']]

    train_y = data_labels[0:(p // params_dict['NeuralNetwork_Settings']['timesteps']), :]
    val_y = data_labels[(p // params_dict['NeuralNetwork_Settings']['timesteps']):len(data_labels), :]

    val_x = data_train[p:len(data_train), :]
    val_x = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                       params_dict=params_dict,
                       dataset=val_x)

    train_x = np.reshape(train_x, (p // params_dict['NeuralNetwork_Settings']['timesteps'],
                                   params_dict['NeuralNetwork_Settings']['timesteps'],
                                   params_dict['NeuralNetwork_Settings']['datasets']))

    val_x = np.reshape(val_x, ((len(data_train) - p) // params_dict['NeuralNetwork_Settings']['timesteps'],
                               params_dict['NeuralNetwork_Settings']['timesteps'],
                               params_dict['NeuralNetwork_Settings']['datasets']))

    # # data_train = np.reshape(data_train, (len(data_train)//params_dict['NeuralNetwork_Settings']['timesteps'], params_dict['NeuralNetwork_Settings']['timesteps'], params_dict['NeuralNetwork_Settings']['datasets']))
    # # data_labels = np.reshape(data_labels, (len(data_train), 1, params_dict['NeuralNetwork_Settings']['output_shape']))
    #
    # # train_x, test_x, train_y, test_y = train_test_split(data_train, data_labels, test_size=params_dict['NeuralNetwork_Settings']['test_split'], random_state=42)
    train_data = (train_x, train_y)
    val_data = (val_x, val_y)

    return train_data, val_data


def create_dataset_separation(path_dict: dict,
                              params_dict: dict,
                              datas: dict) -> tuple:
    """
    :param datas:
    :return:
    """
    file_counting = 0
    filepath = path_dict['path2inputs_trainingdata']

    if os.path.exists(filepath):

        for file in os.listdir(filepath):

            if file.startswith('data_to_train'):
                file_counting += 1

    lengthsum = 0

    for m in range(0, file_counting):
        lengthsum += (len(datas[m]) - params_dict['NeuralNetwork_Settings']['timesteps'])

    data_train = np.zeros((lengthsum * params_dict['NeuralNetwork_Settings']['timesteps'],
                           params_dict['NeuralNetwork_Settings']['datasets']))  # np.empty((0, params_dict['NeuralNetwork_Settings']['datasets']))
    data_labels = np.zeros((lengthsum, params_dict['NeuralNetwork_Settings']['output_shape']))
    lengthsumtwo = 0
    lengthsumtwolabels = 0

    for u in range(0, file_counting):
        data_labels[lengthsumtwolabels:lengthsumtwolabels
                    + len(datas[u])
                    - params_dict['NeuralNetwork_Settings']['timesteps']] \
            = (datas[u])[params_dict['NeuralNetwork_Settings']['timesteps']:,
                         0:params_dict['NeuralNetwork_Settings']['output_shape']]

        for pp in range(0, len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps']):
            idx = lengthsumtwo + pp * params_dict['NeuralNetwork_Settings']['timesteps']
            data_train[idx:idx + params_dict['NeuralNetwork_Settings']['timesteps'], :] =\
                (datas[u])[pp:pp + params_dict['NeuralNetwork_Settings']['timesteps'], :]

        lengthsumtwolabels += ((len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps']))
        lengthsumtwo += ((len(datas[u]) - params_dict['NeuralNetwork_Settings']['timesteps'])
                         * params_dict['NeuralNetwork_Settings']['timesteps'])

    data_train = np.reshape(data_train, (len(data_labels), params_dict['NeuralNetwork_Settings']['timesteps']
                                         * params_dict['NeuralNetwork_Settings']['datasets']))

    indices = np.arange(data_train.shape[0])

    if params_dict['General']['shuffle_mode']:
        np.random.RandomState(params_dict['General']['shuffle_number']).shuffle(indices)

    else:
        np.random.shuffle(indices)

    data_train = data_train[indices]
    data_labels = data_labels[indices]

    data_train = np.reshape(data_train, (len(data_labels) * params_dict['NeuralNetwork_Settings']['timesteps'],
                                         params_dict['NeuralNetwork_Settings']['datasets']))

    p = int(len(data_train) * (1 - params_dict['NeuralNetwork_Settings']['val_split']))
    mod = p % 5
    p = p - mod
    train_x = data_train[0:p, :]

    train_x = scaler(path_dict=path_dict,
                     params_dict=params_dict,
                     dataset=train_x)

    temp = np.zeros((len(data_labels), params_dict['NeuralNetwork_Settings']['datasets']))
    temp[:, 0:params_dict['NeuralNetwork_Settings']['output_shape']] = data_labels

    temp = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                      params_dict=params_dict,
                      dataset=temp)

    data_labels = temp[:, 0:params_dict['NeuralNetwork_Settings']['output_shape']]

    train_y = data_labels[0:(p // params_dict['NeuralNetwork_Settings']['timesteps']), :]
    val_y = data_labels[(p // params_dict['NeuralNetwork_Settings']['timesteps']):len(data_labels), :]

    val_x = data_train[p:len(data_train), :]
    val_x = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                       params_dict=params_dict,
                       dataset=val_x)

    train_x = np.reshape(train_x, (p // params_dict['NeuralNetwork_Settings']['timesteps'],
                                   params_dict['NeuralNetwork_Settings']['timesteps']
                                   * params_dict['NeuralNetwork_Settings']['datasets']))

    val_x = np.reshape(val_x, ((len(data_train) - p * params_dict['NeuralNetwork_Settings']['timesteps']),
                               params_dict['NeuralNetwork_Settings']['timesteps']
                               * params_dict['NeuralNetwork_Settings']['datasets']))

    train_data = (train_x, train_y)
    val_data = (val_x, val_y)

    return train_data, val_data
