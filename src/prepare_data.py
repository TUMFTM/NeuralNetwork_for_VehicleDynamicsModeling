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
           dataset: np.array) -> np.array:
    """Scales dataset during preprocessing.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get scaled
    :type dataset: np.array
    :return:                    scaled dataset
    :rtype: np.array
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

        # save scaling information
        if params_dict['General']['scaler_mode'] == 2:

            with open(os.path.join(path_dict['path2results'], 'scaler_tanh'), 'wb') as f:
                pickle.dump([m, std], f)

        else:
            dump(scalers, path_dict['filepath2scaler_save'])

    return dataset_out


# ----------------------------------------------------------------------------------------------------------------------

def scaler_run(path2scaler: str,
               params_dict: dict,
               dataset: np.array):
    """Scales dataset for runing NN test.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get scaled
    :type dataset: np.array
    :return:                    scaled dataset
    :rtype: np.array
    """

    if params_dict['General']['scaler_mode'] == 2:

        with open('outputs/scaler_tanh', 'rb') as f:
            m, std = pickle.load(f)
            dataset_out = 0.5 * (np.tanh(0.01 * ((dataset - m) / std)) + 1)

    else:
        scalers = load(path2scaler)
        dataset_out = scalers.transform(dataset)

    return dataset_out


# ----------------------------------------------------------------------------------------------------------------------

def scaler_reverse(path2scaler: str,
                   params_dict: dict,
                   dataset: np.array) -> np.array:
    """Rescaled dataset to physical quantities.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param dataset:             dataset which should get rescaled
    :type dataset: np.array
    :return:                    rescaled dataset
    :rtype: np.array
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


# ----------------------------------------------------------------------------------------------------------------------

def create_dataset_separation_run(data_test: np.array,
                                  params_dict: dict,
                                  start: int,
                                  duration: int,
                                  nn_mode: str) -> tuple:
    """Creates a dataset to test the neural network against actual vehicle data.

    :param data_test:           vehicle input data of test data file
    :type data_test: np.array
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param start:               start index (row) of test data file
    :type start: int
    :param duration:            duration of one test setion (as timesteps)
    :type duration: int
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return:                    vehicle input data of test data file
    :rtype: tuple
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    initials = data_test[start:start + input_timesteps, :]

    if nn_mode == "feedforward":
        initials = np.reshape(initials, (1, input_timesteps * input_shape))

    elif nn_mode == "recurrent":
        initials = np.reshape(initials, (1, input_timesteps, input_shape))

    # get vehicle input data of test data file
    steeringangle_rad = data_test[start + input_timesteps:start + duration, output_shape]

    torqueRL_Nm = data_test[start + input_timesteps:start + duration, output_shape + 1]
    torqueRR_Nm = data_test[start + input_timesteps:start + duration, output_shape + 2]

    brakepresF_bar = data_test[start + input_timesteps:start + duration, output_shape + 3]
    brakepresR_bar = data_test[start + input_timesteps:start + duration, output_shape + 4]

    return initials, steeringangle_rad, torqueRL_Nm, torqueRR_Nm, brakepresF_bar, brakepresR_bar


# ----------------------------------------------------------------------------------------------------------------------

def extract_part(datax,
                 params_dict: dict,
                 data_infox,
                 z):

    summ = np.sum(data_infox[0:1 + z, :])
    data_part = datax[summ - data_infox[z, 0]:summ, :]
    labels_part = data_part[2 * params_dict['NeuralNetwork_Settings']['input_timesteps']
                            - 1::params_dict['NeuralNetwork_Settings']['input_timesteps'],
                            0:params_dict['NeuralNetwork_Settings']['output_shape']]

    data_part = data_part[0:len(data_part) - params_dict['NeuralNetwork_Settings']['input_timesteps'], :]

    return np.array(data_part), np.array(labels_part)


# ----------------------------------------------------------------------------------------------------------------------

def create_dataset_separation(path_dict: dict,
                              params_dict: dict,
                              data: dict,
                              nn_mode: str) -> tuple:
    """Creates a training and validation dataset out of the provided data.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param data:                complete dataset which should be used for NN training
    :type data: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return:                    training and validation datasets
    :rtype: tuple
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape']
    output_shape = params_dict['NeuralNetwork_Settings']['output_shape']
    input_timesteps = params_dict['NeuralNetwork_Settings']['input_timesteps']

    # count training data files
    file_counting = 0
    filepath = path_dict['path2inputs_trainingdata']

    if os.path.exists(filepath):

        for file in os.listdir(filepath):

            if file.startswith('data_to_train'):
                file_counting += 1

    # generate datasets
    lengthsum = 0
    lengthsumtwo = 0
    lengthsumtwolabels = 0

    for m in range(0, file_counting):
        lengthsum += (len(data[m]) - input_timesteps)

    data_train = np.zeros((lengthsum * input_timesteps, input_shape))
    data_labels = np.zeros((lengthsum, output_shape))

    # generate the training dataset which contains 'input_timesteps' number of previous data points
    for u in range(0, file_counting):
        data_labels[lengthsumtwolabels:lengthsumtwolabels + len(data[u]) - input_timesteps] \
            = (data[u])[input_timesteps:, 0:output_shape]

        for pp in range(0, len(data[u]) - input_timesteps):
            idx = lengthsumtwo + pp * input_timesteps
            data_train[idx:idx + input_timesteps, :] = (data[u])[pp:pp + input_timesteps, :]

        lengthsumtwolabels += ((len(data[u]) - input_timesteps))
        lengthsumtwo += ((len(data[u]) - input_timesteps) * input_timesteps)

    # reshape training dataset
    if nn_mode == "feedforward":
        data_train = np.reshape(data_train, (len(data_labels), input_timesteps * input_shape))

    elif nn_mode == "recurrent":
        data_train = np.reshape(data_train, (len(data_train) // input_timesteps, input_timesteps, input_shape))

    # shuffle training dataset
    indices = np.arange(data_train.shape[0])

    if params_dict['General']['shuffle_mode']:
        np.random.RandomState(params_dict['General']['shuffle_number']).shuffle(indices)

    else:
        np.random.shuffle(indices)

    data_labels = data_labels[indices]
    data_train = np.reshape(data_train[indices], (len(data_labels) * input_timesteps, input_shape))

    # split provided dataset into training and validation datasets
    p = int(len(data_train) * (1 - params_dict['NeuralNetwork_Settings']['val_split']))
    mod = p % 5
    p = p - mod

    train_x = scaler(path_dict=path_dict,
                     params_dict=params_dict,
                     dataset=data_train[0:p, :])

    temp = np.zeros((len(data_labels), input_shape))
    temp[:, 0:output_shape] = data_labels

    temp = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                      params_dict=params_dict,
                      dataset=temp)

    data_labels = temp[:, 0:output_shape]

    # prepare training data
    if nn_mode == "feedforward":
        train_x = np.reshape(train_x, (p // input_timesteps, input_timesteps * input_shape))

    elif nn_mode == "recurrent":
        train_x = np.reshape(train_x, (p // input_timesteps, input_timesteps, input_shape))

    train_y = data_labels[0:(p // input_timesteps), :]

    # prepare validation data
    val_x = data_train[p:len(data_train), :]
    val_x = scaler_run(path2scaler=path_dict['filepath2scaler_save'],
                       params_dict=params_dict,
                       dataset=val_x)

    if nn_mode == "feedforward":
        val_x = np.reshape(val_x, ((len(data_train) - p * input_timesteps), input_timesteps * input_shape))

    elif nn_mode == "recurrent":
        val_x = np.reshape(val_x, ((len(data_train) - p) // input_timesteps, input_timesteps, input_shape))

    val_y = data_labels[(p // input_timesteps):len(data_labels), :]

    return (train_x, train_y), (val_x, val_y)
