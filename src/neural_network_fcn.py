from tensorflow import keras

# custom modules
import helper_funcs_NN

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def create_nnmodel(path_dict: dict,
                   params_dict: dict,
                   nn_mode: str):
    """Creates a new neural network model or loads an existing one.

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param nn_mode:             Neural network mode which defines type of NN (feedforward or recurrent)
    :type nn_mode: str
    :return: [description]
    :rtype: [type]
    """

    if not nn_mode == "feedforward" or not nn_mode == "recurrent":
        ValueError('unknown "neural network mode"; must be either "feedforard" or "recurrent"')

    if nn_mode == "feedforward":
        filepath2inputs_trainedmodel = path_dict['filepath2inputs_trainedmodel_ff']

    elif nn_mode == "recurrent":
        filepath2inputs_trainedmodel = path_dict['filepath2inputs_trainedmodel_recurr']

    if params_dict['General']['bool_load_existingmodel']:
        print('LOAD ALREADY CREATED MODEL FOR FURTHER TRAINING')
        model_create = keras.models.load_model(filepath2inputs_trainedmodel)

    else:
        print('CREATE NEW MODEL')

        if nn_mode == "feedforward":
            model_create = create_model_feedforward(path_dict=path_dict,
                                                    params_dict=params_dict)

        elif nn_mode == "recurrent":
            model_create = create_model_recurrent(path_dict=path_dict,
                                                  params_dict=params_dict)

        optimizer = helper_funcs_NN.src.select_optimizer.select_optimizer(
            optimizer=params_dict['NeuralNetwork_Settings']['Optimizer']['optimizer_set'],
            learning_rate=params_dict['NeuralNetwork_Settings']['learning_rate'],
            clipnorm=params_dict['NeuralNetwork_Settings']['Optimizer']['clipnorm'])

        model_create.compile(optimizer=optimizer,
                             loss=params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'],
                             metrics=[keras.metrics.mae, keras.metrics.mse])

        model_create.summary()

    return model_create


# ----------------------------------------------------------------------------------------------------------------------

def create_model_feedforward(path_dict: dict,
                             params_dict: dict):
    """Set up a new feedforward NN model

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :return:                    neural network model
    :rtype: [type]
    """

    print('CREATE FEEDFORWARD NEURAL NETWORK')

    model_create = keras.models.Sequential()

    if params_dict['NeuralNetwork_Settings']['Initializer'] == "he":
        kernel_init = keras.initializers.he_uniform(seed=True)

    elif params_dict['NeuralNetwork_Settings']['Initializer'] == "glorot":
        kernel_init = keras.initializers.GlorotUniform(seed=True)

    reg_dense = keras.regularizers.l1_l2(params_dict['NeuralNetwork_Settings']['l1regularization'],
                                         params_dict['NeuralNetwork_Settings']['l2regularization'])

    input_shape = params_dict['NeuralNetwork_Settings']['input_shape'] \
        * params_dict['NeuralNetwork_Settings']['input_timesteps']

    model_create.add(
        keras.layers.Dense(input_shape=(input_shape,),
                           units=params_dict['NeuralNetwork_Settings']['Feedforward']['neurons_first_layer'],
                           use_bias=True,
                           bias_initializer='zeros',
                           activation=params_dict['NeuralNetwork_Settings']['Feedforward']['activation_1']))

    if params_dict['NeuralNetwork_Settings']['Feedforward']['leakyrelu'] == 1:
        model_create.add(keras.layers.LeakyReLU(alpha=0.2))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_1']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['Feedforward']['neurons_second_layer'],
                           bias_initializer='zeros',
                           use_bias=True,
                           activation=params_dict['NeuralNetwork_Settings']['Feedforward']['activation_2'],
                           ))

    if params_dict['NeuralNetwork_Settings']['Feedforward']['leakyrelu'] == 1:
        model_create.add(keras.layers.LeakyReLU(alpha=0.2))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['output_shape'], activation='linear'))

    return model_create


# ----------------------------------------------------------------------------------------------------------------------

def create_model_recurrent(path_dict: dict,
                           params_dict: dict):
    """Set up a new recurrent NN model

    :param path_dict:           dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:         dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :return:                    neural network model
    :rtype: [type]
    """

    print('CREATE RECURRENT NEURAL NETWORK')

    model_create = keras.models.Sequential()

    if params_dict['NeuralNetwork_Settings']['Initializer'] == "he":
        kernel_init = keras.initializers.he_uniform(seed=True)

    elif params_dict['NeuralNetwork_Settings']['Initializer'] == "glorot":
        kernel_init = keras.initializers.GlorotUniform(seed=True)

    reg_layer = keras.regularizers.l1_l2(params_dict['NeuralNetwork_Settings']['l1regularization'],
                                         params_dict['NeuralNetwork_Settings']['l2regularization'])

    # load specified recurrent layer type (from parameter file)
    if params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'GRU':
        recurrent_mode = keras.layers.GRU

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'LSTM':
        recurrent_mode = keras.layers.LSTM

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'SimpleRNN':
        recurrent_mode = keras.layers.SimpleRNN

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'ConvLSTM2D':
        recurrent_mode = keras.layers.ConvLSTM2D

    elif params_dict['NeuralNetwork_Settings']['Recurrent']['recurrent_mode'] == 'RNN':
        recurrent_mode = keras.layers.RNN

    model_create.add(
        recurrent_mode(input_shape=(params_dict['NeuralNetwork_Settings']['input_timesteps'],
                                    params_dict['NeuralNetwork_Settings']['input_shape']),
                       units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_first_layer_recurrent'],
                       return_sequences=False,
                       use_bias=True,
                       bias_initializer='zeros',
                       activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent']))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_1']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['Recurrent']['neurons_second_layer_recurrent'],
                           use_bias=True,
                           bias_initializer='zeros',
                           activation=params_dict['NeuralNetwork_Settings']['Recurrent']['activation_dense_recurrent']))

    if params_dict['NeuralNetwork_Settings']['bool_use_dropout']:
        model_create.add(keras.layers.Dropout(params_dict['NeuralNetwork_Settings']['drop_2']))

    model_create.add(
        keras.layers.Dense(units=params_dict['NeuralNetwork_Settings']['output_shape'], activation='linear'))

    return model_create
