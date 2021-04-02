import toml

"""
Created by: Leonhard Hermansdorfer
Created on: 01.12.2018
"""


def handle_params(path_dict: dict) -> dict:
    """Reads the specified parameters located in the params folder.

    Input
    :param path_dict: dictionary which contains all paths to the module's folders and files
    :type path_dict: dict

    Error
    :raises FileNotFoundError: if parameter file is not found
    :raises ValueError: if parameter file does not contain any values

    Output
    :return: dictionary which contains all parameters
    :rtype: dict
    """

    # read interface config file for specified target ------------------------------------------------------------------
    try:
        with open(path_dict['filepath2params'], 'r') as fh:
            params = toml.load(fh)
    except FileNotFoundError:
        raise FileNotFoundError('Parameter file does not exist!') from None

    if not params:
        raise ValueError('Specified parameter file is empty!')

    # change None string in .toml file to None type in Python

    if params['NeuralNetwork_Settings']['Feedforward']['activation_1'] == 'None':
        params['NeuralNetwork_Settings']['Feedforward']['activation_1'] = None

    if params['NeuralNetwork_Settings']['Feedforward']['activation_2'] == 'None':
        params['NeuralNetwork_Settings']['Feedforward']['activation_2'] = None

    if params['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent'] == 'None':
        params['NeuralNetwork_Settings']['Recurrent']['activation_1_recurrent'] = None

    if params['NeuralNetwork_Settings']['Recurrent']['activation_dense_recurrent'] == 'None':
        params['NeuralNetwork_Settings']['Recurrent']['activation_dense_recurrent'] = None

    return params


# ----------------------------------------------------------------------------------------------------------------------
# Testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    import os.path

    # get path to top-level module
    path2module = os.path.join(os.path.dirname(os.path.abspath(__file__))
                               .split('NeuralNetwork_for_VehicleDynamicsModeling')[0],
                               'NeuralNetwork_for_VehicleDynamicsModeling')

    path_dict = dict()

    path_dict['filepath2params'] = os.path.join(path2module, "params", "parameters.toml")

    params_dict = handle_params(path_dict=path_dict)

    print(params_dict)
