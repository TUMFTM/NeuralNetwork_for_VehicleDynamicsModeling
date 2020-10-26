import os.path
import datetime

"""
Created by: Leonhard Hermansdorfer
Created on: 10.04.2019
"""


def manage_paths() -> dict:
    """Creates a dictionary which contains paths to all relevant module folders and files.

    Input
    ---

    Output
    :return: dictionary which contains paths to all relevant folders and files of this module
    :rtype: dict
    """

    # get path to top-level module
    path_root2module = os.path.join(os.path.dirname(os.path.abspath(__file__)).split('Neural_Network')[0],
                                    'Neural_Network')

    # create datetime-dependent paths
    path_day = datetime.datetime.now().strftime('%Y_%m_%d')
    path_datetime = datetime.datetime.now().strftime('%H_%M_%S')

    # create path dictionary for easy hand-over
    path_dict = dict()

    # specify paths on top level according to folder structure ---------------------------------------------------------

    path_dict['path2inputs'] = os.path.join(path_root2module, 'inputs')
    path_dict['path2outputs'] = os.path.join(path_root2module, 'outputs')
    path_dict['path2params'] = os.path.join(path_root2module, 'params')

    # specify paths to folders
    path_dict['path2inputs_trainingdata'] = os.path.join(path_dict['path2inputs'], 'trainingdata')
    path_dict['path2inputs_trainedmodels'] = os.path.join(path_dict['path2inputs'], 'trainedmodels')

    path_dict['path2results'] = os.path.join(path_dict['path2outputs'], path_day, path_datetime)
    path_dict['path2results_matfiles'] = os.path.join(path_dict['path2results'], 'matfiles')
    path_dict['path2results_figures'] = os.path.join(path_dict['path2results'], 'figures')

    # specify paths to certain files -----------------------------------------------------------------------------------

    path_dict['filepath2params'] = os.path.join(path_dict['path2params'], 'parameters.toml')

    # file name of input data: training and test data, trained models
    path_dict['filename_trainingdata'] = 'data_to_train'
    # path_dict['filepath2inputs_labels'] = os.path.join(path_dict['path2inputs_labels'], 'data_to_train_labels.mat')
    path_dict['filepath2inputs_testdata'] = os.path.join(path_dict['path2inputs_trainingdata'], 'data_to_run')

    path_dict['filepath2scaler_load'] = os.path.join(path_dict['path2inputs_trainedmodels'], 'scaler.plk')
    path_dict['filepath2inputs_trainedmodel_ff'] = os.path.join(path_dict['path2inputs_trainedmodels'],
                                                                'keras_model.h5')
    path_dict['filepath2inputs_trainedmodel_recurr'] = os.path.join(path_dict['path2inputs_trainedmodels'],
                                                                    'keras_model_recurrent.h5')

    # file name of output data: trained models
    path_dict['filepath2scaler_save'] = os.path.join(path_dict['path2results'], 'scaler.plk')
    path_dict['filepath2results_trainedmodel_ff'] = os.path.join(path_dict['path2results'], 'keras_model.h5')
    path_dict['filepath2results_trainedmodel_recurr'] = os.path.join(path_dict['path2results'],
                                                                     'keras_model_recurrent.h5')

    # check if folders exist and create new ones -----------------------------------------------------------------------

    # setup inputs folder
    if not os.path.exists(path_dict['path2inputs']):
        os.mkdir(path_dict['path2inputs'])

    if not os.path.exists(path_dict['path2inputs_trainingdata']):
        os.mkdir(path_dict['path2inputs_trainingdata'])

    if not os.path.exists(path_dict['path2inputs_trainedmodels']):
        os.mkdir(path_dict['path2inputs_trainedmodels'])

    # setup outputs folder
    if not os.path.exists(path_dict['path2outputs']):
        os.mkdir(path_dict['path2outputs'])

    if not os.path.exists(path_dict['path2results']):
        os.makedirs(path_dict['path2results'])

    if not os.path.exists(path_dict['path2results_matfiles']):
        os.mkdir(path_dict['path2results_matfiles'])

    if not os.path.exists(path_dict['path2results_figures']):
        os.mkdir(path_dict['path2results_figures'])

    return path_dict


# ----------------------------------------------------------------------------------------------------------------------
# Testing --------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':

    test_dict = manage_paths()

    print(test_dict)
