import numpy as np
import matplotlib.pyplot as plt
import os.path
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def plot_run(path_dict: dict,
             params_dict: dict,
             counter,
             start):
    """Plots test results of comparison between neural network and provided vehicle data.

    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param counter: [description]
    :type counter: [type]
    :param start: [description]
    :type start: [type]
    """

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 1:
        filename_model = 'prediction_result'

    if params_dict['NeuralNetwork_Settings']['run_file_mode'] == 2:
        filename_model = 'prediction_result_recurrent'

    filepath2results = os.path.join(path_dict['path2results_matfiles'], filename_model + str(counter) + '.csv')

    # load results
    with open(filepath2results, 'r') as fh:
        results = np.loadtxt(fh)

    # load label data
    with open(path_dict['filepath2inputs_testdata'] + '.csv', 'r') as fh:
        labels = np.loadtxt(fh, delimiter=',')

    yaw_result = np.array(results[:, 0])
    vy_result = np.array(results[:, 1])
    vx_result = np.array(results[:, 2])
    ay_result = np.array(results[:, 3])
    ax_result = np.array(results[:, 4])

    yaw_label = np.array(labels[start:params_dict['Test']['run_timespan'] + start, 0])
    vy_label = np.array(labels[start:params_dict['Test']['run_timespan'] + start, 1])
    vx_label = np.array(labels[start:params_dict['Test']['run_timespan'] + start, 2])
    ay_label = np.array(labels[start:params_dict['Test']['run_timespan'] + start, 3])
    ax_label = np.array(labels[start:params_dict['Test']['run_timespan'] + start, 4])

    yaw_diff = yaw_label - yaw_result
    vy_diff = vy_label - vy_result
    vx_diff = vx_label - vx_result
    ay_diff = ay_label - ay_result
    ax_diff = ax_label - ax_result

    scaler_results = MinMaxScaler(feature_range=(0, 1))

    yaw_result = np.reshape(yaw_result, (len(yaw_result), 1))
    vy_result = np.reshape(vy_result, (len(vy_result), 1))
    vx_result = np.reshape(vx_result, (len(vx_result), 1))
    ay_result = np.reshape(ay_result, (len(ay_result), 1))
    ax_result = np.reshape(ax_result, (len(ax_result), 1))

    yaw_label = np.reshape(yaw_label, (len(yaw_label), 1))
    vy_label = np.reshape(vy_label, (len(vy_label), 1))
    vx_label = np.reshape(vx_label, (len(vx_label), 1))
    ay_label = np.reshape(ay_label, (len(ay_label), 1))
    ax_label = np.reshape(ax_label, (len(ax_label), 1))

    scaler_temp_result = np.concatenate((yaw_result, vy_result, vx_result, ay_result, ax_result), axis=1)
    scaler_temp_label = np.concatenate((yaw_label, vy_label, vx_label, ay_label, ax_label), axis=1)
    scaler_temp = np.concatenate((scaler_temp_result, scaler_temp_label), axis=0)

    scaler_results = scaler_results.fit(scaler_temp)
    scaler_temp_result = scaler_results.transform(scaler_temp_result)
    scaler_temp_label = scaler_results.transform(scaler_temp_label)

    yaw_result_scaled = scaler_temp_result[:, 0]
    vy_result_scaled = scaler_temp_result[:, 1]
    vx_result_scaled = scaler_temp_result[:, 2]
    ay_result_scaled = scaler_temp_result[:, 3]
    ax_result_scaled = scaler_temp_result[:, 4]

    yaw_label_scaled = scaler_temp_label[:, 0]
    vy_label_scaled = scaler_temp_label[:, 1]
    vx_label_scaled = scaler_temp_label[:, 2]
    ay_label_scaled = scaler_temp_label[:, 3]
    ax_label_scaled = scaler_temp_label[:, 4]

    print('\n')
    print('MSE AND MAE OF UNSCALED VALUES')
    print('\n')

    print('MSE of Yaw No.   ' + str(counter) + ':   ' + str(round(mean_squared_error(yaw_label, yaw_result), 8)))
    print('MSE of Vy No.    ' + str(counter) + ':   ' + str(round(mean_squared_error(vy_label, vy_result), 8)))
    print('MSE of Vx No.    ' + str(counter) + ':   ' + str(round(mean_squared_error(vx_label, vx_result), 8)))
    print('MSE of ay No.    ' + str(counter) + ':   ' + str(round(mean_squared_error(ay_label, ay_result), 8)))
    print('MSE of ax No.    ' + str(counter) + ':   ' + str(round(mean_squared_error(ax_label, ax_result), 8)) + '\n')

    print('MAE of Yaw No    ' + str(counter) + ':   ' + str(round(mean_absolute_error(yaw_label, yaw_result), 8)))
    print('MAE of Vy No.    ' + str(counter) + ':   ' + str(round(mean_absolute_error(vy_label, vy_result), 8)))
    print('MAE of Vx No.    ' + str(counter) + ':   ' + str(round(mean_absolute_error(vx_label, vx_result), 8)))
    print('MAE of ay No.    ' + str(counter) + ':   ' + str(round(mean_absolute_error(ay_label, ay_result), 8)))
    print('MAE of ax No.    ' + str(counter) + ':   ' + str(round(mean_absolute_error(ax_label, ax_result), 8)))

    print('\n')
    print('MSE AND MAE OF SCALED VALUES')
    print('\n')

    print('MSE of Yaw No.   ' + str(counter) + ':   '
          + str(round(mean_squared_error(yaw_label_scaled, yaw_result_scaled), 8)))
    print('MSE of Vy No.    ' + str(counter) + ':   '
          + str(round(mean_squared_error(vy_label_scaled, vy_result_scaled), 8)))
    print('MSE of Vx No.    ' + str(counter) + ':   '
          + str(round(mean_squared_error(vx_label_scaled, vx_result_scaled), 8)))
    print('MSE of ay No.    ' + str(counter) + ':   '
          + str(round(mean_squared_error(ay_label_scaled, ay_result_scaled), 8)))
    print('MSE of ax No.    ' + str(counter) + ':   '
          + str(round(mean_squared_error(ax_label_scaled, ax_result_scaled), 8)) + '\n')

    print('MAE of Yaw No    ' + str(counter) + ':   '
          + str(round(mean_absolute_error(yaw_label_scaled, yaw_result_scaled), 8)))
    print('MAE of Vy No.    ' + str(counter) + ':   '
          + str(round(mean_absolute_error(vy_label_scaled, vy_result_scaled), 8)))
    print('MAE of Vx No.    ' + str(counter) + ':   '
          + str(round(mean_absolute_error(vx_label_scaled, vx_result_scaled), 8)))
    print('MAE of ay No.    ' + str(counter) + ':   '
          + str(round(mean_absolute_error(ay_label_scaled, ay_result_scaled), 8)))
    print('MAE of ax No.    ' + str(counter) + ':   '
          + str(round(mean_absolute_error(ax_label_scaled, ax_result_scaled), 8)))

    # plot and save comparsion between NN predicted and actual vehicle state
    plot_and_save(params_dict, yaw_result, yaw_label, yaw_diff, 'Yaw rate rad/s',
                  os.path.join(path_dict['path2results_figures'], 'yaw' + str(counter) + '.pdf'))
    plot_and_save(params_dict, vy_result, vy_label, vy_diff, 'Velocity Vy m/s',
                  os.path.join(path_dict['path2results_figures'], 'vy' + str(counter) + '.pdf'))
    plot_and_save(params_dict, vx_result, vx_label, vx_diff, 'Velocity Vx m/s',
                  os.path.join(path_dict['path2results_figures'], 'vx' + str(counter) + '.pdf'))
    plot_and_save(params_dict, ay_result, ay_label, ay_diff, 'Velocity ay m/s2',
                  os.path.join(path_dict['path2results_figures'], 'ay' + str(counter) + '.pdf'))
    plot_and_save(params_dict, ax_result, ax_label, ax_diff, 'Velocity ax m/s2',
                  os.path.join(path_dict['path2results_figures'], 'ax' + str(counter) + '.pdf'))


# ----------------------------------------------------------------------------------------------------------------------

def plot_and_save(params_dict: dict,
                  inp_1,
                  inp_2,
                  inp_3,
                  value,
                  savename):
    """Plots and saves comparison of NN predicted and actual vehicle states values.

    :param params_dict:     dictionary which contains paths to all relevant folders and files of this module
    :type params_dict: dict
    :param inp_1:           NN predicted vehicle state value
    :type inp_1: [type]
    :param inp_2:           actual vehicle state value from test data
    :type inp_2: [type]
    :param inp_3:           calculated difference between predicted and actual vehicle state
    :type inp_3: [type]
    :param value:           name of compared vehicle state value
    :type value: [type]
    :param savename:        filename where to save plot
    :type savename: [type]
    """

    fig, (ax1, ax2) = plt.subplots(2, 1)

    ax1.plot(inp_1, label='Result', color='tab:orange')
    ax1.plot(inp_2, label='Label', color='tab:blue')
    ax2.plot(inp_3, label='Difference', color='tab:blue', linewidth=1.0)

    ax1.set_ylabel(value)
    ax2.set_ylabel('Difference value')
    ax1.set_xlabel('Time steps (8 ms)')
    ax2.set_xlabel('Time steps (8 ms)')
    ax1.legend()
    ax2.legend()

    if params_dict['General']['plot_result']:
        plt.show()

    if params_dict['General']['save_figures']:
        fig.savefig(savename, format='pdf')


# ----------------------------------------------------------------------------------------------------------------------

def plot_mse(path_dict: dict,
             params_dict: dict,
             histories):
    """Plots the MSE of comparion between the neural network's vehicle state output and the real vehicle state.

    :param path_dict:       dictionary which contains paths to all relevant folders and files of this module
    :type path_dict: dict
    :param params_dict:    dictionary which contains all parameters necessary to run this module
    :type params_dict: dict
    :param histories: [description]
    :type histories: [type]
    """

    # Plot training & validation accuracy values
    fig = plt.figure()

    plt.plot(histories.history[params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']])
    plt.plot(histories.history['val_' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function']])

    plt.axis([0, params_dict['NeuralNetwork_Settings']['epochs'],
              params_dict['General']['min_scale_plot'], params_dict['General']['max_scale_plot']])

    plt.xlabel('Epoche')
    plt.ylabel(params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'])

    plt.title('Model ' + params_dict['NeuralNetwork_Settings']['Optimizer']['loss_function'])
    plt.legend(['Training loss', 'Validation loss'], loc='upper left')
    plt.show()

    fig.savefig(os.path.join(path_dict['path2results_figures'], 'loss_function.pdf'), format='pdf')
