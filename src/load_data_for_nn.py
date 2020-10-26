import numpy as np
import os

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def load_data(path2inputs_trainingdata: str,
              filename_trainingdata: str) -> np.array:
    """doc string
    """

    file_counting = 0

    if os.path.exists(path2inputs_trainingdata):

        for file in os.listdir(path2inputs_trainingdata):

            if file.startswith('data_to_train'):
                file_counting += 1

    data = [None] * file_counting

    for i in range(0, file_counting):

        with open(os.path.join(path2inputs_trainingdata, filename_trainingdata) + '_' + str(i) + '.csv', 'r') as fh:
            data[i] = np.loadtxt(fh, delimiter=',')

    print('LOADING DATA DONE')

    return data
