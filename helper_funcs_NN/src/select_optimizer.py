from tensorflow.keras import optimizers

"""
Created by: Rainer Trauth
Created on: 01.04.2020
"""


def select_optimizer(optimizer: str,
                     learning_rate: float,
                     clipnorm: float) -> object:
    """Sets the optimizer for the training process of the neural network using the specified parameters.

    Input
    :param optimizer: name of the optimizer algorithm to be used (options: ADAM, SGD, RMSPROP, ADADELTA, Nesterov-ADAM)
    :type optimizer: str
    :param learning_rate: initial learning rate of the optimzier for neural network training
    :type learning_rate: float
    :param clipnorm: gradients will be clipped when their L2 norm exceeds this value.
    :type clipnorm: float

    Output
    :return: abstract optimizer base class of the keras package
    :rtype: object
    """

    if optimizer == 'ADAM':
        optimizer = optimizers.Adam(lr=learning_rate,
                                    clipnorm=clipnorm)  # , beta_1=0.9, beta_2=0.999)

    if optimizer == 'SGD':
        optimizer = optimizers.SGD(lr=learning_rate,
                                   nesterov=True,
                                   clipnorm=clipnorm)

    if optimizer == 'RMSPROP':
        optimizer = optimizers.RMSprop(lr=learning_rate,
                                       momentum=0.9,
                                       clipnorm=clipnorm)

    if optimizer == 'ADADELTA':
        optimizer = optimizers.Adadelta(lr=learning_rate)  # , rho=0.95)

    if optimizer == 'Nesterov-ADAM':
        optimizer = optimizers.Nadam(lr=learning_rate,
                                     clipnorm=clipnorm)  # , beta_1=0.91, beta_2=0.997)

    return optimizer
