# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/02/27 13:45:09 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from math import log as logN


def log(x: float) -> float:
    """
    Function to make a save version of log10 that return 0 in case of error.
    @param  x   The number to make log.
    @return The log number.
    """
    if x <= 0:
        return 0
    return logN(x, 10)


def ln(x: float) -> float:
    """
    Function to make a save version of ln that return 0 in case of error.
    @param  x   The number to make log.
    @return The log number.
    """
    if x <= 0:
        return 0
    return logN(x)


def computeLoss(ys: list[float],
                ps: list[float]) -> float:
    """
    Function to compute the loss of the network by cross entropy.
    @param  results     The list of wanted results.
    @param  NNresults   The list of neural networks results.
    @return The loss.
    """
    loss = 0.0
    nbData = len(ys)

    for n in range(nbData):
        # Compare network output with wanted one
        loss += ys[n] * log(ps[n]) + (1 - ys[n]) * log(1 - ps[n])

    # Compute the average of the loss
    loss /= -nbData

    return loss


def computeOneLoss(y: float,
                   p: float) -> float:
    """
    Function to compute the loss of the network by cross entropy.
    @param  y   The wanted results.
    @param  p   The neural network result.
    @return The loss.
    """
    loss = 0.0

    # Compare network output with wanted one
    loss += y * log(p, 10) + (1 - y) * log(1 - p, 10)

    return -loss


def computeOneLossDerivate(y: float,
                           p: float) -> float:
    """
    Function to compute the derivate loss of the network by cross entropy.
    @param  y   The wanted results.
    @param  p   The neural network result.
    @return The derivate loss.
    """
    loss = 0.0
    ln10 = ln(10)

    # Compare network output with wanted one
    try:
        loss += y / (ln10 * p) + (y - 1) / (ln10 * (1 - p))
    except:
        pass

    return -loss
