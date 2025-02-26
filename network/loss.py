# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/02/26 14:36:30 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from math import log

def computeLoss(results: list[tuple[float]],
                NNresults: list[list[float]]) -> float:
    """
    Function to compute the loss of the network by cross entropy.
    @param  results     The list of wanted results.
    @param  NNresults   The list of neural networks results.
    @return The loss.
    """
    loss = 0.0
    outputSize = len(results[0])
    nbData = len(results)

    for i in range(nbData):
        # Compare network output with wanted one
        for j in range(outputSize):
            y = results[i][j]
            p = NNresults[i][j]
            loss += y * log(p, 10) + (1 - y) * log(1 - p, 10)

    # Compute the average of the loss
    loss /= -nbData

    return loss


def computeOneLoss(result: tuple[float],
                   NNresult: list[float]) -> float:
    """
    Function to compute the loss of the network by cross entropy.
    @param  results     The list of wanted results.
    @param  NNresults   The list of neural networks results.
    @return The loss.
    """
    loss = 0.0
    outputSize = len(result)

    # Compare network output with wanted one
    for i in range(outputSize):
        y = result[i]
        p = NNresult[i]
        loss += y * log(p, 10) + (1 - y) * log(1 - p, 10)

    return -loss


def computeOneLossDerivate(result: tuple[float],
                           NNresult: list[float]) -> float:
    """
    Function to compute the derivate loss of the network by cross entropy.
    @param  results     The list of wanted results.
    @param  NNresults   The list of neural networks results.
    @return The derivate loss.
    """
    loss = 0.0
    outputSize = len(result)
    ln10 = log(10, 2)

    # Compare network output with wanted one
    for i in range(outputSize):
        y = result[i]
        p = NNresult[i]
        loss += y / (ln10 * p) + (y - 1) / (ln10 * (1 - p))

    return -loss
