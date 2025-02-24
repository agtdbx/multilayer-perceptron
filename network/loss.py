# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/02/24 18:30:09 by gugus            ###   ########.fr        #
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
            loss += y * log(p) + (1 - y) * log(1 - p)

    # Compute the average of the loss
    loss /= -nbData

    return loss
