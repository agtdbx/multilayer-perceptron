# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/02/24 17:24:55 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from network.neuralNetwork import   NeuralNetwork

from math import log

def computeLoss(neuralNetwork: NeuralNetwork,
                datas: list[tuple[list[float], int]],
                nbData: int) -> float:
    """
    Function to compute the loss of the network by cross entropy.
    @param  neuralNetwork   The neural network object.
    @param  datas           The list of data.
    @param  nbData          The number of data.
    @return A tuple like loss.
    """
    # Create a result list of the size of neural network output
    loss = 0.0
    lstResult = [(1, 0), (0, 1)]

    for inputs, result in datas:
        # Compute network result
        NNresult = neuralNetwork.forward(inputs)

        # Compare network output with wanted one
        for i in range(neuralNetwork.outputSize):
            y = lstResult[result][i]
            p = NNresult[i]
            loss += y * log(p) + (1 - y) * log(1 - p)

    # Compute the average of the loss
    loss /= -nbData

    return loss
