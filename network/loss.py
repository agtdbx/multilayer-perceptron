# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/02/28 21:28:49 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import          EPSILON
from network.matrix import  Matrix

from math import log


def logLoss(A: Matrix, Y: Matrix) -> float:
    """
    Function to compute the cost of a given output.
    @param  A   Network output.
    @param  Y   Wanted output.
    @return The cost of the output.
    """
    lossSum = 0
    for i in range(Y.height):
        a = A[0, i]
        y = Y[0, i]
        lossSum += -y * log(a + EPSILON) - (1 - y) * log(1 - a + EPSILON)

    return 1 / Y.height * lossSum
