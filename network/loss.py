# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/24 15:41:15 by gugus             #+#    #+#              #
#    Updated: 2025/03/07 19:09:48 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import  EPSILON

import numpy as np
from numpy import ndarray as Matrix
from math import log


def logLoss(A: Matrix, Y: Matrix) -> float:
    """
    Function to compute the cost of a given output.
    @param  A   Network output.
    @param  Y   Wanted output.
    @return The cost of the output.
    """
    return 1 / len(Y) * np.sum(-Y * np.log(A + EPSILON)\
                               - (1 - Y) * np.log(1 - A + EPSILON))
