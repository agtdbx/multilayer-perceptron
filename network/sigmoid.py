# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:28 by aderouba          #+#    #+#              #
#    Updated: 2025/02/27 12:35:23 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from math import exp

def sigmoid(x: float) -> float:
    """
    Function to put any real number between 0 and 1.
    @param  x   Real number in input.
    @return Real number between 0 and 1.
    """
    try:
        return 1 / (1 + exp(-x))
    except:
        return 0


def sigmoidDerivate(x: float) -> float:
    """
    Derivation of the function to put any real number between 0 and 1.
    @param  x   Real number in input.
    @return Real number between 0 and 1.
    """
    try:
        tmp = exp(x)
        return tmp / ((tmp + 1) ** 2)
    except:
        return 0
