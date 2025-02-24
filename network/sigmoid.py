# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:28 by aderouba          #+#    #+#              #
#    Updated: 2025/02/24 18:11:25 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from math import exp

def sigmoid(x: float) -> float:
    """
    Function to put any real number between 0 and 1.
    @param  x   Real number in input.
    @return Real number between 0 and 1.
    """
    return 1 / (1 + exp(-x))


def sigmoidDerivate(x: float) -> float:
    """
    Derivation of the function to put any real number between 0 and 1.
    @param  x   Real number in input.
    @return Real number between 0 and 1.
    """
    tmp = exp(x)
    return tmp / ((tmp + 1) ** 2)
