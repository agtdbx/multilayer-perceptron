# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:28 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 15:14:28 by aderouba         ###   ########.fr        #
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
