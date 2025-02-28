# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:28 by aderouba          #+#    #+#              #
#    Updated: 2025/02/28 15:51:35 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from network.matrix import  Matrix

from math import exp


def sigmoid(z: Matrix) -> Matrix:
    """
    Activation function sigmoid for weighted sum.
    @param  z   weighted sum matrix.
    @return Activation matrix.
    """
    a = Matrix((z.width, z.height))
    for y in z.height:
        try:
            a[1, y] = 1 / (1 + exp(z[1, y]))
        except:
            a[1, y] = 0

    return a
