# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    sigmoid.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:28 by aderouba          #+#    #+#              #
#    Updated: 2025/02/28 20:09:48 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from network.matrix import  Matrix

from math import exp


def sigmoid(Z: Matrix) -> Matrix:
    """
    Activation function sigmoid for weighted sum.
    @param  Z   weighted sum matrix.
    @return Activation matrix.
    """
    a = Matrix((Z.width, Z.height))
    for y in range(Z.height):
        for x in range(Z.width):
            try:
                a[x, y] = 1 / (1 + exp(Z[x, y]))
            except:
                a[x, y] = 0

    return a
