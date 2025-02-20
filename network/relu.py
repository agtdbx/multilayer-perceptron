# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    relu.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 16:10:09 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 16:11:45 by aderouba         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def relu(x: float) -> float:
    """
    Function to put any real number between 0 and +∞.
    @param  x   Real number in input.
    @return Real number between 0 and 1.
    """
    return max(0, x)
