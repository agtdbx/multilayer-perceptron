# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    print.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:12 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 17:45:18 by aderouba         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys

def printError(message: str):
    """
    Function for print an error on stderr.
    @param  message The message to print
    """
    print(f"Error : {message}", file=sys.stderr)
