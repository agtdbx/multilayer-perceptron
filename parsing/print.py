# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    print.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:12 by aderouba          #+#    #+#              #
#    Updated: 2025/02/21 14:55:39 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys

def printError(message: str, prePrint: str = "Error"):
    """
    Function for print an error on stderr.
    @param  message The message to print.
    @param  prePrint The string before the :, "Error" by default.
    """
    print(f"{prePrint} : {message}", file=sys.stderr)
