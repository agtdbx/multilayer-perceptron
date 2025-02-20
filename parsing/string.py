# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    string.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:10 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 18:41:58 by aderouba         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

def strToInt(string: str) -> int:
    """
    Function to convert a string to an integer.
    @param  string  The string input.
    @return The number in string or None in case of error.
    """
    try:
        return int(string)
    except:
        return None


def strToFloat(string: str) -> float:
    """
    Function to convert a string to a float.
    @param  string  The string input.
    @return The number in string or None in case of error.
    """
    try:
        return float(string)
    except:
        return None


def strToResult(string: str) -> int:
    """
    Function to convert a string to a int representation of the result.
    B -> 0
    M -> 1
    @param  string  The string input.
    @return The result as an int or None in case of error.
    """
    if len(string) != 1 or string not in "BM":
        return None

    if string == "B":
        return 0
    return 1
