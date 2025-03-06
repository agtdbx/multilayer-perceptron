# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    string.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:10 by aderouba          #+#    #+#              #
#    Updated: 2025/03/06 18:08:44 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from parsing.print import   printError

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


def strToVector(string: str) -> list[float]:
    """
    Function to convert a string to a vector of float.
    @param  string  The string input.
    @return The result as a list of float, or None in case of error.
    """
    result = string.split(',')

    for i in range(len(result)):
        value = strToFloat(result[i])
        if value == None:
            return None
        result[i] = value

    return result


def strToLayers(string: str) -> list[int]:
    """
    Function to convert a string to network layers as a list of int.
    Print it's error.
    @param  string  The string input.
    @return The result as a list of int, or None in case of error.
    """
    result = string.split(',')
    if len(result) == 0:
        printError("It must have at least one hidden layer")
        return None

    for i in range(len(result)):
        value = strToInt(result[i])
        if value == None:
            printError("Layer dimension must be an interger")
            return None
        if value <= 0:
            printError("Layer dimension must be greater than 0")
            return None
        result[i] = value

    return result


def strToMatrix(string: str) -> list[list[float]]:
    """
    Function to convert a string to a matrix of float.
    @param  string  The string input.
    @return The result as a list of list of float, or None in case of error.
    """
    result = string.split(';')

    for i in range(len(result)):
        value = strToVector(result[i])
        if value == None:
            return None
        result[i] = value

    return result


def vectorToStr(vector: list[float]) -> str:
    """
    Function to convert a vector of float to a string.
    @param  vector  The vector as a list of float.
    @return The string vector.
    """
    result = ""

    for i in range(len(vector)):
        if i > 0:
            result += ','
        result += str(vector[i])

    return result


def layersToStr(layers: list[int]) -> str:
    """
    Function to convert network layers as a vector of int to a string.
    @param  vector  The layers as a list of int.
    @return The string layers.
    """
    result = ""

    for i in range(len(layers)):
        if i > 0:
            result += ','
        result += str(layers[i])

    return result


def matrixToStr(matrix: str) -> list[list[float]]:
    """
    Function to convert a matrix of float to a string.
    @param  matrix  The string input.
    @return The string matrix.
    """
    result = ""

    for i in range(len(matrix)):
        if i > 0:
            result += ';'
        result += vectorToStr(matrix[i])

    return result
