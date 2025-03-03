# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    parse.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/13 17:27:15 by aderouba          #+#    #+#              #
#    Updated: 2025/03/03 16:18:16 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import          NUMBER_OF_COLUMN
from parsing.print import   printError
from parsing.string import  strToInt, strToFloat, strToResult,\
                            strToVector, strToMatrix


def parseData(fileContent: str) -> list[tuple[list[float], int, int]]:
    """
    Function for parse the data file content and return a list of these data,
    like this [(mileage, price),...].
    Print it's error.
    @param  fileContent The file content to parse
    @return The file parsed, or None in case of error.
    """
    # Split file by line
    lines = fileContent.split("\n")

    # Error if file is empty
    if len(lines) == 0:
        printError("file is empty")
        return None

    # Parse all other lines
    data = []
    lineNumber = 1
    for line in lines:
        # If a line is empty, skip it
        if line == "":
            continue

        # Parse a data line. Return None in case of error
        dataLine = _parseLineData(line, lineNumber)
        if dataLine == None:
            return None
        data.append(dataLine)
        lineNumber += 1

    return data


def parseNNparams(fileContent: str) -> tuple[list[float], list[tuple[list[list[float]], list[float]]]]:
    """
    Function for parse the network params file content and return a list tuples
    like [(weigth matrix, bias vector),...].
    Print it's error.
    @param  fileContent The file content to parse
    @return The file parsed as a tupple, or None in case of error.
    """
    # Split file by line
    lines = fileContent.split("\n")

    # Error if file is empty
    if len(lines) == 0:
        printError("file is empty")
        return None

    # Parse first line
    line = lines.pop(0)
    maxes = strToVector(line)
    if maxes == None:
        printError(f"[l{1};c1] - bad maxes format")
        return None

    # Parse all other lines
    result = []
    lineNumber = 2
    for line in lines:
        # If a line is empty, skip it
        if line == "":
            continue

        # Parse a data line. Return None in case of error
        dataLine = _parseLineNNparams(line, lineNumber)
        if dataLine == None:
            return None
        result.append(dataLine)
        lineNumber += 1

    return maxes, result


def _parseLineData(line: str, lineNumber: int) -> tuple[list[float], int, int]:
    """
    Function for parse a line from data as a tupple like (input lst, result, number).
    Print it's error.
    @param  line    The line to parse
    @param  line    The line number for the error message.
    @return The line parsed, or None in case or error.
    """
    values = line.split(',')
    if len(values) != NUMBER_OF_COLUMN:
        printError(f"[l{lineNumber};c1] - bad line format")
        return None

    inputs = []
    # Parse first int
    number = strToInt(values[0])
    if number == None:
        printError(f"[l{lineNumber};c1] - '{values[0]}' isn't an int")
        return None

    # Parse result
    result = strToResult(values[1])
    if result == None:
        printError(f"[l{lineNumber};c2] - '{values[1]}' isn't a result (B or M)")
        return None

    # Parse all remaining float
    for i in range(2, NUMBER_OF_COLUMN):
        value = strToFloat(values[i])
        if value == None:
            printError(f"[l{lineNumber};c{i + 1}] - '{values[i]}' isn't a float")
            return None
        inputs.append(value)

    return (inputs, result, number)


def _parseLineNNparams(line: str, lineNumber: int) -> tuple[list[float], int]:
    """
    Function for parse a line from network params as a tupple
    like (weights matrix, bias vector).
    Print it's error.
    @param  line    The line to parse
    @param  line    The line number for the error message.
    @return The line parsed, or None in case or error.
    """
    values = line.split('|')
    if len(values) != 2:
        printError(f"[l{lineNumber};c1] - bad line format")
        return None

    weigths = strToMatrix(values[0])
    if weigths == None:
        printError(f"[l{lineNumber};c1] - bad weight matrix format")
        return None

    biases = strToVector(values[1])
    if biases == None:
        printError(f"[l{lineNumber};c1] - bad bias vector format")
        return None

    return (weigths, biases)
