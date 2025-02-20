# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    splitData.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: aderouba <aderouba@student.42.fr>          +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 17:54:23 by aderouba          #+#    #+#              #
#    Updated: 2025/02/20 19:33:32 by aderouba         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import          NUMBER_OF_COLUMN
from parsing.file import    readFile, writeInFile
from parsing.parse import   parseData
from parsing.print import   printError
from parsing.string import  strToFloat

import sys

# Check number of parameter
argc = len(sys.argv)
if argc != 3:
    printError("bad number of parameter.")
    print("Usage : python3 splitData.py <data_path.cvs> <percentage of data for training>")
    exit(1)


# Get data parameter
dataPath = sys.argv[1]
dataContent = readFile(dataPath)
if dataContent == None:
    exit(1)

parsedData = parseData(dataContent)
if parsedData == None:
    exit(1)


# Get percentage parameter
percentageStr = sys.argv[2]
percentage = strToFloat(percentageStr)
if percentage == None:
    printError(f"'{percentageStr}' isn't a float")
    exit(1)

if percentage < 0.1 or percentage > 99.9:
    printError(f"percentage must be between 1 and 99")
    exit(1)


# Split data in two part
nbData = len(parsedData)
print(f"{nbData} data")
nbTrainingData = int(nbData * (percentage / 100) + 0.5)
print(f"{nbTrainingData} data for training")
nbValidationData = nbData - nbTrainingData
print(f"{nbValidationData} data for validation")

trainingData = ""
validationData = ""
resultChar = "BM"
for i in range(nbData):
    values = parsedData[i]
    strValues = f"{values[0][0]},{resultChar[values[1]]}"
    for j in range(1, NUMBER_OF_COLUMN - 1):
        value = values[0][j]
        intVal = int(value)
        if intVal == value:
            strValues += f",{intVal}"
        else:
            strValues += f",{value}"
    strValues += '\n'

    if i < nbTrainingData:
        trainingData += strValues
    else:
        validationData += strValues

writeInFile("trainingData.csv", trainingData)
writeInFile("validationData.csv", validationData)
