# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    splitData.py                                       :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 17:54:23 by aderouba          #+#    #+#              #
#    Updated: 2025/02/21 15:04:56 by gugus            ###   ########.fr        #
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

if percentage < 0 or percentage > 100:
    printError(f"percentage must be between 0 and 100")
    exit(1)


# Split data in two part
nbData = len(parsedData)
if nbData < 2:
    printError(f"the dataset must have at least 2 lines")
    exit(1)
print(f"{nbData} data")

nbTrainingData = int(nbData * (percentage / 100) + 0.5)
if nbTrainingData == 0:
    nbTrainingData = 1
    printError("training data must have at least 1 line", "Warning")
elif nbTrainingData == nbData:
    nbTrainingData = nbData - 1
    printError("validation data must have at least 1 line", "Warning")
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
