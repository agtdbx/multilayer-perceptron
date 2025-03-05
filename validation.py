# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    validation.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/05 14:27:44 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  VALIDATION_DATA_FILE, PARAMS_FILE,\
                                    HIDDEN_LAYERS, INPUT_SIZE
from parsing.file import            readFile
from parsing.parse import           parseData, parseNNparams
from network.neuralNetwork import   NeuralNetwork

import numpy as np


# Get datas
dataPath = ""
dataContent = readFile(VALIDATION_DATA_FILE)
if dataContent == None:
    exit(1)

datas = parseData(dataContent)
if datas == None:
    exit(1)
nbData = len(datas)


# Get neural network params
NNparamsContent = readFile(PARAMS_FILE)
if NNparamsContent == None:
    exit(1)

parsedParams = parseNNparams(NNparamsContent)
if parsedParams == None:
    exit(1)
maxes, NNparams = parsedParams


# Put data into good format for training
lst_inputs = []
lst_result = []
for inputs, result, _ in datas:
    lst_inputs.append(inputs)
    lst_result.append([result])

X_validation = np.array(lst_inputs).T
Y_validation = np.array(lst_result).T


# Normalize data
for x in range(INPUT_SIZE):
    nb = maxes[x]
    for y in range(nbData):
        X_validation[x, y] /= nb


# Create neural network
dimensions = list(HIDDEN_LAYERS)
dimensions.insert(0, X_validation.shape[0])
dimensions.append(Y_validation.shape[0])

neuralNetwork = NeuralNetwork(dimensions)
try:
    neuralNetwork.loadParameters(NNparams)
except Exception as error:
    print(error)
    exit(1)


# Number of errors
result = neuralNetwork.predict(X_validation)

nbError = 0
for i in range(nbData):
    if Y_validation[0, i] != result[0, i]:
        nbError += 1

precision = (1 - (nbError / nbData)) * 100
print(f"{nbError} errors on {nbData} datas, {precision:.2f}% of precision")
