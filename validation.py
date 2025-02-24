# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    validation.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/02/24 20:53:07 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  VALIDATION_DATA_FILE, PARAMS_FILE, LAYERS
from parsing.file import            readFile
from parsing.parse import           parseData, parseNNparams
from network.neuralNetwork import   NeuralNetwork

# Get datas
dataPath = ""
dataContent = readFile(VALIDATION_DATA_FILE)
if dataContent == None:
    exit(1)

datas = parseData(dataContent)
if datas == None:
    exit(1)


# Get neural network params
NNparamsContent = readFile(PARAMS_FILE)
if NNparamsContent == None:
    exit(1)

NNparams = parseNNparams(NNparamsContent)
if NNparams == None:
    exit(1)


# Create neural network
neuralNetwork = NeuralNetwork(LAYERS)
if not neuralNetwork.loadParameters(NNparams):
    exit(1)


# Number of errors
nbData = len(datas)
nbError = 0

for inputs, result, _ in datas:
    NNresult = neuralNetwork.forward(inputs)

    if (NNresult[0] <= NNresult[1]) != result:
        nbError += 1


print(f"{nbError} errors on {nbData} datas ({(nbError / nbData) * 100:.2f}%)")
