# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/02/24 17:24:22 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, PARAMS_FILE, LAYERS,\
                                    LEARNING_LOOP, LEARNING_RATIO
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            computeLoss
from network.neuralNetwork import   NeuralNetwork

# Get data
dataPath = ""
dataContent = readFile(TRAINING_DATA_FILE)
if dataContent == None:
    exit(1)

datas = parseData(dataContent)
if datas == None:
    exit(1)


# Create neural network
neuralNetwork = NeuralNetwork(LAYERS)


# Gradient descent
nbData = len(datas)
for i in range(LEARNING_LOOP):
    loss = computeLoss(neuralNetwork, datas, nbData)
    print(f"loop {i} : loss {loss}")


# Save params
NNparams = neuralNetwork.getParameters()

strParams = ""
for weightMatrix, biasVector in NNparams:
    strWeigths = matrixToStr(weightMatrix)
    strBias = vectorToStr(biasVector)
    strParams += f"{strWeigths}|{strBias}\n"

if not writeInFile(PARAMS_FILE, strParams):
    print(strParams)
