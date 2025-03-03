# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/03 15:44:02 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, PARAMS_FILE,\
                                    HIDDEN_LAYERS, INPUT_SIZE,\
                                    LEARNING_LOOP
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            logLoss
from network.neuralNetwork import   NeuralNetwork

import numpy as np
from tqdm import tqdm

# Get data
dataPath = ""
dataContent = readFile(TRAINING_DATA_FILE)
if dataContent == None:
    exit(1)

datas = parseData(dataContent)
if datas == None:
    exit(1)
nbData = len(datas)


# Put data into good format for training
lst_inputs = []
lst_result = []
for inputs, result, _ in datas:
    lst_inputs.append(inputs)
    lst_result.append([result])

X_train = np.array(lst_inputs).T
Y_train = np.array(lst_result).T

# minimumLoss = logLoss(Y_train, Y_train)
# print(f"minimum loss {minimumLoss}")


# Create neural network
dimensions = list(HIDDEN_LAYERS)
dimensions.insert(0, X_train.shape[0])
dimensions.append(Y_train.shape[0])

neuralNetwork = NeuralNetwork(dimensions)


maxes = []
for x in range(INPUT_SIZE):
    inputMax = 0
    for y in range(nbData):
        inputMax = max(X_train[x, y], inputMax)
    for y in range(nbData):
        X_train[x, y] /= inputMax
    maxes.append(inputMax)


# Gradient descent
trainLosses = []
trainAccuracy = []
for nbLoop in tqdm(range(LEARNING_LOOP)):
    results = neuralNetwork.forwardPropagation(X_train)
    gradients = neuralNetwork.backwardProgatation(Y_train)
    neuralNetwork.applyGradients(gradients)

    if nbLoop % 10 == 0:
        trainLosses.append(logLoss(Y_train, results))
        errors = 0
        for y in range(nbData):
            if Y_train[0, y] != (results[0, y] >= 0.5):
                errors += 1
        trainAccuracy.append((1 - (errors / nbData)) * 100)


# Plots of training
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

ax[0].plot(trainLosses, label="train loss")
ax[0].legend()

ax[1].plot(trainAccuracy, label="train accuracy")
ax[1].legend()

plt.show()


# # TODO: Unnormalize weights and biases
# NNparams = neuralNetwork.getParameters()

# # Save params
# strParams = ""
# for weightMatrix, biasVector in NNparams:
#     strWeigths = matrixToStr(weightMatrix)
#     strBias = vectorToStr(biasVector)
#     strParams += f"{strWeigths}|{strBias}\n"

# if not writeInFile(PARAMS_FILE, strParams):
#     print(strParams)
