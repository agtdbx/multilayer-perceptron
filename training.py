# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/03 16:40:05 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, VALIDATION_DATA_FILE,\
                                    PARAMS_FILE,\
                                    HIDDEN_LAYERS, INPUT_SIZE,\
                                    LEARNING_LOOP
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            logLoss
from network.neuralNetwork import   NeuralNetwork

import numpy as np
from tqdm import tqdm

# Get datas
dataTrainContent = readFile(TRAINING_DATA_FILE)
if dataTrainContent == None:
    exit(1)

datasTrain = parseData(dataTrainContent)
if datasTrain == None:
    exit(1)
nbDataTrain = len(datasTrain)

dataTestContent = readFile(VALIDATION_DATA_FILE)
if dataTestContent == None:
    exit(1)

datasTest = parseData(dataTestContent)
if datasTest == None:
    exit(1)
nbDataTest = len(datasTest)


# Put data into good format for training
lst_inputsTrain = []
lst_resultTrain = []
for inputs, result, _ in datasTrain:
    lst_inputsTrain.append(inputs)
    lst_resultTrain.append([result])

X_train = np.array(lst_inputsTrain).T
Y_train = np.array(lst_resultTrain).T


# Put data into good format for testing
lst_inputsTest = []
lst_resultTest = []
for inputs, result, _ in datasTest:
    lst_inputsTest.append(inputs)
    lst_resultTest.append([result])

X_test = np.array(lst_inputsTest).T
Y_test = np.array(lst_resultTest).T


# Create neural network
dimensions = list(HIDDEN_LAYERS)
dimensions.insert(0, X_train.shape[0])
dimensions.append(Y_train.shape[0])

neuralNetwork = NeuralNetwork(dimensions)


# Normalize data
maxes = []
for x in range(INPUT_SIZE):
    inputMax = 0
    for y in range(nbDataTrain):
        inputMax = max(X_train[x, y], inputMax)
    for y in range(nbDataTrain):
        X_train[x, y] /= inputMax
    for y in range(nbDataTest):
        X_test[x, y] /= inputMax
    maxes.append(inputMax)


# Gradient descent
trainLosses = []
testLosses = []
trainAccuracy = []
testAccuracy = []
for nbLoop in tqdm(range(LEARNING_LOOP)):
    results = neuralNetwork.forwardPropagation(X_train)
    gradients = neuralNetwork.backwardProgatation(Y_train)
    neuralNetwork.applyGradients(gradients)

    if nbLoop % 10 == 0:
        # Train
        trainLosses.append(logLoss(Y_train, results))
        errors = 0
        for y in range(nbDataTrain):
            if Y_train[0, y] != (results[0, y] >= 0.5):
                errors += 1
        trainAccuracy.append((1 - (errors / nbDataTrain)) * 100)

        # Test
        results = neuralNetwork.forwardPropagation(X_test)
        testLosses.append(logLoss(Y_test, results))
        errors = 0
        for y in range(nbDataTest):
            if Y_test[0, y] != (results[0, y] >= 0.5):
                errors += 1
        testAccuracy.append((1 - (errors / nbDataTest)) * 100)


# Save params
NNparams = neuralNetwork.getParameters()

strParams = vectorToStr(maxes) + '\n'
for weightMatrix, biasVector in NNparams:
    strWeigths = matrixToStr(weightMatrix)
    strBias = vectorToStr(biasVector)
    strParams += f"{strWeigths}|{strBias}\n"

if not writeInFile(PARAMS_FILE, strParams):
    print(strParams)


# Model accuracy
results = neuralNetwork.forwardPropagation(X_train)
errors = 0
for y in range(nbDataTrain):
    if Y_train[0, y] != (results[0, y] >= 0.5):
        errors += 1
print(f"Model train accuracy : {(1 - (errors / nbDataTrain)) * 100:.2f}%")

results = neuralNetwork.forwardPropagation(X_test)
errors = 0
for y in range(nbDataTest):
    if Y_test[0, y] != (results[0, y] >= 0.5):
        errors += 1
print(f"Model test accuracy  : {(1 - (errors / nbDataTest)) * 100:.2f}%")


# Plots of training
import matplotlib.pyplot as plt
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(18, 4))

ax[0].plot(trainLosses, label="train loss")
ax[0].legend()

ax[1].plot(testLosses, label="test loss")
ax[1].legend()

ax[2].plot(trainAccuracy, label="train accuracy")
ax[2].plot(testAccuracy, label="test accuracy")
ax[2].legend()

plt.show()
