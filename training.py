# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/05 13:12:53 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, VALIDATION_DATA_FILE,\
                                    PARAMS_FILE, HIDDEN_LAYERS, INPUT_SIZE,\
                                    LEARNING_LOOP, EARLY_SAME_LIMIT,\
                                    EARLY_SAME_PRECISION
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            logLoss
from network.neuralNetwork import   NeuralNetwork

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Get datas
dataTrainContent = readFile(TRAINING_DATA_FILE)
if dataTrainContent == None:
    exit(1)

datasTrain = parseData(dataTrainContent)
if datasTrain == None:
    exit(1)
nbDataTrain = len(datasTrain)

dataValidationContent = readFile(VALIDATION_DATA_FILE)
if dataValidationContent == None:
    exit(1)

datasValidation = parseData(dataValidationContent)
if datasValidation == None:
    exit(1)
nbDataValidation = len(datasValidation)


# Put data into good format for training
lst_inputsTrain = []
lst_resultTrain = []
for inputs, result, _ in datasTrain:
    lst_inputsTrain.append(inputs)
    lst_resultTrain.append([result])

X_train = np.array(lst_inputsTrain).T
Y_train = np.array(lst_resultTrain).T
print(f"Train data shape {X_train.shape}")


# Put data into good format for validation
lst_inputsValidation = []
lst_resultValidation = []
for inputs, result, _ in datasValidation:
    lst_inputsValidation.append(inputs)
    lst_resultValidation.append([result])

X_validation = np.array(lst_inputsValidation).T
Y_validation = np.array(lst_resultValidation).T
print(f"Validation data shape {X_validation.shape}")


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
    for y in range(nbDataValidation):
        X_validation[x, y] /= inputMax
    maxes.append(inputMax)


# Gradient descent
x_plots = []
trainLosses = []
validationLosses = []
trainAccuracy = []
validationAccuracy = []
nbEarlySame = 0

trainLoop = tqdm(range(LEARNING_LOOP))
for itLoop in trainLoop:
    results = neuralNetwork.forwardPropagation(X_train)
    gradients = neuralNetwork.backwardProgatation(Y_train)
    neuralNetwork.applyGradients(gradients)

    if itLoop % 10 == 0:
        x_plots.append(itLoop)

        # Train
        loss = logLoss(Y_train, results)
        trainLosses.append(loss)
        errors = 0
        for y in range(nbDataTrain):
            if Y_train[0, y] != (results[0, y] >= 0.5):
                errors += 1
        accuracy = (1 - (errors / nbDataTrain)) * 100
        trainAccuracy.append(accuracy)

        # Update display
        loss /= 1e+17
        description = f"loss {loss:.2f}e+17, accuracy {accuracy:.2f}%"
        trainLoop.set_description(description)

        # Validation
        results = neuralNetwork.forwardPropagation(X_validation)
        loss = logLoss(Y_validation, results)
        validationLosses.append(loss)
        errors = 0
        for y in range(nbDataValidation):
            if Y_validation[0, y] != (results[0, y] >= 0.5):
                errors += 1
        validationAccuracy.append((1 - (errors / nbDataValidation)) * 100)
        if itLoop == 0: continue
        if (validationLosses[-2] - loss) < EARLY_SAME_PRECISION:
            nbEarlySame += 1
            if nbEarlySame == EARLY_SAME_LIMIT:
                trainLoop.total = itLoop
                break


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

results = neuralNetwork.forwardPropagation(X_validation)
errors = 0
for y in range(nbDataValidation):
    if Y_validation[0, y] != (results[0, y] >= 0.5):
        errors += 1
accuracy = (1 - (errors / nbDataValidation)) * 100
print(f"Model validation accuracy : {accuracy:.2f}%")


# Plots of training
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 5))
fig.canvas.manager.set_window_title("Training plots")

ax[0].plot(x_plots, trainLosses, label="train loss")
ax[0].set_title("Train loss over training")
ax[0].set_xlabel("Training loop")
ax[0].set_ylabel("Train loss")
ax[0].grid(visible=True)
ax[0].legend()

ax[1].plot(x_plots, validationLosses, label="validation loss")
ax[1].set_title("Validation loss over training")
ax[1].set_xlabel("Training loop")
ax[1].set_ylabel("Validation loss")
ax[1].grid(visible=True)
ax[1].legend()

ax[2].plot(x_plots, trainAccuracy, label="train accuracy")
ax[2].plot(x_plots, validationAccuracy, label="validation accuracy")
ax[2].set_title("Accuracy over training")
ax[2].set_xlabel("Training loop")
ax[2].set_ylabel("Accuracy")
ax[2].grid(visible=True)
ax[2].legend()

plt.show()
