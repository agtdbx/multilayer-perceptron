# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    compare.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/07 19:31:28 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, VALIDATION_DATA_FILE,\
                                    INPUT_SIZE, LEARNING_LOOP, COMPARE_LAYERS
from parsing.file import            readFile
from parsing.parse import           parseData
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
lst_resultTrainFlat = []
for inputs, result, _ in datasTrain:
    lst_inputsTrain.append(inputs)
    if result == 0:
        lst_resultTrain.append([1, 0])
    else:
        lst_resultTrain.append([0, 1])
    lst_resultTrainFlat.append(result)

X_train = np.array(lst_inputsTrain).T
Y_train = np.array(lst_resultTrain).T
Y_trainFlat = np.array(lst_resultTrainFlat)


# Put data into good format for validation
lst_inputsValidation = []
lst_resultValidation = []
lst_resultValidationFlat = []
for inputs, result, _ in datasValidation:
    lst_inputsValidation.append(inputs)
    if result == 0:
        lst_resultValidation.append([1, 0])
    else:
        lst_resultValidation.append([0, 1])
    lst_resultValidationFlat.append(result)

X_validation = np.array(lst_inputsValidation).T
Y_validation = np.array(lst_resultValidation).T
Y_validationFlat = np.array(lst_resultValidationFlat)


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


# Plot creation
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(21, 5))
fig.canvas.manager.set_window_title("Training plots")

ax[0].set_title("Train loss over training")
ax[0].set_xlabel("Training loop")
ax[0].set_ylabel("Train loss")
ax[0].grid(visible=True)

ax[1].set_title("Validation loss over training")
ax[1].set_xlabel("Training loop")
ax[1].set_ylabel("Validation loss")
ax[1].grid(visible=True)

ax[2].set_title("Accuracy over training")
ax[2].set_xlabel("Training loop")
ax[2].set_ylabel("Accuracy")
ax[2].grid(visible=True)

x_plots = [i * 10 for i in range(LEARNING_LOOP // 10)]


# Compare loop
nbNetwork = len(COMPARE_LAYERS)
if nbNetwork == 0:
    print("Error, must have at least network to compare")
    exit(1)

for networkID in range(nbNetwork):
    hiddenLayers = COMPARE_LAYERS[networkID]
    print(f"Network {networkID + 1}/{nbNetwork} with hidden layer :",
          hiddenLayers)

    # Create neural network
    dimensions = list(hiddenLayers)
    dimensions.insert(0, X_train.shape[0])
    dimensions.append(Y_train.shape[0])
    neuralNetwork = NeuralNetwork(dimensions)

    # Gradients descent
    trainLosses = []
    validationLosses = []
    trainAccuracy = []
    validationAccuracy = []

    trainLoop = tqdm(range(LEARNING_LOOP))
    for itLoop in trainLoop:
        results = neuralNetwork.forwardPropagation(X_train)
        gradients = neuralNetwork.backwardProgatation(Y_train)
        neuralNetwork.applyGradients(gradients)

        if itLoop % 10 == 0:
            # Train
            trainLosses.append(logLoss(Y_train, results))
            errors = 0
            for y in range(nbDataTrain):
                if Y_trainFlat[y] != (results[0, y] <= results[1, y]):
                    errors += 1
            accuracy = (1 - (errors / nbDataTrain)) * 100
            trainAccuracy.append(accuracy)

            # Validation
            results = neuralNetwork.forwardPropagation(X_validation)
            validationLosses.append(logLoss(Y_validation, results))
            errors = 0
            for y in range(nbDataValidation):
                if Y_validationFlat[y] != (results[0, y] <= results[1, y]):
                    errors += 1
            validationAccuracy.append((1 - (errors / nbDataValidation)) * 100)

    # Fill Plot of trainings
    name = f"NN {networkID + 1}"
    ax[0].plot(x_plots, trainLosses, label=f"{name} train loss")
    ax[1].plot(x_plots, validationLosses, label=f"{name} validation loss")
    ax[2].plot(x_plots, trainAccuracy, label=f"{name} train accuracy")


# End plots setup and display them
ax[0].legend()
ax[1].legend()
ax[2].legend()

plt.show()
