# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/03/07 19:25:02 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, VALIDATION_DATA_FILE,\
                                    PARAMS_FILE, HIDDEN_LAYERS, INPUT_SIZE,\
                                    LEARNING_LOOP, EARLY_SAME_LIMIT,\
                                    EARLY_SAME_PRECISION
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr,\
                                    strToLayers, layersToStr
from network.loss import            logLoss
from network.neuralNetwork import   NeuralNetwork

import sys
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


# Check if we have a parameter
argc = len(sys.argv)
if argc > 1:
    hiddenLayers = strToLayers(sys.argv[1])
    if hiddenLayers == None:
        exit(1)
else:
    hiddenLayers = HIDDEN_LAYERS


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
print(f"Train data shape {X_train.shape}")


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
print(f"Validation data shape {X_validation.shape}")


# Create neural network
dimensions = list(hiddenLayers)
dimensions.insert(0, X_train.shape[0])
dimensions.append(Y_train.shape[0])

print(f"Network dimensions : {dimensions}")
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
trainConfusion = []
validationConfusion = []
nbEarlySame = 0
lastLossEarly = 0

trainLoop = tqdm(range(LEARNING_LOOP))
for itLoop in trainLoop:
    results = neuralNetwork.forwardPropagation(X_train)
    # loss = logLoss(Y_train, results)
    # print(loss)
    # exit()
    gradients = neuralNetwork.backwardProgatation(Y_train)
    neuralNetwork.applyGradients(gradients)

    if itLoop % 10 == 0:
        x_plots.append(itLoop)

        # Train
        loss = logLoss(Y_train, results)
        trainLosses.append(loss)
        errors = 0
        resultMatrix = np.array([[0, 0], [0, 0]])
        for i in range(nbDataTrain):
            y = Y_trainFlat[i]
            a = int(results[0, i] <= results[1, i])
            resultMatrix[y, a] += 1
            if y != a:
                errors += 1
        accuracy = (1 - (errors / nbDataTrain)) * 100
        trainAccuracy.append(accuracy)
        trainConfusion.append(resultMatrix)

        # Update display
        description = f"loss {loss:.2f}, accuracy {accuracy:.2f}%"
        trainLoop.set_description(description)

        # Validation
        results = neuralNetwork.forwardPropagation(X_validation)
        loss = logLoss(Y_validation, results)
        validationLosses.append(loss)
        errors = 0
        resultMatrix = np.array([[0, 0], [0, 0]])
        for i in range(nbDataValidation):
            y = Y_validationFlat[i]
            a = int(results[0, i] <= results[1, i])
            resultMatrix[y, a] += 1
            if y != a:
                errors += 1
        validationAccuracy.append((1 - (errors / nbDataValidation)) * 100)
        validationConfusion.append(resultMatrix)

        # Early stop
        if itLoop == 0:
            continue

        if (lastLossEarly - loss) < EARLY_SAME_PRECISION:
            nbEarlySame += 1
            if nbEarlySame == EARLY_SAME_LIMIT:
                trainLoop.total = itLoop
                break
        lastLossEarly = loss


# Save params
NNparams = neuralNetwork.getParameters()

strParams = layersToStr(hiddenLayers) + '\n' + vectorToStr(maxes) + '\n'
for weightMatrix, biasVector in NNparams:
    strWeigths = matrixToStr(weightMatrix)
    strBias = vectorToStr(biasVector)
    strParams += f"{strWeigths}|{strBias}\n"

if not writeInFile(PARAMS_FILE, strParams):
    print(strParams)


# Model accuracy
results = neuralNetwork.predict(X_train)
errors = 0
for y in range(nbDataTrain):
    if Y_trainFlat[y] != results[y]:
        errors += 1
print(f"Model train accuracy : {(1 - (errors / nbDataTrain)) * 100:.2f}%")

results = neuralNetwork.predict(X_validation)
errors = 0
for y in range(nbDataValidation):
    if Y_validationFlat[y] != results[y]:
        errors += 1
accuracy = (1 - (errors / nbDataValidation)) * 100
print(f"Model validation accuracy : {accuracy:.2f}%")


# Transform list of result matrix into lists for plotting
train_expected_0 = []
train_error_0 = []
train_error_1 = []
train_expected_1 = []
scale = 1 / nbDataTrain * 100
for confusionMatrix in trainConfusion:
    train_expected_0.append(confusionMatrix[0, 0] * scale)
    train_error_0.append(confusionMatrix[0, 1] * scale)
    train_error_1.append(confusionMatrix[1, 0] * scale)
    train_expected_1.append(confusionMatrix[1, 1] * scale)

validation_expected_0 = []
validation_error_0 = []
validation_error_1 = []
validation_expected_1 = []
scale = 1 / nbDataValidation * 100
for confusionMatrix in validationConfusion:
    validation_expected_0.append(confusionMatrix[0, 0] * scale)
    validation_error_0.append(confusionMatrix[0, 1] * scale)
    validation_error_1.append(confusionMatrix[1, 0] * scale)
    validation_expected_1.append(confusionMatrix[1, 1] * scale)


# Plots of training
fig, ax = plt.subplots(nrows=2, ncols=3, figsize=(18, 10))
fig.canvas.manager.set_window_title("Training plots")

ax[0][0].plot(x_plots, trainLosses, label="train loss")
ax[0][0].set_title("Train loss over training")
ax[0][0].set_xlabel("Training loop")
ax[0][0].set_ylabel("Train loss")
ax[0][0].grid(visible=True)
ax[0][0].legend()

ax[0][1].plot(x_plots, validationLosses, label="validation loss")
ax[0][1].set_title("Validation loss over training")
ax[0][1].set_xlabel("Training loop")
ax[0][1].set_ylabel("Validation loss")
ax[0][1].grid(visible=True)
ax[0][1].legend()

ax[0][2].plot(x_plots, trainAccuracy, label="train accuracy")
ax[0][2].plot(x_plots, validationAccuracy, label="validation accuracy")
ax[0][2].set_title("Accuracy over training")
ax[0][2].set_xlabel("Training loop")
ax[0][2].set_ylabel("Accuracy")
ax[0][2].grid(visible=True)
ax[0][2].legend()

ax[1][0].plot(x_plots, train_expected_0, label="expected train 0")
ax[1][0].plot(x_plots, train_error_0, label="error train 0")
ax[1][0].plot(x_plots, validation_expected_0, label="expected validation 0")
ax[1][0].plot(x_plots, validation_error_0, label="error validation 0")
ax[1][0].set_title("Confusion of 0 over training")
ax[1][0].set_xlabel("Training loop")
ax[1][0].set_ylabel("precentage of result")
ax[1][0].grid(visible=True)
ax[1][0].legend()

ax[1][1].plot(x_plots, train_expected_1, label="expected train 1")
ax[1][1].plot(x_plots, train_error_1, label="error train 1")
ax[1][1].plot(x_plots, validation_expected_1, label="expected validation 1")
ax[1][1].plot(x_plots, validation_error_1, label="error validation 1")
ax[1][1].set_title("Confusion of 1 over training")
ax[1][1].set_xlabel("Training loop")
ax[1][1].set_ylabel("precentage of result")
ax[1][1].grid(visible=True)
ax[1][1].legend()

confusionMatrix = validationConfusion[-1] / nbDataValidation
ax[1][2].matshow(confusionMatrix, cmap='seismic')
for (x, y), value in np.ndenumerate(confusionMatrix):
    ax[1][2].text(y, x, f'{value:0.2f}', ha="center", va="center",
                  color="white", size=15)
ax[1][2].set_title("Confusion matrix on validation data")
ax[1][2].set_xlabel("Neural network ouput")
ax[1][2].set_ylabel("Wanted output")
fig.tight_layout(pad=5)
plt.show()
