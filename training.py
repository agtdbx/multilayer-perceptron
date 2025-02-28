# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/02/28 16:35:29 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, PARAMS_FILE,\
                                    HIDDEN_LAYERS, INPUT_SIZE,\
                                    LEARNING_LOOP,\
                                    LOSS_MAX_REPEAT, LOSS_PRECISION
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            logLoss
from network.matrix import          Matrix
from network.neuralNetwork import   NeuralNetwork

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


# Create neural network
neuralNetwork = NeuralNetwork(HIDDEN_LAYERS)


# Put data into good format for training
lst_inputs = []
lst_result = []
for inputs, result, _ in datas:
    lst_inputs.append(inputs)
    lst_result.append([result])

X_train = Matrix((INPUT_SIZE, nbData), lst_inputs).transpose()
Y_train = Matrix((1, nbData), lst_result).transpose()

minimumLoss = logLoss(Y_train, Y_train)
print(f"minimum loss {minimumLoss}")


# Normalize data
maxes = []
for y in range(INPUT_SIZE):
    inputMax = max(X_train.data[y])
    for x in range(nbData):
        X_train[x, y] /= inputMax
    maxes.append(inputMax)


# Gradient descent
for nbLoop in tqdm(range(LEARNING_LOOP)):
    activations = neuralNetwork.forwardPropagation(X_train)
    print(activations)


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
