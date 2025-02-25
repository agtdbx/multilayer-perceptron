# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/02/25 14:36:24 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import                  TRAINING_DATA_FILE, PARAMS_FILE, LAYERS,\
                                    INPUT_SIZE, LEARNING_LOOP, LEARNING_RATE
from parsing.file import            readFile, writeInFile
from parsing.parse import           parseData
from parsing.string import          vectorToStr, matrixToStr
from network.loss import            computeLoss, computeOneLossDerivate
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


# Compute wanted output for data
tmp = [(1, 0), (0, 1)]
dataResults = []
for _, result, _ in datas:
    dataResults.append(tmp[result])


# Normalize data
maxes = [0] * INPUT_SIZE
for inputs, _, _ in datas:
    for i in range(INPUT_SIZE):
        maxes[i] = max(maxes[i], inputs[i])
for inputs, _, _ in datas:
    for i in range(INPUT_SIZE):
        inputs[i] /= maxes[i]


# Gradient descent
nbData = len(datas)
invNbData = 1.0 / nbData
for nbLoop in range(LEARNING_LOOP):
    # Compute result from data]
    gradientWeights = [[]] * neuralNetwork.nbNeurons
    gradientBiases = [0.0] * neuralNetwork.nbNeurons
    NNresults = []
    for i in range(nbData):
        inputs = datas[i][0]
        # Compute NN result
        NNresult, hiddenResult = neuralNetwork.forwardLearning(inputs)
        # Save result for loss computation
        NNresults.append(NNresult)

        # Compute loss
        derivateLoss = computeOneLossDerivate(dataResults[i], NNresult)

        dLossWeigths, dLossBiases = neuralNetwork.computeGradients(derivateLoss, inputs, hiddenResult)

        # Add derivate loss by weigths and bias to grandients
        for j in range(neuralNetwork.nbNeurons):
            if len(gradientWeights[j]) != 0:
                for k in range(len(dLossWeigths[j])):
                    gradientWeights[j][k] += dLossWeigths[j][k]
            else:
                for k in range(len(dLossWeigths[j])):
                    gradientWeights[j].append(dLossWeigths[j][k])
            gradientBiases[j] += dLossBiases[j]

    # Make grandient avg and multiply it by learning rate
    for j in range(neuralNetwork.nbNeurons):
        for k in range(len(dLossWeigths[j])):
            gradientWeights[j][k] = gradientWeights[j][k] * invNbData  * LEARNING_RATE
        gradientBiases[j] = gradientBiases[j] * invNbData  * LEARNING_RATE

    # Apply gradient
    neuralNetwork.applyGradient(gradientWeights, gradientBiases)

    # Compute loss
    loss = computeLoss(dataResults, NNresults)
    print(f"loop {nbLoop + 1} : loss {loss}")


# TODO: Unnormalize weights and biases
NNparams = neuralNetwork.getParameters()

# Save params
strParams = ""
for weightMatrix, biasVector in NNparams:
    strWeigths = matrixToStr(weightMatrix)
    strBias = vectorToStr(biasVector)
    strParams += f"{strWeigths}|{strBias}\n"

if not writeInFile(PARAMS_FILE, strParams):
    print(strParams)
