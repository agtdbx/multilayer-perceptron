# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    training.py                                        :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/21 15:08:58 by gugus             #+#    #+#              #
#    Updated: 2025/02/26 18:36:58 by gugus            ###   ########.fr        #
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
minimumLoss = computeLoss(dataResults, dataResults)
print(f"minimum loss {minimumLoss}")


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
    lst_gradients = []
    NNresults = []
    nbError = 0
    # For each data in datas
    for i in range(nbData):
        inputs = datas[i][0]
        # Compute NN result
        NNresult = neuralNetwork.forward(inputs)
        # Save result for loss computation
        NNresults.append(NNresult)

        # Update error number
        if datas[i][1] == 0:
            if NNresult[0] <= NNresult[1]:
                nbError += 1
        else:
            if NNresult[0] >= NNresult[1]:
                nbError += 1

        # Compute derivate loss
        dLoss = computeOneLossDerivate(dataResults[i], NNresult)
        # Compute gradients
        gradients = neuralNetwork.backprogatation(dLoss)
        lst_gradients.append(gradients)

    # Make grandient avg and multiply it by learning rate
    gradients = []
    for l in range(neuralNetwork.nbLayers):
        weigthsGradients = []
        biasesGradients = []
        for n in range(neuralNetwork.layers[l].nbNeurons):
            # Weigths gradients
            lst_weightGradient = []
            for i in range(neuralNetwork.layers[l].inputSize):
                weightGradient = 0
                for index in range(nbData):
                    weightGradient += lst_gradients[index][l][0][n][i]
                # Make the average and multiply by learning rate
                weightGradient = weightGradient * invNbData * LEARNING_RATE
                lst_weightGradient.append(weightGradient)

            weigthsGradients.append(lst_weightGradient)

            # Bias gradients
            gradientBias = 0
            for index in range(nbData):
                gradientBias += lst_gradients[index][l][1][n]
            # Make the average and multiply by learning rate
            gradientBias = gradientBias * invNbData * LEARNING_RATE
            biasesGradients.append(gradientBias)

        gradients.append((weigthsGradients, biasesGradients))

    # Apply gradient
    neuralNetwork.applyGradient(gradients)

    # Compute loss
    loss = computeLoss(dataResults, NNresults)
    print(f"loop {nbLoop + 1:3} : loss {loss:.3f}, errors {nbError:3}/{nbData:3}")


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
