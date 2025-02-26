# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/26 14:37:07 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from network.layer import   Layer
from network.sigmoid import sigmoid, sigmoidDerivate

class NeuralNetwork:
    """
    The class for the neural network.
    """

    def __init__(self, layersSize: list[int]) -> 'NeuralNetwork':
        """
        The constructor of NeuralNetwork class.
        @param  layersSize  The number of neurons per layer.
        """
        self.nbLayers = len(layersSize)
        if self.nbLayers < 2:
            raise ValueError("You need to have at least 2 layers for a neural network")

        self.nbNeurons = 0
        # Layers creation
        self.layers: list[Layer] = []
        inputSize = layersSize[0]
        for layerSize in layersSize:
            self.layers.append(Layer(inputSize, layerSize))
            inputSize = layerSize
            self.nbNeurons += layerSize

        self.inputSize = layersSize[0]
        self.outputSize = layersSize[-1]


    def loadParameters(self, params: list[tuple[list[list[float]], list[float]]]) -> bool:
        """
        Method to load weights and biases into the network.
        @param  params  The list of tupple (weigth matrix, bias vector).
        @return False in case or error, True else.
        """
        # Give parameters to each layer
        for i in range(self.nbLayers):
            if not self.layers[i].loadParameters(params[i][0], params[i][1]):
                return False

        return True


    def getParameters(self) -> list[tuple[list[list[float]], list[float]]]:
        """
        Method to get the network's parameters.
        @return A list as [(weight matrix, bias vector), ..].
        """
        result = []

        for layer in self.layers:
            result.append(layer.getParameters())

        return result


    def forward(self, inputs: list[float], activationFunction=sigmoid) -> list[float]:
        """
        Method to process inputs through the neural network.
        @param  inputs  The input vector.
        @param  activationFunction  The activation function, by default sigmoid.
        @return The network result.
        """
        # Give inputs to first layer, and layer result to the next layer.
        for layer in self.layers:
            inputs = layer.forward(inputs, activationFunction)

        return inputs


    def backprogatation(self,
                        dLoss: float,
                        derivateFunction=sigmoidDerivate
                        )-> list[tuple[list[list[float]], list[float]]]:
        """
        Method to compute the gradient by back propagation.
        @param  dLoss               The derivation of the loss by the neural network result.
        @param  derivateFunction    The derivation of the activation function for forward.
        @return The list of tuple of gradient like [(weigths gradient, biases gradient), ...].
        """
        gradients = [None] * self.nbLayers
        l = self.nbLayers - 1
        dLosses = [dLoss] * self.layers[l].nbNeurons

        while l >= 0:
            layer = self.layers[l]
            newDLosses = [0.0] * layer.inputSize
            gradientWeights = []
            gradientBiases = []

            # For each neurons
            for n in range(layer.nbNeurons):
                # Derivate of result by weighted sum
                dA = derivateFunction(layer.hiddenResults[n])

                lst_dW = []
                for i in range(layer.inputSize):
                    # Derivate of weighted sum by weight
                    dZ = layer.inputs[i]

                    # Derivate of loss by weight
                    dW = dZ * dA * dLosses[n]
                    lst_dW.append(dW)

                    # Derivate of loss by input
                    dI = layer.weights[n][i] * dA * dLosses[n]
                    newDLosses[i] += dI
                gradientWeights.append(lst_dW)

                # Derivate of loss by bias
                dB = dA * dLosses[n]
                gradientBiases.append(dB)

            # Update dLosses
            dLosses = newDLosses

            gradients[l] = (gradientWeights, gradientBiases)

            l -= 1

        return gradients


    def applyGradient(self, gradients: list[tuple[list[list[float]], list[float]]]):
        """
        Method to apply the gradient to weigths and biases to each layer.
        @param  gradients   The gradients to apply to weigths and biases.
        """
        for l in range(self.nbLayers):
            self.layers[l].applyGradient(gradients[l][0], gradients[l][1])
