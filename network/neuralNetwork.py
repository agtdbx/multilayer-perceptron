# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/25 13:13:13 by gugus            ###   ########.fr        #
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


    def forwardLearning(self, inputs: list[float],
                        activationFunction=sigmoid) -> tuple[list[float], list[tuple[float, float]]]:
        """
        Method to process inputs through the neural network when learning.
        @param  inputs  The input vector.
        @param  activationFunction  The activation function, by default sigmoid.
        @return The network result and hidden results for back propagation.
        """
        hiddenResults = []

        # Give inputs to first layer, and layer result to the next layer.
        for layer in self.layers:
            inputs, hiddenRes = layer.forwardLearning(inputs, activationFunction)
            hiddenResults.append((inputs, hiddenRes))

        return inputs, hiddenResults


    def computeGradients(self, derivateLoss: float, inputs: list[float],
                         hiddenResults: list[tuple[float, float]],
                         derivateFunction=sigmoidDerivate) -> tuple[list[list[float]], list[float]]:
        """
        Method to compute the gradient by back propagation.
        @param  derivateLoss        The derivate loss of the current result.
        @param  inputs              The inputs given to the neural network.
        @param  hiddenResults       The hidden result save from forwardLearning method.
        @param  derivateFunction    The derivation of the activation function for forward.
        @return The tuple of gradient like (weigths gradient, biases gradient).
        """
        gradientWeights = [None] * self.nbNeurons
        gradientBiases = [None] * self.nbNeurons

        # Compute gradient per layer
        l = self.nbLayers - 1
        n_id = self.nbNeurons
        while l >= 0:
            z_lst = hiddenResults[l][1]
            inputSize = self.layers[l].inputSize
            n_id -= self.layers[l].nbNeurons
            for i in range(self.layers[l].nbNeurons):
                z = z_lst[i]
                da = derivateFunction(z)

                # Weights gradients
                weightsGradient = []
                for w_id in range(inputSize):
                    if l > 0:
                        dz = hiddenResults[l - 1][0][w_id]
                    else:
                        inputs[w_id]
                    dLossWeigth = dz * da * derivateLoss
                    weightsGradient.append(dLossWeigth)
                gradientWeights[n_id + i] = weightsGradient

                # Bias gradient
                gradientBiases[n_id + i] = da * derivateLoss

            l -= 1

        return (gradientWeights, gradientBiases)


    def applyGradient(self, gradientWeights: list[list[float]],
                      gradientBiases: list[float]):
        """
        Method to apply the gradient to weigths and biases to each layer.
        @param  gradientWeights The gradient to apply to weigths.
        @param  gradientBiases  The gradient to apply to biases.
        """
        start = 0
        for l in range(self.nbLayers):
            end = start + self.layers[l].nbNeurons
            self.layers[l].applyGradient(gradientWeights[start:end], gradientBiases[start:end])
            start = end
