# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/24 18:18:11 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from network.layer import   Layer
from network.sigmoid import sigmoid

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

        # Layers creation
        self.layers: list[Layer] = []
        inputSize = layersSize[0]
        for layerSize in layersSize:
            self.layers.append(Layer(inputSize, layerSize))
            inputSize = layerSize

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

