# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/21 16:47:44 by gugus            ###   ########.fr        #
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


    def loadParameters(self, weights: list[list[list[float]]], biases: list[list[float]]):
        """
        Method to load weights and biases into the network.
        @param  weights The list of weigth matrix.
        @param  biases  The list of bias vector.
        """
        # Give parameters to each layer
        for i in range(self.nbLayers):
            self.layers[i].loadParameters(weights[i], biases[i])


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
