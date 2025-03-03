# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/03/03 15:39:05 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import  LEARNING_RATE

import numpy as np
from numpy import ndarray as Matrix

class NeuralNetwork:
    """
    The class for the neural network.
    """

    def __init__(self, dimensions: list[int]) -> 'NeuralNetwork':
        """
        The constructor of NeuralNetwork class.
        @param  dimensions  The number of neurons per layers.
        """
        # Init weights and biases per layer
        self.parameters: dict[str, Matrix] = {}

        for l in range(1, len(dimensions)):
            W = np.random.randn(dimensions[l], dimensions[l - 1])
            b = np.random.randn(dimensions[l], 1)
            self.parameters[f"W{l}"] = W
            self.parameters[f"b{l}"] = b

        self.nbLayers = len(self.parameters) // 2


    # def loadParameters(self,
    #                    params: list[tuple[list[list[float]],list[float]]]):
    #     """
    #     Method to load weights and biases into the network.
    #     @param  params  The list of tupple (weigth matrix, bias vector).
    #     """
    #     # Give parameters to each layer
    #     for l in range(self.nbLayers):
    #         self.weigths[l].mapData(params[l][0])
    #         self.biases[l].mapData(params[l][1])


    # def getParameters(self) -> list[tuple[list[list[float]], list[float]]]:
    #     """
    #     Method to get the network's parameters.
    #     @return A list as [(weight matrix, bias vector), ..].
    #     """
    #     result = []

    #     for l in range(self.nbLayers):
    #         result.append((self.weigths[l].data.copy(),
    #                        self.biases[l].data.copy()))

    #     return result

    def forwardPropagation(self,
                           X: Matrix) -> Matrix:
        """
        Method to process inputs through the neural network.
        @param  X       The input matrix.
        @return The network result.
        """
        self.activations: dict[str, Matrix] = {"A0" : X}

        for l in range(1, self.nbLayers + 1):
            W = self.parameters[f"W{l}"]
            A = self.activations[f"A{l - 1}"]
            b = self.parameters[f"b{l}"]
            Z = W.dot(A) + b
            self.activations[f"A{l}"] = 1 / (1 + np.exp(-Z))

        return self.activations[f"A{self.nbLayers}"]


    def backwardProgatation(self, y: Matrix)-> dict[str, Matrix]:
        """
        Method to compute the gradient by backward propagation.
        Return format : [(weigths gradient, biases gradient), ...].
        @param  y   The result wanted.
        @return The list of tuple of gradient.
        """
        m = y.shape[1]

        dZ = self.activations[f"A{self.nbLayers}"] - y
        gradients: dict[str, Matrix] = {}

        for l in range(self.nbLayers, 0, -1):
            A = self.activations[f"A{l - 1}"]
            gradients[f"dW{l}"] = 1 / m * np.dot(dZ, A.T)
            gradients[f"db{l}"] = 1 / m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                WT = self.parameters[f"W{l}"].T
                dZ = np.dot(WT, dZ) * A * (1 - A)

        return gradients


    def applyGradients(self, gradients: dict[str, Matrix]):
        """
        Method to apply the gradient to weigths and biases to each layer.
        @param  gradients   The gradients to apply to weigths and biases.
        """
        for l in range(1, self.nbLayers + 1):
            W = self.parameters[f"W{l}"]
            b = self.parameters[f"b{l}"]
            dW = gradients[f"dW{l}"]
            db = gradients[f"db{l}"]
            self.parameters[f"W{l}"] = W - LEARNING_RATE * dW
            self.parameters[f"b{l}"] = b - LEARNING_RATE * db


    def predict(self, X: Matrix) -> Matrix:
        """
        Method to predict a result by given input.
        @param  X   Input to predict.
        @return A matrix of predicted result.
        """
        A = self.forwardPropagation(X)

        return A >= 0.5
