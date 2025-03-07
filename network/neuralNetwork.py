# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/03/07 18:53:52 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import  LEARNING_RATE, SEED

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

        # If the seed isn't None, use it for as random seed
        if SEED != None:
            np.random.seed(SEED)

        for l in range(1, len(dimensions)):
            W = np.random.randn(dimensions[l], dimensions[l - 1])
            b = np.random.randn(dimensions[l], 1)
            self.parameters[f"W{l}"] = W
            self.parameters[f"b{l}"] = b

        self.nbLayers = len(self.parameters) // 2


    def loadParameters(self,
                       params: list[tuple[list[list[float]],list[float]]]):
        """
        Method to load weights and biases into the network.
        @param  params  The list of tupple (weigth matrix, bias vector).
        """
        # Give parameters to each layer
        for l in range(1, self.nbLayers + 1):
            W = self.parameters[f"W{l}"]
            weights = params[l - 1][0]

            if W.shape[1] != len(weights):
                raise ValueError(f"Bad weights height for layer {l}")

            for y in range(W.shape[1]):
                if W.shape[0] != len(weights[y]):
                    raise ValueError(f"Bad weights width size for layer {l}")
                for x in range(W.shape[0]):
                    W[x, y] = weights[y][x]

            b = self.parameters[f"b{l}"]
            biases = params[l - 1][1]
            if b.shape[0] != len(biases):
                raise ValueError(f"Bad biases size for layer {l}")

            for x in range(b.shape[0]):
                b[x, 0] = biases[x]


    def getParameters(self) -> list[tuple[list[list[float]], list[float]]]:
        """
        Method to get the network's parameters.
        @return A list as [(weight matrix, bias vector), ..].
        """
        result = []

        for l in range(1, self.nbLayers + 1):
            W = self.parameters[f"W{l}"]
            weights = []
            for y in range(W.shape[1]):
                lst = []
                for x in range(W.shape[0]):
                    lst.append(W[x, y])
                weights.append(lst)

            b = self.parameters[f"b{l}"]
            biases = []
            for x in range(b.shape[0]):
                biases.append(b[x, 0])

            result.append((weights, biases))

        return result


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
        invM = 1 / y.shape[1]

        dZ = self.activations[f"A{self.nbLayers}"] - y
        gradients: dict[str, Matrix] = {}

        for l in range(self.nbLayers, 0, -1):
            A = self.activations[f"A{l - 1}"]
            gradients[f"dW{l}"] = invM * np.dot(dZ, A.T)
            gradients[f"db{l}"] = invM * np.sum(dZ, axis=1, keepdims=True)
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
        NNresults = self.forwardPropagation(X)
        lst_result = []
        for i in range(NNresults.shape[1]):
            lst_result.append(int(NNresults[0, i] <= NNresults[1, i]))
        return np.array(lst_result).T
