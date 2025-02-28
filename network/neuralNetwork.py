# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/28 21:32:41 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import          LEARNING_RATE

from network.matrix import  Matrix
from network.sigmoid import sigmoid

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
            W = Matrix((dimensions[l], dimensions[l - 1]))
            b = Matrix((dimensions[l], 1))

            W.fillRandom()
            b.fillRandom()

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
            Z = A * W + b
            self.activations[f"A{l}"] = sigmoid(-Z)

        return self.activations[f"A{self.nbLayers}"]


    def backwardProgatation(self, y: Matrix)-> dict[str, Matrix]:
        """
        Method to compute the gradient by backward propagation.
        Return format : [(weigths gradient, biases gradient), ...].
        @param  y   The result wanted.
        @return The list of tuple of gradient.
        """
        m = y.height

        dZ = self.activations[f"A{self.nbLayers}"] - y
        gradients: dict[str, Matrix] = {}

        for l in range(self.nbLayers, 0, -1):
            A = self.activations[f"A{l - 1}"]
            dW = 1 / m * (A.transpose() * dZ)
            gradients[f"dW{l}"] = dW

            sumDZ = 0
            for i in range(dZ.height):
                sumDZ += dZ[0, i]
            db = 1 / m * sumDZ
            gradients[f"db{l}"] = db

            if l > 1:
                WT = self.parameters[f"W{l}"].transpose()
                dZ = dZ * WT * A * (1 - A)

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
            self.parameters[f"W{l}"] = W - dW * LEARNING_RATE
            self.parameters[f"b{l}"] = b - db * LEARNING_RATE


    def predict(self, X: Matrix) -> Matrix:
        """
        Method to predict a result by given input.
        @param  X   Input to predict.
        @return A matrix of predicted result.
        """
        a = self.forwardPropagation(X)
        for x in range(a.width):
            if a[x, 0] >= 0.5:
                a[x, 0] = 1
            else:
                a[x, 0] = 0

        return a
