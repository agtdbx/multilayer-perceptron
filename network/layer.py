# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layer.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:17 by aderouba          #+#    #+#              #
#    Updated: 2025/02/28 13:22:45 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from parsing.print import printError

import random


class Layer:
    """
    Class for a layer of perceptron.
    """

    def __init__(self, inputSize: int, nbNeurons: int) -> 'Layer':
        """
        Constructor of Layer class.
        @param  inputSize   The size of the input give to the layer.
        @param  nbNeurons   The number of neurons in the layer.
        @return The Layer's object.
        """


    def loadParameters(self,
                       weights: list[list[float]],
                       biases: list[float]) -> bool:
        """
        Method to load weights and biases into the layer.
        Print it's error.
        @param  weights The weight matrix.
        @param  biases  The vector of bias.
        @return False in case or error, True else.
        """
        if len(weights) != self.nbNeurons:
            printError("bad number of weigth vectors")
            return False

        for w in weights:
            if len(w) != self.inputSize:
                printError("bad size of weigth vector")
                return False

        if len(biases) != self.nbNeurons:
            printError("bad size of bias vector")
            return False

        self.weights = weights
        self.biases = biases
        return True


    def getParameters(self) -> tuple[list[list[float]], list[float]]:
        """
        Method to get the layer's parameters.
        @return A tupple as (weight matrix, bias vector).
        """
        return (self.weights, self.biases)


    def forwardPropagation(self,
                           inputs: list[float],
                           sigma) -> list[float]:
        """
        A method to compute the inputs throw the layer.
        @param  inputs  The inputs to process.
        @param  sigma   The activation function, such as sigmoid or ReLU.
        @return The result of computation.
        """


    def applyGradient(self,
                      gradientWeights: list[list[float]],
                      gradientBiases: list[float]):
        """
        Method to apply the gradient to weigths and biases.
        @param  gradientWeights The gradient to apply to weigths.
        @param  gradientBiases  The gradient to apply to biases.
        """
