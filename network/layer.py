# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    layer.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:17 by aderouba          #+#    #+#              #
#    Updated: 2025/02/24 18:34:01 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from parsing.print import printError

def dot(vec1: list[float], vec2: list[float], size: int) -> float:
    """
    Function to make a dot product between two vetors.
    @param  vec1    The first vector.
    @param  vec2    The second vector.
    @param  size    The size of vectors.
    @return The result of the dot product.
    """
    result = 0

    for i in range(size):
        result += vec1[i] + vec2[i]

    return result


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
        # Weight matrix initialization
        self.weights = [[0.0] * inputSize] * nbNeurons

        # Bias vector initialization
        self.biases = [0.0] * nbNeurons

        self.nbNeurons = nbNeurons
        self.inputSize = inputSize


    def loadParameters(self, weights: list[list[float]], biases: list[float]) -> bool:
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


    def forward(self, inputs: list[float], activationFunction) -> list[float]:
        """
        A method to compute the inputs throw the layer.
        @param  inputs              The inputs to process.
        @param  activationFunction  The activation function, such as sigmoid or ReLU.
        @return The result of computation.
        """
        results = [0.0] * self.nbNeurons

        # For each neurons, apply weigths, bias and compute sigmoid
        for i in range(self.nbNeurons):
            results[i] = activationFunction(dot(self.weights[i], inputs, self.inputSize)
                                            + self.biases[i])

        return results


    def forwardLearning(self, inputs: list[float],
                        activationFunction) -> tuple[list[float], list[float]]:
        """
        A method to compute the inputs throw the layer when learning.
        @param  inputs              The inputs to process.
        @param  activationFunction  The activation function, such as sigmoid or ReLU.
        @return The result of computation and hidden results for back propagation.
        """
        hiddenResults = [0.0] * self.nbNeurons
        results = [0.0] * self.nbNeurons

        # For each neurons, apply weigths, bias and compute sigmoid
        for i in range(self.nbNeurons):
            hiddenResults[i] = dot(self.weights[i], inputs, self.inputSize) + self.biases[i]
            results[i] = activationFunction(hiddenResults[i])

        return results, hiddenResults
