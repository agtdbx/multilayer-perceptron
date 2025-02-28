# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neuralNetwork.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/20 15:14:35 by aderouba          #+#    #+#              #
#    Updated: 2025/02/28 16:48:04 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from define import          INPUT_SIZE, OUTPUT_SIZE, LEARNING_RATE

from network.matrix import  Matrix
from network.sigmoid import sigmoid

class NeuralNetwork:
    """
    The class for the neural network.
    """

    def __init__(self, hidderLayers: list[int]) -> 'NeuralNetwork':
        """
        The constructor of NeuralNetwork class.
        @param  hidderLayers    The number of neurons per hidden layers.
        """
        # Create network dimensions
        dimensions = hidderLayers.copy()
        dimensions.insert(0, INPUT_SIZE)
        dimensions.append(OUTPUT_SIZE)

        self.nbLayers = len(dimensions)
        dimensions.append(INPUT_SIZE) # for l = 0, input is l - 1

        # Init weights and biases per layer
        self.weigths: list[Matrix] = []
        self.biases: list[Matrix] = []
        self.activations: list[Matrix] = []

        for l in range(0, self.nbLayers):
            print(f"input {dimensions[l - 1]}, neurons {dimensions[l]}")

            weigths = Matrix((dimensions[l - 1], dimensions[l]))
            weigths.fillRandom()
            self.weigths.append(weigths)

            biases = Matrix((dimensions[l], 1))
            biases.fillRandom()
            self.biases.append(biases)


    def loadParameters(self,
                       params: list[tuple[list[list[float]],list[float]]]):
        """
        Method to load weights and biases into the network.
        @param  params  The list of tupple (weigth matrix, bias vector).
        """
        # Give parameters to each layer
        for l in range(self.nbLayers):
            self.weigths[l].mapData(params[l][0])
            self.biases[l].mapData(params[l][1])


    def getParameters(self) -> list[tuple[list[list[float]], list[float]]]:
        """
        Method to get the network's parameters.
        @return A list as [(weight matrix, bias vector), ..].
        """
        result = []

        for l in range(self.nbLayers):
            result.append((self.weigths[l].data.copy(),
                           self.biases[l].data.copy()))

        return result


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


    def forwardPropagation(self,
                           X: Matrix) -> Matrix:
        """
        Method to process inputs through the neural network.
        @param  X       The input matrix.
        @return The network result.
        """
        self.activations: list[Matrix] = [X.copy()]

        for l in range(self.nbLayers):
            print(f"Layer {l}")
            print(f" W  {self.weigths[l].size()}")
            print(f" A  {self.activations[l].size()}")
            tkt = self.weigths[l] * self.activations[l]
            print(f" WA {tkt.size()}")
            print(f" B  {self.biases[l].size()}")
            z = tkt + self.biases[l]
            self.activations.append(sigmoid(-z))

        return self.activations[-1]


    def backwardProgatation(self, y: Matrix)-> list[tuple[Matrix, Matrix]]:
        """
        Method to compute the gradient by backward propagation.
        Return format : [(weigths gradient, biases gradient), ...].
        @param  y   The result wanted.
        @return The list of tuple of gradient.
        """
        m = y.height

        dZ = self.activations[-1] - y
        gradients: list[tuple[Matrix, Matrix]] = [None] * self.nbLayers

        for l in range(self.nbLayers - 1, -1, -1):
            dW = 1 / m * dZ * self.activations[l].transpose()

            sumDZ = 0
            for i in range(dZ.height):
                sumDZ += dZ[0, i]
            db = 1 / m * sumDZ

            gradients[l] = (dW, db)

            if l > 0:
                dZ = self.weigths[l].transpose() * self.activations[l]\
                        * -self.activations[l]


    def applyGradients(self, gradients: list[tuple[Matrix, Matrix]]):
        """
        Method to apply the gradient to weigths and biases to each layer.
        @param  gradients   The gradients to apply to weigths and biases.
        """

        for l in range(self.nbLayers):
            dW, db = gradients[l]

            self.weigths[l] = self.weigths[l] - dW * LEARNING_RATE
            self.biases[l] = self.biases[l] - db * LEARNING_RATE
