# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/28 13:22:16 by gugus             #+#    #+#              #
#    Updated: 2025/02/28 16:41:43 by gugus            ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import random

class Matrix:
    """
    Class for 2D matrix of float.
    """

    def __init__(self, size: tuple[int, int],
                 data: list[list[float]] = None) -> 'Matrix':
        """
        Constructor of matrix class.
        @param  size    The matrix size.
        @param  data    The data to map. None by default for null matrix.
        """
        self.data = []
        for y in range(size[1]):
            line = []
            for x in range(size[0]):
                if data != None:
                    line.append(data[y][x])
                else:
                    line.append(0)
            self.data.append(line)

        self.width = size[0]
        self.height = size[1]


    def __getitem__(
            self,
            index: tuple) -> float:
        """
        Operator overload get a value from self[].
        @param  index   A tuple of index as (x, y).
        @return The number at this index.
        """
        x, y = index
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError(f"Matrix : Position ({x},{y}) is out of bounds")
        return self.data[y][x]


    def __setitem__(
            self,
            index: tuple,
            number: float):
        """
        Operator overload set a value from self[].
        @param  index   A tuple of index as (x, y).
        @param  number  The number to set.
        """
        x, y = index
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            raise ValueError(f"Matrix : Position ({x},{y}) is out of bounds")
        self.data[y][x] = number


    def __str__(self) -> str:
        """
        Method to create a string from to matrix.
        @return The matrix as a string.
        """
        string = ""
        for y in range(self.height):
            string += '|'
            for x in range(self.width):
                if x > 0:
                    string += ','
                string += f"{self.data[y][x]:5.2f}"
            string += '|\n'

        return string


    def __repr__(self) -> str:
        """
        Method to create a string from to matrix.
        @return The matrix as a string.
        """
        return self.__str__()


    def __add__(self, mat: 'Matrix') -> 'Matrix':
        """
        Operator overload for self + mat.
        @param  mat A matrix to add.
        @return Result of matrix add.
        """
        test = type(mat)
        if test != Matrix:
            raise ValueError(f"Matrix : Cannot add matrix with {test}")

        return self.add(mat)


    def __radd__(self, mat) -> 'Matrix':
        """
        Operator overload for mat + self.
        @param  mat A matrix to add.
        @return Result of matrix add.
        """
        return self.add(mat)


    def __sub__(self, mat) -> 'Matrix':
        """
        Operator overload for self - mat.
        @param  mat A matrix to sub.
        @return Result of matrix sub.
        """
        test = type(mat)
        if test != Matrix:
            raise ValueError(f"Matrix : Cannot sub matrix with {test}")

        return self.sub(mat)


    def __rsub__(self, mat) -> 'Matrix':
        """
        Operator overload for self - mat.
        @param  mat A matrix to sub.
        @return Result of matrix sub.
        """
        test = type(mat)
        raise ValueError(f"Matrix : Cannot sub matrix with {test}")


    def __mul__(self, object) -> 'Matrix':
        """
        Operator overload for self * obj. obj can be a int, float or matrix.
        @param  object  An object to multiply.
        @return Result of multiply.
        """
        test = type(object)
        if test in [int, float]:
            return self.scale(object)

        if test == Matrix:
            return self.mul(object)

        raise ValueError(f"Matrix : Cannot mul matrix with {test}")


    def __rmul__(self, object) -> 'Matrix':
        """
        Operator overload for obj * self. obj can be a int or a float.
        @param  object  An object to multiply.
        @return Result of multiply.
        """
        self.__mul__(object)


    def __neg__(self) -> 'Matrix':
        """
        Method to compute -self.
        @return The matrix mulitply by -1.
        """
        return self.scale(-1)


    def add(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix addition.
        @param  mat The matrix to add.
        @return The result of the matrix addition.
        """
        if self.width != mat.width or self.height != mat.height:
            raise ValueError("Matrix : Cannot add matrix different size")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] + mat.data[y][x]

        return result


    def sub(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix substraction.
        @param  mat The matrix to sub.
        @return The result of the matrix substraction.
        """
        if self.width != mat.width or self.height != mat.height:
            raise ValueError("Matrix : Cannot sub matrix different size")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] - mat.data[y][x]

        return result


    def scale(self, number: float) -> 'Matrix':
        """
        Method to make matrix scale.
        @param  mat The number for scaling.
        @return The result of the matrix scaling.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] * number

        return result


    def mul(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix multiplication.
        @param  mat The matrix to multiply with.
        @return The result of the matrix multiplication.
        """
        if self.width != mat.height:
            raise ValueError("Matrix : Cannot multiply matrix of size"
                             + f"[{self.width},{self.height}] by"
                             + f"[{mat.width},{mat.height}]")

        result = Matrix((mat.width, self.height))

        for y in range(self.height):
            for x in range(mat.height):
                value = 0
                for i in range(self.width):
                    value += self.data[y][i] * mat.data[i][x]
                result.data[y][x] = value

        return result


    def transpose(self) -> 'Matrix':
        """
        Method to compute the transpose of a 2d matrix.
        @param  mat     The matrix to transpose.
        @param  size    The size of the matrix.
        @return The matrix transposed.
        """
        result = Matrix((self.height, self.width))

        for y in range(self.width):
            for x in range(self.height):
                result.data[y][x] = self.data[x][y]

        return result


    def mapData(self, data: list[list[float]]):
        """
        Method to fill map a list of list of float to the matrix.
        """
        if len(data) != self.height:
            raise ValueError("Matrix : Bad map data size")

        for y in range(self.height):

            if len(data[y]) != self.width:
                raise ValueError("Matrix : Bad map data size")

            for x in range(self.width):
                self.data[y][x] = data[y][x]


    def fillRandom(self):
        """
        Method to fill the matrix with random values between 0 and 1.
        """
        for y in range(self.height):
            for x in range(self.width):
                self.data[y][x] = random.random()


    def getData(self) -> list[list[float]]:
        """
        Method to get data of the matrix.
        @return Copy of matrix data.
        """
        return self.data.copy()


    def copy(self) -> 'Matrix':
        """
        Method to create a copy of the matrix.
        @return The copy of the matrix.
        """
        return Matrix((self.width, self.height), self.data)


    def size(self) -> tuple[int, int]:
        """
        Method to get the size of the matrix.
        @return A tupple of matrix size like (width, height).
        """
        return (self.width, self.height)
