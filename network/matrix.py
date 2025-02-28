# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    matrix.py                                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: gugus <gugus@student.42.fr>                +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2025/02/28 13:22:16 by gugus             #+#    #+#              #
#    Updated: 2025/02/28 21:25:16 by gugus            ###   ########.fr        #
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

        self.isVectorX = (self.width == 1)
        self.isVectorY = (self.height == 1)
        self.isVector = (self.isVectorX or self.isVectorY)


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


    def __add__(self, obj) -> 'Matrix':
        """
        Operator overload for self + obj.
        @param  obj A matrix or a number to add.
        @return Result of matrix add.
        """
        test = type(obj)
        if test in [int, float]:
            return self.__addNumber(obj)

        if test == Matrix:
            if obj.isVector:
                return self.__addVector(obj)
            return self.__addMatrix(obj)

        raise ValueError(f"Matrix : Cannot add matrix with {test.__name__}")


    def __radd__(self, obj) -> 'Matrix':
        """
        Operator overload for obj + self.
        @param  obj A matrix to add.
        @return Result of matrix add.
        """
        return self.__add__(obj)


    def __sub__(self, obj) -> 'Matrix':
        """
        Operator overload for self - obj.
        @param  obj A matrix to sub.
        @return Result of matrix sub.
        """
        test = type(obj)
        if test in [int, float]:
            return self.__subNumber(obj)

        if test == Matrix:
            if obj.isVector:
                return self.__subVector(obj)
            return self.__subMatrix(obj)

        raise ValueError(f"Matrix : Cannot sub matrix with {test.__name__}")


    def __rsub__(self, obj) -> 'Matrix':
        """
        Operator overload for obj - self.
        @param  obj A matrix to sub.
        @return Result of matrix sub.
        """
        test = type(obj)
        if test in [int, float]:
            return self.__subNumberReverse(obj)

        if test == Matrix:
            if obj.isVector:
                return obj.__subVector(self)
            return obj.__subMatrix(self)

        raise ValueError(f"Matrix : Cannot sub {test.__name__} by matrix")


    def __mul__(self, obj) -> 'Matrix':
        """
        Operator overload for self * obj. obj can be a int, float or matrix.
        @param  obj An object to multiply.
        @return Result of multiply.
        """
        test = type(obj)
        if test in [int, float]:
            return self.__mulNumber(obj)

        if test == Matrix:
            if self.size() == obj.size() and self.width != obj.height:
                return self.__mulMatrixElement(obj)
            return self.__mulMatrix(obj)

        raise ValueError(f"Matrix : Cannot mul matrix with {test}")


    def __rmul__(self, obj) -> 'Matrix':
        """
        Operator overload for obj * self. obj can be a int or a float.
        @param  obj An object to multiply.
        @return Result of multiply.
        """
        test = type(obj)
        if test in [int, float]:
            return self.__mulNumber(obj)

        if test == Matrix:
            if self.size() == obj.size() and self.width != obj.height:
                return self.__mulMatrixElement(obj)
            return obj.__mulMatrix(self)

        raise ValueError(f"Matrix : Cannot mul {test.__name__} by matrix")


    def __neg__(self) -> 'Matrix':
        """
        Method to compute -self.
        @return The matrix mulitply by -1.
        """
        return self.__mulNumber(-1)


    def __addNumber(self, number: float) -> 'Matrix':
        """
        Method to add a number to each matrix value.
        @param  number  The number to add.
        @return The result of self + number.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] + number

        return result


    def __addVector(self, vec: 'Matrix') -> 'Matrix':
        """
        Method to make matrix vector addition.
        @param  vec The vector to add.
        @return The result of self + vector.
        """
        # Vector X and Y
        if vec.isVectorX and vec.isVectorY:
            return self.__addNumber(vec[0, 0])

        # Vector X
        if vec.isVectorX:
            if self.height != vec.height:
                raise ValueError("Matrix : Cannot add vector of different"
                                 + f" size {self.size()}, {vec.size()}")

            result = Matrix((self.width, self.height))
            for y in range(self.height):
                for x in range(self.width):
                    result.data[y][x] = self.data[y][x] + vec.data[y][0]
            return result

        # Vector Y
        if self.width != vec.width:
            raise ValueError("Matrix : Cannot add vector of different"
                             + f" size {self.size()}, {vec.size()}")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] + vec.data[0][x]

        return result


    def __addMatrix(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix addition.
        @param  mat The matrix to add.
        @return The result of self + matrix.
        """
        if self.width != mat.width or self.height != mat.height:
            raise ValueError("Matrix : Cannot add matrix of different"
                             + f" size {self.size()}, {mat.size()}")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] + mat.data[y][x]

        return result


    def __subNumber(self, number: float) -> 'Matrix':
        """
        Method to sub a number to each matrix value.
        @param  number  The number to sub.
        @return The result of self - number.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] - number

        return result


    def __subNumberReverse(self, number: float) -> 'Matrix':
        """
        Method to sub a number by each matrix value.
        @param  number  The number to sub.
        @return The result of number - self.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = number - self.data[y][x]

        return result


    def __subVector(self, vec: 'Matrix') -> 'Matrix':
        """
        Method to make matrix vector substraction.
        @param  vec The vector to sub.
        @return The result of self - vector.
        """
        # Vector X and Y
        if vec.isVectorX and vec.isVectorY:
            return self.__subNumber(vec[0, 0])

        # Vector X
        if vec.isVectorX:
            if self.width != vec.width:
                raise ValueError("Matrix : Cannot sub vector of different"
                                 + f" size {self.size()}, {vec.size()}")

            result = Matrix((self.width, self.height))
            for y in range(self.height):
                for x in range(self.width):
                    result.data[y][x] = self.data[y][x] + vec.data[0][x]
            return result

        # Vector Y
        if self.height != vec.height:
            raise ValueError("Matrix : Cannot sub vector of different"
                             + f" size {self.size()}, {vec.size()}")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] + vec.data[y][0]

        return result


    def __subMatrix(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix substraction.
        @param  mat The matrix to sub.
        @return The result of self - matrix.
        """
        if self.width != mat.width or self.height != mat.height:
            raise ValueError("Matrix : Cannot sub matrix of different"
                             + f" size {self.size()}, {mat.size()}")

        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] - mat.data[y][x]

        return result


    def __mulNumber(self, number: float) -> 'Matrix':
        """
        Method to make matrix __mulNumber.
        @param  mat The number for scaling.
        @return The result of self * number.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] * number

        return result


    def __mulMatrix(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make matrix multiplication.
        @param  mat The matrix to multiply with.
        @return The result of self * matrix.
        """
        if self.width != mat.height:
            raise ValueError("Matrix : Cannot multiply matrix of size"
                             + f"{self.size()} and {mat.size()}")

        result = Matrix((mat.width, self.height))

        for y in range(self.height):
            for x in range(mat.width):
                value = 0
                for i in range(self.width):
                    value += self.data[y][i] * mat.data[i][x]
                result.data[y][x] = value

        return result


    def __mulMatrixElement(self, mat: 'Matrix') -> 'Matrix':
        """
        Method to make multiplication element by element.
        @param  mat The matrix to multiply with.
        @return The result of multiplication by element.
        """
        result = Matrix((self.width, self.height))

        for y in range(self.height):
            for x in range(self.width):
                result.data[y][x] = self.data[y][x] * mat.data[y][x]

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
            raise ValueError("Matrix : Bad map data")

        for y in range(self.height):

            if len(data[y]) != self.width:
                raise ValueError("Matrix : Bad map data")

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
