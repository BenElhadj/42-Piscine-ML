import numpy as np


class Matrix():

    def __init__(self, data=[[1.0, 2.0], [3.0, 4.0]]):
        self.data = data

    def OtherMatrixShapeCheck(func):
        def wrapper(self, other):
            if (not isinstance(other, Matrix) and not isinstance(other, Vector)):
                raise BaseException("second matrix is not instance of Matrix")
            if func.__name__ != '__mul__':
                if (self.shape != other.shape):
                    raise BaseException(
                        f"operands could not be broadcast together with shapes {self.shape} {other.shape}")
            else:
                if (self.shape[1] != other.shape[0]):
                    raise BaseException(
                        f"matmul: mismatch (n?,k),(k,m?)->(n?,m?)")
            res = func(self, other)
            return res
        return wrapper

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        for index, line in enumerate(data):
            if index >= 1 and len(data[index]) != len(data[index - 1]):
                raise BaseException(
                    f"SizeError: {data[index]} != {data[index - 1]}")
            for item in line:
                if isinstance(item, int) and isinstance(item, float) and item != item:
                    raise BaseException(
                        f"TypeError: {item} is not (dtype('float') or dtype('int64'))")
        self._data = data
        self._shape = (len(self._data), len(self._data[0])) if isinstance(
            self._data[0], list) or isinstance(self._data[0], np.ndarray) else (len(self._data), 1)

    @property
    def shape(self):
        return self._shape

    @shape.setter
    def shape(self, shape):
        try:
            if not isinstance(shape, tuple):
                raise BaseException(
                    f"TypeError: '{type(shape).__name__}' object cannot be interpreted as an integer")
            if (len(shape) != 2):
                raise BaseException(
                    f"ValueError: cannot reshape array of size {len(shape)} into shape {shape}")
            self.data = [[0 for row in range(shape[1])]
                         for column in range(shape[0])]
        except (Exception, BaseException) as error:
            print(error)

    def T(self):
        T = Matrix()
        T.data = [[self.data[i][j]
                   for i in range(self.shape[0])] for j in range(self.shape[1])]
        return T

    @OtherMatrixShapeCheck
    def __add__(self, other):
        add_matrix = Matrix()
        add_matrix.data = [[self.data[i][j] + other.data[i][j]
                            for j in range(self.shape[1])] for i in range(self.shape[0])]
        return add_matrix

    @OtherMatrixShapeCheck
    def __radd__(self, other):
        radd_matrix = Matrix()
        radd_matrix.data = [
            [self.data[i][j] + other for j in range(self.shape[1])] for i in range(self.shape[0])]
        return radd_matrix

    @OtherMatrixShapeCheck
    def __sub__(self, other):
        sub_matrix = Matrix()
        sub_matrix.data = [[self.data[i][j] - other.data[i][j]
                            for j in range(self.shape[1])] for i in range(self.shape[0])]
        return sub_matrix

    @OtherMatrixShapeCheck
    def __rsub__(self, other):
        rsub_matrix = Matrix()
        rsub_matrix.data = [
            [self.data[i][j] - other for j in range(self.shape[1])] for i in range(self.shape[0])]
        return rsub_matrix

    @OtherMatrixShapeCheck
    def __truediv__(self, other):
        truediv_matrix = Matrix()
        truediv_matrix.data = [[self.data[i][j] / other.data[i][j]
                                for j in range(self.shape[1])] for i in range(self.shape[0])]
        return truediv_matrix

    @OtherMatrixShapeCheck
    def __rtruediv__(self, other):
        rtruediv_matrix = Matrix()
        rtruediv_matrix.data = [[other / self.data[i][j]
                                 for j in range(self.shape[1])] for i in range(self.shape[0])]
        return rtruediv_matrix

    @OtherMatrixShapeCheck
    def __mul__(self, other):
        mul_matrix = Matrix()
        mul_matrix.data = [[sum(a * b for a, b in zip(X_row, Y_col))
                            for Y_col in zip(*other.data)] for X_row in self.data]
        return mul_matrix

    @OtherMatrixShapeCheck
    def __rmul__(self, other):
        rmul_matrix = Matrix()
        rmul_matrix.data = [[sum(a * b for a, b in zip(X_row, Y_col))
                             for Y_col in zip(*self.data)] for X_row in other.data]
        return rmul_matrix

    def __str__(self):
        return ''.join(str(self.data)).replace('],', ']\n').replace(',', '')

    def __repr__(self):
        return 'Matrix([' + ''.join(',\n,'.join(str(col) for col in self.data)).replace(',[', '\t[') + '])'


class Vector(Matrix):

    def __init__(self, data=[[0]]):
        super().__init__()
        if (len(data) != 1 and isinstance(data[0], list) and len(data[0]) != 1):
            raise BaseException('SizeError: size of Vector !!')
        self.data = data

    def dot(self, other):
        dot_vector = Vector()
        dot_vector.data = (self * other).data
        return dot_vector

    def __mul__(self, other):
        mul_vector = Vector()
        mul_vector.data = (super().__mul__(other)).data
        return mul_vector

    def __add__(self, other):
        add_vector = Vector()
        add_vector.data = (super().__add__(other)).data
        return add_vector

    def __radd__(self, other):
        radd_vector = Vector()
        radd_vector.data = (super().__radd__(other)).data
        return radd_vector

    def __sub__(self, other):
        sub_vector = Vector()
        sub_vector.data = (super().__sub__(other)).data
        return sub_vector

    def __rsub__(self, other):
        rsub_vector = Vector()
        rsub_vector.data = (super().__rsub__(other)).data
        return rsub_vector

    def __truediv__(self, other):
        truediv_vector = Vector()
        truediv_vector.data = (super().__truediv__(other)).data
        return truediv_vector

    def __rtruediv__(self, other):
        rtruediv_vector = Vector()
        rtruediv_vector.data = (super().__rtruediv__(other)).data
        return rtruediv_vector

    def __repr__(self):
        return 'Vector([' + ''.join(',\n,'.join(str(col) for col in self.data)).replace(',[', '\t[') + '])'
