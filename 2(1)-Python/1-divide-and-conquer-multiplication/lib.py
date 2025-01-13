from __future__ import annotations
import copy


"""
TODO:
- __setitem__ 구현하기
- __pow__ 구현하기 (__matmul__을 활용해봅시다)
- __repr__ 구현하기
"""


class Matrix:
    MOD = 1000

    def __init__(self, matrix: list[list[int]]) -> None:
        self.matrix = matrix

    @staticmethod
    def full(n: int, shape: tuple[int, int]) -> Matrix:
        return Matrix([[n] * shape[1] for _ in range(shape[0])])

    @staticmethod
    def zeros(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(0, shape)

    @staticmethod
    def ones(shape: tuple[int, int]) -> Matrix:
        return Matrix.full(1, shape)

    @staticmethod
    def eye(n: int) -> Matrix:
        matrix = Matrix.zeros((n, n))
        for i in range(n):
            matrix[i, i] = 1
        return matrix

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self.matrix), len(self.matrix[0]))

    def clone(self) -> Matrix:
        return Matrix(copy.deepcopy(self.matrix))

    def __getitem__(self, key: tuple[int, int]) -> int:
        return self.matrix[key[0]][key[1]]

    def __setitem__(self, key: tuple[int, int], value: int) -> None:
        """
        matrix의 지정된 위치에 값을 설정한다.
        값은 1000보다 작아야하기에 modular 연산을 수행한다.

        Args:
            - key : 값을 설정하기 위한 인덱스.
            - value : matrix에 설정할 값.

        Returns:
            - None : 값이 설정되며 반환값이 없다.
        """
        self.matrix[key[0]][key[1]] = value % self.MOD
        pass

    def __matmul__(self, matrix: Matrix) -> Matrix:
        x, m = self.shape
        m1, y = matrix.shape
        assert m == m1

        result = self.zeros((x, y))

        for i in range(x):
            for j in range(y):
                for k in range(m):
                    result[i, j] += self[i, k] * matrix[k, j]

        return result

    def __pow__(self, n: int) -> Matrix:
        """
        matrix에 대해 거듭제곱을 구한다.
        identity matrix에 clone matrix를 n번 곱해준다.

        Args:
            - n : 거듭제곱 수를 나타낸다.

        Returns:
            - Matrix : 거듭제곱을 수행한 matrix를 반환한다.
        """
        x, y = self.shape
        result = Matrix.eye(self.shape[0])
        temp = self.clone()
        while n > 0:
            if n % 2 == 1:
                result @= temp
                n -= 1
            temp @= temp
            n //= 2
        return result
        pass

    def __repr__(self) -> str:
        """
        matrix의 문자열 표현을 반환한다.
        각 column은 공백으로 구분되며 row 사이에는 줄바꿈이 일어난다.

        Args:

        Returns:
            - str : 행렬을 나타내는 문자열.
        """
        x, y = self.shape
        result = []
        for i in range(x):
            row = []
            for j in range(y):
                row.append(str(self[i, j]))
            result.append(" ".join(row))
        return "\n".join(result)
        pass