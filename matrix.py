#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from copy import deepcopy
from operator import itemgetter


epsilon = 0.1 ** 9
class SquareMatrix:
    def __init__(self, vals):
        self.n = len(vals)
        self.vals = vals
        assert all(self.n == len(row) for row in vals), "not all rows have length n"

    def __add__(self, other):
        n = self.n
        if isinstance(other, SquareMatrix):
            assert n == other.n, "matrices should have equal size"
            new_vals = [[self.vals[i][j] + other.vals[i][j] for j in range(n)]
                        for i in range(n)]
        else:
            new_vals = [[self.vals[i][j] + other for j in range(n)]
                        for i in range(n)]
        return SquareMatrix(new_vals)

    def __sub__(self, other):
        n = self.n
        if isinstance(other, SquareMatrix):
            assert n == other.n, "matrices should have equal size"
            new_vals = [[self.vals[i][j] - other.vals[i][j] for j in range(n)]
                        for i in range(n)]
        else:
            new_vals = [[self.vals[i][j] - other for j in range(n)]
                        for i in range(n)]
        return SquareMatrix(new_vals)

    def __mul__(self, other):
        n = self.n
        if isinstance(other, SquareMatrix):
            assert n == other.n, "matrices should have equal size"
            new_vals = [[sum(self.vals[i][k] * other.vals[k][j]
                             for k in range(n))
                         for j in range(n)]
                        for i in range(n)]
        else:
            new_vals = [[self.vals[i][j] * other for j in range(n)]
                        for i in range(n)]
        return SquareMatrix(new_vals)

    def inv(self):
        raise NotImplementedError

    def __truediv__(self, other):
        return self * (other.inv() if isinstance(other, SquareMatrix) else 1/other)

    def __neg__(self):
        return SquareMatrix([[-num for num in row] for row in self.vals])
    
    def det(self):
        n = self.n
        M = deepcopy(self.vals)
        ans = 1
        for i in range(n-1):
            k, v = max(enumerate((abs(M[j][i]) for j in range(i, n)), start=i),
                       key=itemgetter(1))
            if v < 0.1**9:
                return 0
            if k != i:
                M[i], M[k] = M[k], M[i]
                ans = -ans
            for k in range(i+1,n):
                c = M[k][i] / M[i][i]
                for j in range(i, n):
                    M[k][j] -= M[i][j] * c
            ans *= M[i][i]
        return ans * M[-1][-1]

    def exp(self):
        ans = SquareMatrix.E(self.n)
        tmp = SquareMatrix.E(self.n)
        for i in range(1, 1000):
            tmp *= self / i
            ans += tmp
            if tmp.norm() < epsilon ** 2:
                break
        return ans

    def trace(self):
        return sum(self.vals[i][i] for i in range(self.n))
    
    def norm(self):
        ans = 0
        n = self.n
        for i in range(n):
            for j in range(n):
                ans += self.vals[i][j] ** 2
        return ans

    @classmethod
    def E(cls, n):
        return cls([[int(i==j) for j in range(n)] for i in range(n)])

    # auxilliaty methods:
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return -self.__sub__(other)

    def __rmul__(self, other):
        #it seems to be correct since other should be a number
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other * self.inv()

    def __str__(self):
        max_len = max(len(str(num)) for row in self.vals for num in row)
        return '\n'.join(' '.join(str(num).rjust(max_len) for num in row)
                         for row in self.vals)

    def __repr__(self):
        return str(self)


