#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

from random import random
from matplotlib import pyplot as plt
import numpy as np

from dual_numbers import sin, log, cos, exp, D
from matrix import SquareMatrix
from utils import approx_der

def ssin(x):
    for i in range(100):
        x = sin(x)
    return x

def trivial_function(x):
    return log(exp(x))

def strange_function(x):
    return sin(x + cos(x + sin(x)))

MATRIX_SIZE = 3
A = SquareMatrix([
    [random() for j in range(MATRIX_SIZE)]
    for i in range(MATRIX_SIZE)
])
#A = SquareMatrix([[1, 1], [1, 0.35]])
def matrix_function(t):
    return log((A * t).exp().det())

def test(f, start=1, stop=None, N=101, tol=0.001, plot=False):
    if stop is None:
        stop = start
        start = 0
    x_range = np.linspace(start, stop, N)
    ders = [D(f)(x) for x in x_range]
    approx_ders = [approx_der(f, x) for x in x_range]
    if not np.allclose(ders, approx_ders, atol=tol) or plot:
        print("Fail for function {}".format(f.__name__))
        plt.plot(x_range, ders, label="ders", color="blue")
        plt.scatter(x_range, approx_ders, label="approximate ders", color="red")
        plt.legend()
        plt.show()

test(sin)
test(ssin)
test(trivial_function)
test(strange_function)
print(A.trace())
test(matrix_function, plot=True)
