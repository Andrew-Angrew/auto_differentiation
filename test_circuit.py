#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from pytest import approx
from matplotlib import pyplot as plt

from circuit import Node, Input, Parameter, Constant, Circuit
from circuit import Inv, Sin, Cos, Log, Exp
from optimizer import Optimizer
from utils import approx_der, Mapping

def test_simple_calc():
    x1, x2 = Input(), Input()
    coef1, coef2 = Parameter(1), Parameter(3)
    func = Circuit([x1, x2], [x1 * coef1 + x2 * coef2])
    assert func([1, 1]) == [4]
    assert func([10, -1]) == [7]
    
    func_clone = func.clone()
    coef1.val = -1
    assert func([1, 1]) == [2]
    assert func([10, -1]) == [-13]
    
    assert func_clone([1, 1]) == [4]
    assert func_clone([10, -1]) == [7]
    
def test_unary_operation(operation, start=1, stop=None, N=101, tol=0.001):
    if stop is None:
        stop = start
        start = 0
    x_range = np.linspace(start, stop, N)
    
    in_ = Input()
    out = operation(in_)
    circ = Circuit([in_], [out])
    f = lambda x: circ([x])[0]
    def df(x):
        circ([x])
        circ.back_prop([1])
        return circ.get_input_ders()[0]
    ders = [df(x) for x in x_range]
    approx_ders = [approx_der(f, x) for x in x_range]
    if not np.allclose(ders, approx_ders, atol=tol):
        print("Fail for operation {}".format(operation.__name__))
        plt.plot(x_range, ders, label="ders")
        plt.plot(x_range, approx_ders, label="approximate ders")
        plt.legend()
        plt.show()

def test_simple_descent(verbose=False):
    x = Input()
    mean = Parameter(0)
    func = Circuit([x], [(x - mean) * (x - mean)])

    sample_size = 100
    n_epochs = 20
    learning_rate = 0.1

    np.random.seed(0)
    data = np.random.random(sample_size)
    if verbose:
        print("Data mean:", data.mean())
    
    for epoch in range(n_epochs):
        ders = Mapping()
        values = []
        for point in data:
            values.append(func([point]))
            func.back_prop([1])
            param_der = func.get_param_ders()
            assert approx(param_der[mean]) == -2 * (point - mean.val)
            ders = ders + param_der
        func.update_params((-learning_rate / sample_size) * ders)
        if verbose:
            print("{:<2}: param={:<8.6} loss={:.6}".format(
                epoch, mean.val, np.mean(values)
            ))
    assert abs(mean.val - data.mean()) < (1 - learning_rate / 0.5) ** n_epochs + 1e-10

def test_optimizer(verbose=False):
    coef = Parameter(0)
    in_ = Input()
    model = Circuit([in_], [in_ * coef])
    y_pred, y_true = Input(), Input()
    loss_function = Circuit(
        [y_pred, y_true],
        [(y_pred - y_true) * (y_pred - y_true)]
    )
    optimizer = Optimizer(model, loss_function, verbose)
    np.random.seed(0)
    sample_size = 100

    y = np.random.random(sample_size)
    X = np.ones(sample_size).reshape((-1, 1))

    validation_y = np.random.random(sample_size)
    validation_X = np.ones(sample_size).reshape((-1, 1))
    
    model = optimizer.fit(
        X, y, eval_set=(validation_X, validation_y),
        n_epochs=10, learning_rate=0.1, batch_size=20
    )
    assert abs(model([1])[0] - np.mean(y)) < 0.01

test_simple_calc()
for operation in [Sin, Cos, Exp]:
    test_unary_operation(operation)
test_unary_operation(Log, 1, 10)
test_unary_operation(Inv, 1, 10)
test_simple_descent()
test_optimizer()

