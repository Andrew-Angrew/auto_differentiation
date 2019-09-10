#!/usr/bin/env python3.6
# -*- coding: utf-8 -*-

import math

# Object of this class describes an expression a + b * epsilon + o(epsilon).
# Define all arithmetic operations 
class DualNumber:
    def __init__(self, a, b=None):
        if b is None and isinstance(a, DualNumber):
            self.a = a.a
            self.b = a.b
        else:
            self.a = a
            self.b = 0 if b is None else b

    def __add__(self, other):
        other = DualNumber(other)
        return DualNumber(self.a + other.a, self.b + other.b)

    def __sub__(self, other):
        other = DualNumber(other)
        return DualNumber(self.a - other.a, self.b - other.b)

    def __mul__(self, other):
        other = DualNumber(other)
        return DualNumber(self.a * other.a, self.a * other.b + other.a * self.b)

    def __truediv__(self, other):
        other = DualNumber(other)
        if other.a == 0:
            return None
        return DualNumber(
            self.a / other.a, 
            (self.b * other.a - other.b * self.a) / (other.a**2)
        )

    def __pow__(self, power):
        if isinstance(power, (int, float)):
            if power != 0:
                return DualNumber(self.a ** power,
                                  power * (self.a ** (power - 1)) * self.b)
            elif self.a != 0:
                return 1
            else:
                return None
        elif isinstance(power, DualNumber):
            raise NotImplementedError
        else:
            raise ValueError

    def __neg__(self):
        return DualNumber(-self.a, -self.b)
    
    def __abs__(self):
        if self < 0:
            return DualNumber(-self.a, -self.b)
        else:
            return DualNumber(self.a, self.b)            

    def __lt__(self, other):
        other = DualNumber(other)
        return (self.a, self.b) < (other.a, other.b)

    def __eq__(self, other):
        other = DualNumber(other)
        return (self.a, self.b) == (other.a, other.b)

    def __le__(self, other):
        other = DualNumber(other)
        return (self.a, self.b) <= (other.a, other.b)

#auxilliaty methods:
    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return DualNumber(other).__sub__(self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return DualNumber(other).__truediv__(self)

    def __str__(self):
        return "{}{:+}epsilon".format(self.a, self.b)

    def __repr__(self):
        return str(self)

def sin(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.sin(x.a), math.cos(x.a) * x.b)
    return math.sin(x)

def cos(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.cos(x.a), -math.sin(x.a) * x.b)
    return math.cos(x)

def exp(x):
    if isinstance(x, DualNumber):
        return DualNumber(math.exp(x.a), math.exp(x.a) * x.b)
    return math.exp(x)

def log(x):
    if isinstance(x, DualNumber):
        if x.a <= 0.:
            raise ValueError
        return DualNumber(log(x.a), x.b / x.a)
    return math.log(x)

def D(f):
    def f_derivative(x):
        x = DualNumber(x, 1)
        return f(x).b
    return f_derivative


