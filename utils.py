#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections.abc import Iterable
from numbers import Number

def approx_der(f, x, eps=0.0001):
    return (f(x + eps) - f(x)) / eps

class Mapping(dict):
    def __add__(self, other):
        assert isinstance(other, Mapping)
        all_keys = set(self.keys())
        all_keys.update(other.keys())
        return Mapping(
            (key, self.get(key, 0) + other.get(key, 0))
            for key in all_keys
        )
    
    def __mul__(self, num):
        assert isinstance(num, Number)
        return Mapping((key, num * val) for key, val in self.items())
    
    def __rmul__(self, num):
        return self.__mul__(num)

    def apply(self, func):
        return Mapping((key, func(val)) for key, val in self.items())
        

def to_list(x):
    if isinstance(x, Iterable):
        return list(x)
    return [x]

