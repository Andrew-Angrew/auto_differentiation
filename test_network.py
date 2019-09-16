#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

from network import cross_entropy_loss

def test_cross_entropy():
    n_classes = 2
    test_size = 3
    learning_rate = 0.1
    loss_function = cross_entropy_loss(n_classes)
    probs = np.ones(n_classes) / n_classes
    labels = np.array([0, 0, 1])
    for i in range(100):
        loss_function([p * np.ones(test_size) for p in probs] + [labels])
        loss_function.back_prop([np.ones(test_size)])
        prob_ders = loss_function.get_input_ders()[:n_classes]
        prob_ders = np.array([ders.mean() for ders in prob_ders])
        prob_ders -= prob_ders.mean()
        probs -= learning_rate * prob_ders
    assert np.allclose(probs, [2/3, 1/3])

test_cross_entropy()
