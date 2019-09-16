#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from circuit import Parameter, Input, Constant, Circuit
from circuit import Log, Exp, Add, Equals

def add_dense_layer(input_nodes, layer_size):
    input_size = len(input_nodes)
    params = [
        [Parameter() for j in range(input_size)]
        for i in range(layer_size)
    ]
    return [
        Add([p * in_ for p, in_ in zip(neuron_params, input_nodes)])
        for neuron_params in params
    ]

def add_softmax_layer(input_nodes):
    exponentiated_inputs = [Exp(node) for node in input_nodes]
    exp_sum = Add(exponentiated_inputs)
    return [node / exp_sum for node in exponentiated_inputs]

def build_shallow_network(input_size, output_size):
    network_inputs = [Input() for i in range(input_size)]
    output_nodes = add_dense_layer(network_inputs, output_size)
    output_nodes = add_softmax_layer(output_nodes)
    return Circuit(network_inputs, output_nodes)

def cross_entropy_loss(classes_count):
    class_probs = [Input() for i in range(classes_count)]
    class_label = Input()
    out = -Log(Add([
        prob * Equals([class_label, Constant(i)])
        for i, prob in enumerate(class_probs)
    ]))
    return Circuit(class_probs + [class_label], [out])


