#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from graphviz import Digraph
import numpy as np

from utils import Mapping, to_list

class Node:
    def __init__(self, val=0, der=0):
        self.val = val
        self.der = der
        self.dependencies = []
    
    def calc(self):
        return self.val

    def back_prop(self):
        pass

    def __add__(self, other):
        return Add([self, other])
    
    def __mul__(self, other):
        return Mul([self, other])
    
    def __neg__(self):
        return self * Constant(-1)
    
    def __sub__(self, other):
        return self + (-other)

    def __truediv__(self, other):
        return self * Inv(other)

class Constant(Node):
    pass

class Parameter(Node):
    pass

class Input(Node):
    pass

class Operation(Node):
    ARITY = None
    def __init__(self, dependencies):
        dependencies = to_list(dependencies)
        assert all(isinstance(node, Node) for node in dependencies)
        if self.ARITY is not None:
            assert len(dependencies) == self.ARITY, "wrong number of input nodes"
        self.val = 0
        self.der = 0
        self.dependencies = dependencies

class Add(Operation):
    def calc(self):
        self.val = sum(node.val for node in self.dependencies)

    def back_prop(self):
        for node in self.dependencies:
            node.der += self.der

class Mul(Operation):
    ARITY = 2
    def calc(self):
        self.val = 1
        for node in self.dependencies:
            self.val *= node.val

    def back_prop(self):
        node1, node2 = self.dependencies
        node1.der += self.der * node2.val
        node2.der += self.der * node1.val

class Equals(Operation):
    ARITY = 2
    def calc(self):
        node1, node2 = self.dependencies
        self.val = (node1.val == node2.val)

    def back_prop(self):
        for node in self.dependencies:
            node.der = np.nan

class UnaryOperation(Operation):
    ARITY = 1

def make_unary_operation(name, func, func_der):
    class WrappingOperation(UnaryOperation):
        def calc(self):
            self.val = func(self.dependencies[0].val)
        
        def back_prop(self):
            self.dependencies[0].der += (
                self.der * func_der(self.dependencies[0].val)
            )
    WrappingOperation.__name__ = name
    return WrappingOperation

Inv = make_unary_operation("Inv", lambda x: 1/x, lambda x: -1 / (x * x))
Sin = make_unary_operation("Sin", np.sin, np.cos)
Cos = make_unary_operation("Cos", np.cos, lambda x: -np.sin(x))
Exp = make_unary_operation("Exp", np.exp, np.exp)
Log = make_unary_operation("Log", np.log, lambda x: 1 / x)

class Circuit:
    def __init__(self, input_nodes, output_nodes):
        assert isinstance(input_nodes, list)
        assert all(isinstance(node, Input) for node in input_nodes)
        assert isinstance(output_nodes, list)
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes

        seen_nodes = set()
        added_nodes = set()
        self.nodes = []
        self.params = []

        def add_node_dependencies(node):
            if node in added_nodes:
                return
            assert node not in seen_nodes
            seen_nodes.add(node)
            for dependency in node.dependencies:
                add_node_dependencies(dependency)
            self.nodes.append(node)
            added_nodes.add(node)
            if isinstance(node, Parameter):
                self.params.append(node)

        for out_node in output_nodes:
            add_node_dependencies(out_node)

    def __call__(self, inputs):
        assert len(inputs) == len(self.input_nodes), (
            len(inputs), len(self.input_nodes))
        for node, val in zip(self.input_nodes, inputs):
            node.val = val
        for node in self.nodes:
            node.der = 0
            node.calc()
        return [node.val for node in self.output_nodes]

    def back_prop(self, output_ders):
        assert len(output_ders) == len(self.output_nodes)
        for node, der in zip(self.output_nodes, output_ders):
            node.der = der
        for node in reversed(self.nodes):
            node.back_prop()

    def get_input_ders(self):
        return [node.der for node in self.input_nodes]

    def get_param_ders(self):
        return Mapping((param, param.der) for param in self.params)

    def update_params(self, param_deltas):
        assert isinstance(param_deltas, Mapping)
        for param, delta in param_deltas.items():
            param.val += delta

    def __repr__(self):
        node_ids = {node: i for i, node in enumerate(self.nodes)}
        def node_repr(node):
            dependency_list = ", ".join(
                str(node_ids[dependency]) for dependency in node.dependencies
            )
            return "{} {}: {}".format(
                node_ids[node], node.__class__.__name__, dependency_list
            )
        return "\n".join(node_repr(node) for node in self.nodes)
    
    def clone(self):
        clones = {}
        cloned_nodes = []
        for node in self.nodes:
            if isinstance(node, Operation):
                cloned_nodes.append(node.__class__(
                    [clones[d] for d in node.dependencies])
                )
            else:
                assert isinstance(node, (Constant, Input, Parameter))
                cloned_nodes.append(node.__class__(node.val, node.der))
            clones[node] = cloned_nodes[-1]
        for node in self.input_nodes:
            if node not in clones:
                clones[node] = Input(node.val, node.der)
        return Circuit(
            [clones[node] for node in self.input_nodes],
            [clones[node] for node in self.output_nodes]
        )
        
    def view(self):
        graph = Digraph()
        graph.attr(size="10,10", rankdir="DU")
        def node_repr(node):
            return "{}: val={}; der={}".format(
                node.__class__.__name__, node.val, node.der
            )
        for node in self.nodes:
            graph.node(str(id(node)), label=node_repr(node))
            for dependency in node.dependencies:        
                graph.edge(str(id(dependency)), str(id(node)))
        graph.view()
