#    Copyright 2024 Alexander Koziell-Pipe

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#        http://www.apache.org/licenses/LICENSE-2.0

#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Commonly used PyTorch modules."""
from copy import deepcopy
from functools import reduce

import torch
import torch.nn as nn

from strinng.hypergraph import Hyperedge, Hypergraph, Vertex


class NeuronAggregator(nn.Module):
    """The input aggregator of a neuron."""
    def __init__(self, n_in: int) -> None:
        super().__init__()
        weights = torch.randn(n_in)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        return self.weights @ x

    def _get_name(self):
        return '+'


def create_neuron(n_in: int, act_fn=None) -> Hypergraph:
    neuron = Hypergraph()
    neuron.inputs = [neuron.add_vertex(Vertex())
                     for _ in range(n_in)]
    neuron.outputs = [neuron.add_vertex(Vertex())]
    agg = neuron.add_edge(Hyperedge(NeuronAggregator(n_in),
                                    neuron.inputs.copy(), [None]))
    if act_fn is None:
        neuron.add_connection(agg, neuron.outputs[0],
                              0, False)
    else:
        v = neuron.add_vertex(Vertex())
        neuron.add_connection(agg, v,
                              0, False)
        neuron.add_edge(Hyperedge(act_fn, [v],
                                  neuron.outputs.copy()))
    return neuron


def create_fully_connected(n_in: int, n_neurons: int,
                           act_fn=None) -> Hypergraph:
    """Return a fully connected layer."""
    # Create the neurons in the fully connected layer
    neurons = [create_neuron(n_in, act_fn) for _ in range(n_neurons)]
    layer = reduce(lambda x, y: x @ y, neurons)
    # Inputs from the previous layer are sent to mulitple neurons
    # in the fully connected layer, therefore inputs between neurons
    # correspond to the same value. Hence we merge them.
    to_merge = []
    for i in range(n_in):
        for j in range(1, n_neurons):
            to_merge.append((layer.inputs[i], layer.inputs[i + n_in * j]))
    for v1, v2 in to_merge:
        layer.merge_vertices(v1, v2)
    layer.inputs = layer.inputs[:n_in]

    return layer


class Linear(Hypergraph):
    """A linear layer."""

    def __init__(self,
                 in_features: int, out_features: int,
                 bias: bool = True) -> None:
        super().__init__()
        self.inputs = [self.add_vertex(Vertex())]
        self.outputs = [self.add_vertex(Vertex())]
        self.add_edge(
            Hyperedge(nn.Linear(in_features, out_features, bias),
                      self.inputs.copy(),
                      self.outputs.copy())
        )


class Flatten(Hypergraph):
    """A Flatten operation."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs = [self.add_vertex(Vertex())]
        self.outputs = [self.add_vertex(Vertex())]
        self.add_edge(
            Hyperedge(nn.Flatten(),
                      self.inputs.copy(),
                      self.outputs.copy())
        )


class CrossEntropyLoss(Hypergraph):
    """Cross entropy loss."""

    def __init__(self) -> None:
        super().__init__()
        self.inputs = [self.add_vertex(Vertex()),
                       self.add_vertex(Vertex())]
        self.outputs = [self.add_vertex(Vertex())]
        self.add_edge(
            Hyperedge(nn.CrossEntropyLoss(),
                      self.inputs.copy(),
                      self.outputs.copy())
        )


def add_loss(network: Hypergraph, loss: Hypergraph) -> Hypergraph:
    """Adds a loss function to a network."""
    label = network.add_vertex(Vertex())
    network.inputs.append(label)
    network.outputs.append(label)
    return network >> loss


class Sum(nn.Module):
    def forward(self, *xs):
        return sum(xs)

    def _get_name(self):
        return '+'


sum_module = Sum()


def add_residual(network: Hypergraph) -> Hypergraph:
    """Add a residual connection to a hypergraph."""
    network = deepcopy(network)
    new_output = network.add_vertex(Vertex())

    network.add_edge(
        Hyperedge(sum_module,
                  network.outputs + network.inputs,
                  [new_output])
    )
    network.outputs = [new_output]
    return network
