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

import torch
import torch.nn as nn

from strinng.hypergraph import Hyperedge, Hypergraph, Vertex


class Neuron(nn.Module):
    """A single neuron."""
    def __init__(self, n_in: int, act_fn=nn.Identity) -> None:
        super().__init__()
        weights = torch.randn(n_in)
        self.weights = nn.Parameter(weights)
        self.act_fn = act_fn()

    def forward(self, *xs):
        x = torch.stack(xs)
        if self.act_fn is None:
            return self.weights @ x
        return self.act_fn(self.weights @ x)

    def _get_name(self):
        return ('Neuron\nActivation=' +
                f'{self.act_fn._get_name()}')


def create_neuron(n_in: int, act_fn=None) -> Hypergraph:
    neuron = Hypergraph()
    neuron.inputs = [neuron.add_vertex(Vertex())
                     for _ in range(n_in)]
    neuron.outputs = [neuron.add_vertex(Vertex())]
    neuron.add_edge(Hyperedge(Neuron(n_in, act_fn),
                              neuron.inputs.copy(), neuron.outputs.copy()))
    return neuron


def create_fully_connected(n_in: int, n_neurons: int,
                           act_fn=nn.Identity) -> Hypergraph:
    """Return a fully connected layer."""
    layer = Hypergraph()
    layer.inputs = [layer.add_vertex(Vertex())
                    for _ in range(n_in)]
    layer.outputs = [layer.add_vertex(Vertex())
                     for _ in range(n_neurons)]
    # Create the neurons in the fully connected layer
    for i in range(n_neurons):
        layer.add_edge(Hyperedge(Neuron(n_in, act_fn),
                       layer.inputs.copy(), [layer.outputs[i]]))
    return layer


def create_flatten() -> Hypergraph:
    """Return a flatten hypergraph."""
    flatten = Hypergraph()
    flatten.inputs = [flatten.add_vertex(Vertex())]
    flatten.outputs = [flatten.add_vertex(Vertex())]
    flatten.add_edge(
        Hyperedge(nn.Flatten(),
                  flatten.inputs.copy(),
                  flatten.outputs.copy())
    )
    return flatten


def create_linear(in_features: int, out_features: int,
                  bias: bool = True) -> Hypergraph:
    """Return a linear layer."""
    linear = Hypergraph()
    linear.inputs = [linear.add_vertex(Vertex())]
    linear.outputs = [linear.add_vertex(Vertex())]
    linear.add_edge(
        Hyperedge(nn.Linear(in_features, out_features, bias),
                  linear.inputs.copy(),
                  linear.outputs.copy())
    )
    return linear


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
