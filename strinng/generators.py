#    Copyright 2023 Alexander Koziell-Pipe

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


class Add(nn.Module):
    def forward(self, *xs):
        return sum(xs)

    def _get_name(self):
        return '+'


add = Add()


def add_residual(network: Hypergraph) -> Hypergraph:
    """Add a residual connection to a hypergraph."""
    network = deepcopy(network)
    new_output = network.add_vertex(Vertex())

    network.add_edge(
        Hyperedge(add,
                  network.outputs + network.inputs,
                  [new_output])
    )
    network.outputs = [new_output]
    return network
