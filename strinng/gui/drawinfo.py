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
"""Functionality for laying out hypergraphs in cartesian coordinates."""

from dataclasses import dataclass
from typing import Dict

import cvxpy as cp
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem

from strinng.hypergraph import Hypergraph, Hyperedge, Vertex


@dataclass
class VertexDrawInfo:
    """Rendering information for a vertex."""

    x: float
    """x-coordinate for the vertex."""
    y: float
    """y-coordinate for the vertex."""
    label: str | None
    """Retrieves a label for annotating the vertex."""


@dataclass
class EdgeDrawInfo:
    """Rendering information for a hyperedge."""

    x: float
    """x-coordinate for the edge."""
    y: float
    """y-coordinate for the edge."""
    label: str | None
    """Label to be drawn on the edge."""
    identity: bool
    """Whether the edge is an identity."""
    box_height: float
    """The height of the box for it to be displayed nicely."""


class HypergraphDrawInfo:
    """Rendering information for a hypergraph.

    Attributes:
        graph: The :py:class:`Hypergraph` instance to be rendered.
    """

    def __init__(self, graph: Hypergraph,
                 layout: str = 'convex_opt') -> None:
        """Initialize a :py:class:`HypergraphDrawInfo` instance."""
        self.graph = graph
        self.vertices: Dict[int, VertexDrawInfo] = {}
        self.edges: Dict[int, EdgeDrawInfo] = {}
        if layout == 'convex_opt':
            self.frobenius_convex_optimization_layout()

    def add_vertex(self, vertex: Vertex) -> int:
        """Add a vertex to the hypergraph and create draw info."""
        vertex_id = self.graph.add_vertex(vertex)
        self.vertices[vertex_id] = VertexDrawInfo(
            0, 0, vertex.label if hasattr(vertex, 'label') else None
        )
        return vertex_id

    def add_edge(self, edge: Hyperedge) -> int:
        """Add an edge to the hypergraph and create draw info."""
        edge_id = self.graph.add_edge(edge)
        self.edges[edge_id] = EdgeDrawInfo(
            0, 0, edge.label, edge.identity,
            1.0 if len(edge.sources) == 1 == len(edge.targets) else 2.0
        )
        return edge_id

    def add_connection(self, edge_id, vertex_id, port_num, is_input) -> None:
        """Add a connection between and edge and a vertex."""
        self.graph.add_connection(edge_id, vertex_id, port_num, is_input)

    def convex_optimization_layout(self) -> None:
        """Determine x and y coordinates for vertices and hyperedges."""
        # Decompose the graph into layers of edges
        decomposed, edge_layers = self.graph.layer_decomp()

        # Initialize vertex and edge graphics items for decomposed hypergraph
        self.vertices = {
            vertex_id: VertexDrawInfo(
                0, 0, vertex.label if hasattr(vertex, 'label') else None
            )
            for vertex_id, vertex in decomposed.vertices.items()
        }
        self.edges = {
            edge_id: EdgeDrawInfo(0, 0, edge.label, edge.identity,
                                  1.0 if len(edge.sources) == 1
                                  == len(edge.targets) else 2.0)
            for edge_id, edge in decomposed.edges.items()
        }

        # Initialize x-coordinates and rough y-coordinates
        # Each layer occupies an x-width of 3.0, with the
        # layers centered around x = 0

        # The full width of the graph is 3.0 * len(edge_layers), hence
        # input vertices have x = -(full width)/2 = 1.5 * len(edge_layers)
        x = -len(edge_layers) * 1.5
        inputs = decomposed.inputs
        for i, vertex_id in enumerate(inputs):
            self.vertices[vertex_id].x = x
            # Input vertices centered at y = 0.0 and 1.0 apart
            # Hence (len(inputs)-1)/2 is half the y-width of
            # the input layer of vertices
            self.vertices[vertex_id].y = i - (len(inputs) - 1)/2

        for layer in edge_layers:
            # Edges lie at x-coordinate centered in each width-3.0 block
            x += 1.5
            # The target vertices of the edges of the current layer
            next_vertex_layer = []
            for i, edge_id in enumerate(layer):
                self.edges[edge_id].x = x
                # Same reasoning as with input vertices above
                self.edges[edge_id].y = i - (len(layer) - 1)/2
                next_vertex_layer += decomposed.edges[edge_id].targets

            # Target vertices lie a bit closer to edges, since layout places
            # wire crossings between the vertices and their target edges
            x += 0.7
            for i, vertex_id in enumerate(next_vertex_layer):
                self.vertices[vertex_id].x = x
                # As above
                self.vertices[vertex_id].y = i - (len(next_vertex_layer) - 1)/2
            # Place current x coordinate at the end of this 3.0 block
            x += 0.8

        # Ensure outputs lie at end
        outputs = decomposed.outputs
        for i, vertex_id in enumerate(outputs):
            self.vertices[vertex_id].x = x
            # As above
            self.vertices[vertex_id].y = i - (len(outputs) - 1)/2

        # If there are no connections, no need for convex optimization
        if len(self.vertices) == 0 or len(self.edges) == 0:
            # If successful, update self.graph to account for new identities
            # added in layer_decomp
            self.graph = decomposed
            return

        # Convex optimization of y-coordinates

        # Variables to be optimized
        vertex_ys = Variable(len(self.vertices), 'vertex_ys')
        edge_ys = Variable(len(self.edges), 'edge_ys')

        # Mapping from optimized y coordinates to the vertices
        # and edges they are for
        vertexid_to_var = {vertex_id: i
                           for i, vertex_id in enumerate(self.vertices.keys())}
        edgeid_to_var = {edge_id: i
                         for i, edge_id in enumerate(self.edges.keys())}

        constraints = []
        optimize = []

        # Constraints and optimization targets for input and output vertices
        for vertices in (decomposed.inputs, decomposed.outputs):
            for i in range(len(vertices) - 1):
                v1 = vertices[i]
                v2 = vertices[i + 1]
                # Inputs and outputs should be at least 1.0 apart
                constraints.append(vertex_ys[vertexid_to_var[v2]]
                                   - vertex_ys[vertexid_to_var[v1]]
                                   >= Constant(1.0))
                # But try to keep them as close together as possible
                optimize.append((vertex_ys[vertexid_to_var[v2]]
                                - vertex_ys[vertexid_to_var[v1]])
                                # Weighting on this optimization target
                                * Constant(0.1))

        # Constraints and optimization targets for edges
        for layer in edge_layers:
            for i in range(len(layer) - 1):
                e1 = layer[i]
                e2 = layer[i + 1]
                # Vertical distance between centers of box for each edge
                # if placed vertically adjacent
                centre_spacing = (self.edges[e1].box_height
                                  + self.edges[e2].box_height) / 2
                # Make sure edges do not overlap
                constraints.append(edge_ys[edgeid_to_var[e2]]
                                   - edge_ys[edgeid_to_var[e1]]
                                   >= Constant(centre_spacing))
                # But try to keep them as close together as possible
                optimize.append((edge_ys[edgeid_to_var[e2]]
                                - edge_ys[edgeid_to_var[e1]])
                                # Weighting on this optimization target
                                * Constant(0.1))

        for vertex_id, vertex in decomposed.vertices.items():
            vertex_y = vertex_ys[vertexid_to_var[vertex_id]]

            # If the vertex has any source edges
            if len(vertex.sources) >= 1:
                # Note that in the case of a monogamous
                # hypergraph, there is only one source edge
                for edge_id in vertex.sources:
                    targets = decomposed.edges[edge_id].targets
                    # The ideal position of this vertex relative to the center
                    # of the box, if source vertices were equally spaced to
                    # occupy half the box width and vertically centered
                    # at the box center
                    y_shift = Constant(
                        0.0 if len(targets) <= 1 else
                        (self.edges[edge_id].box_height / 2)
                        * (targets.index(vertex_id))/(len(targets) - 1) - 0.5)
                    # Try to position the vertex in a nice position relative
                    # to this edge
                    optimize.append(edge_ys[edgeid_to_var[edge_id]] + y_shift
                                    - vertex_y)

            # If the vertex has any target edges
            if len(vertex.targets) >= 1:
                # Note that in the case of a monogamous hypergraph,
                # there is only one target edge
                for edge_id in vertex.targets:
                    sources = decomposed.edges[edge_id].sources
                    # The ideal position of this vertex relative to the center
                    # of the box, if source vertices were equally spaced to
                    # occupy half the box width and vertically centered
                    # at the box center
                    y_shift = Constant(
                        0.0 if len(sources) <= 1 else
                        (self.edges[edge_id].box_height / 2)
                        * (sources.index(vertex_id))/(len(sources) - 1) - 0.5)
                    # Try to position the vertex in a nice position relative
                    # to this edge
                    optimize.append(edge_ys[edgeid_to_var[edge_id]] + y_shift
                                    - vertex_y)

            # Try to position the above edges on either side of the vertex
            # to align as closely as possible after accounting for the
            # vertex's offset relative to these edges

        # Allows cvxpy to treat list of optimization targets as a vector
        optimize = cp.vstack(optimize)

        # Problem: minimize the absolute values of the optimization targets
        # subject to the constraints
        problem = Problem(Minimize(cp.norm1(optimize)), constraints)
        problem.solve()

        # For centering the vertex distribution at y = 0.0
        min_vertex_y = None
        max_vertex_y = None

        # If optmized vertex y values were found, register them in the
        # drawing instructions
        if vertex_ys.value is not None:
            for vertex_id, index in vertexid_to_var.items():
                y = vertex_ys.value[index]
                if min_vertex_y is None or y < min_vertex_y:
                    min_vertex_y = y
                if max_vertex_y is None or y > max_vertex_y:
                    max_vertex_y = y
                self.vertices[vertex_id].y = y

        # Center the vertex distribution at y = 0.0
        if min_vertex_y is not None and max_vertex_y is not None:
            y_shift = (min_vertex_y + max_vertex_y) / 2
            for vertex_draw_info in self.vertices.values():
                vertex_draw_info.y -= y_shift
        else:
            y_shift = 0.0

        # Register the optimized edge y values in the drawing
        # instructions, accounting for the centering of the
        # vertex distribution
        if edge_ys.value is not None:
            for edge_id, index in edgeid_to_var.items():
                self.edges[edge_id].y = edge_ys.value[index] - y_shift
                targets = decomposed.edges[edge_id].targets
                # Align target non-boundary vertices nicely with the edge
                for i, vertex_id in enumerate(targets):
                    if vertex_id not in decomposed.inputs + decomposed.outputs:
                        target_y_shift = (0 if len(targets) <= 1
                                          else (i / (len(targets) - 1)) - 0.5)
                        self.vertices[vertex_id].y = (self.edges[edge_id].y
                                                      + target_y_shift)

        # If successful, update self.graph to account for new identities added
        # in layer_decomp
        self.graph = decomposed

    def frobenius_convex_optimization_layout(self) -> None:
        """Determine x and y coordinates for vertices and hyperedges."""
        # Decompose the graph into layers of edges
        decomposed, layers = self.graph.frobenius_layer_decomp()

        # Initialize vertex and edge graphics items for decomposed hypergraph
        self.vertices = {
            vertex_id: VertexDrawInfo(
                0, 0, vertex.label if hasattr(vertex, 'label') else None
            )
            for vertex_id, vertex in decomposed.vertices.items()
        }
        self.edges = {
            edge_id: EdgeDrawInfo(0, 0, edge.module._get_name(), edge.identity,
                                  1.0 if len(edge.sources) == 1
                                  == len(edge.targets) else 2.0)
            for edge_id, edge in decomposed.edges.items()
        }

        # Initialize x-coordinates and rough y-coordinates
        # Each layer occupies an x-width of 3.0, with the
        # layers centered around x = 0

        # The full width of the graph is 1.5 * len(layers), hence
        # input vertices have x = -(full width)/2 = 0.75 * len(edge_layers)
        x = -len(layers) * 0.75 - 0.8

        for vertex_layer_num in range(0, len(layers), 2):
            x += 0.8
            vertex_layer = layers[vertex_layer_num]
            for i, vertex_id in enumerate(vertex_layer):
                if (vertex_layer_num == 0
                   and vertex_id in decomposed.inputs):
                    # Distinguish input vertices at beginning
                    self.vertices[vertex_id].x = x - 0.8
                else:
                    self.vertices[vertex_id].x = x
                # Vertices centered at y = 0.0 and 1.0 apart
                # Hence (len(vertex_layer)-1)/2 is half the y-width of
                # the layer of vertices
                self.vertices[vertex_id].y = i - (len(vertex_layer) - 1)/2

            # Edges lie at x-coordinate centered in each width-3.0 block
            x += 1.5
            edge_layer_num = vertex_layer_num + 1
            # If we are not at the output vertex layer, an edge layer follows
            if edge_layer_num != len(layers):
                edge_layer = layers[edge_layer_num]
                for i, edge_id in enumerate(edge_layer):
                    self.edges[edge_id].x = x
                    # Same reasoning as with input vertices above
                    self.edges[edge_id].y = i - (len(edge_layer) - 1)/2

                # Next vertex_layer lies a bit closer to edges, since wire
                # crossings occure between the vertices and their target edges
                x += 0.7

        # Distinguish output vertices at end
        x += 0.8
        outputs = decomposed.outputs
        for i, vertex_id in enumerate(outputs):
            self.vertices[vertex_id].x = x
            # As above
            self.vertices[vertex_id].y = i - (len(outputs) - 1)/2

        # If there are no connections, no need for convex optimization
        if len(self.vertices) == 0 or len(self.edges) == 0:
            # If successful, update self.graph to account for new identities
            # added in layer_decomp
            self.graph = decomposed
            return

        # Convex optimization of y-coordinates

        # Variables to be optimized
        vertex_ys = Variable(len(self.vertices), 'vertex_ys')
        edge_ys = Variable(len(self.edges), 'edge_ys')

        # Mapping from optimized y coordinates to the vertices
        # and edges they are for
        vertexid_to_var = {vertex_id: i
                           for i, vertex_id in enumerate(self.vertices.keys())}
        edgeid_to_var = {edge_id: i
                         for i, edge_id in enumerate(self.edges.keys())}

        constraints = []
        optimize = []

        # Constraints and optimization targets for input and output vertices
        for vertices in (decomposed.inputs, decomposed.outputs):
            for i in range(len(vertices) - 1):
                v1 = vertices[i]
                v2 = vertices[i + 1]
                # Inputs and outputs should be at least 1.0 apart
                constraints.append(vertex_ys[vertexid_to_var[v2]]
                                   - vertex_ys[vertexid_to_var[v1]]
                                   >= Constant(1.0))
                # But try to keep them as close together as possible
                optimize.append((vertex_ys[vertexid_to_var[v2]]
                                - vertex_ys[vertexid_to_var[v1]])
                                # Weighting on this optimization target
                                * Constant(0.1))

        # Constraints and optimization targets for edges
        for layer in layers[1::2]:  # odd indices are edge layers
            for i in range(len(layer) - 1):
                e1 = layer[i]
                e2 = layer[i + 1]
                # Vertical distance between centers of box for each edge
                # if placed vertically adjacent
                centre_spacing = (self.edges[e1].box_height
                                  + self.edges[e2].box_height) / 2
                # Make sure edges do not overlap
                constraints.append(edge_ys[edgeid_to_var[e2]]
                                   - edge_ys[edgeid_to_var[e1]]
                                   >= Constant(centre_spacing))
                # But try to keep them as close together as possible
                optimize.append((edge_ys[edgeid_to_var[e2]]
                                - edge_ys[edgeid_to_var[e1]])
                                # Weighting on this optimization target
                                * Constant(0.1))

        for vertex_id, vertex in decomposed.vertices.items():
            vertex_y = vertex_ys[vertexid_to_var[vertex_id]]

            # If the vertex has any source edges
            if len(vertex.sources) >= 1:
                # Note that in the case of a monogamous
                # hypergraph, there is only one source edge
                for edge_id in vertex.sources:
                    targets = decomposed.edges[edge_id].targets
                    # The ideal position of this vertex relative to the center
                    # of the box, if source vertices were equally spaced to
                    # occupy half the box width and vertically centered
                    # at the box center
                    y_shift = Constant(
                        0.0 if len(targets) <= 1 else
                        (self.edges[edge_id].box_height / 2)
                        * (targets.index(vertex_id))/(len(targets) - 1) - 0.5)
                    # Try to position the vertex in a nice position relative
                    # to this edge
                    optimize.append(edge_ys[edgeid_to_var[edge_id]] + y_shift
                                    - vertex_y)

            # If the vertex has any target edges
            if len(vertex.targets) >= 1:
                # Note that in the case of a monogamous hypergraph,
                # there is only one target edge
                for edge_id in vertex.targets:
                    sources = decomposed.edges[edge_id].sources
                    # The ideal position of this vertex relative to the center
                    # of the box, if source vertices were equally spaced to
                    # occupy half the box width and vertically centered
                    # at the box center
                    y_shift = Constant(
                        0.0 if len(sources) <= 1 else
                        (self.edges[edge_id].box_height / 2)
                        * (sources.index(vertex_id))/(len(sources) - 1) - 0.5)
                    # Try to position the vertex in a nice position relative
                    # to this edge
                    optimize.append(edge_ys[edgeid_to_var[edge_id]] + y_shift
                                    - vertex_y)

            # Try to position the above edges on either side of the vertex
            # to align as closely as possible after accounting for the
            # vertex's offset relative to these edges

        # Allows cvxpy to treat list of optimization targets as a vector
        optimize = cp.vstack(optimize)

        # Problem: minimize the absolute values of the optimization targets
        # subject to the constraints
        problem = Problem(Minimize(cp.norm1(optimize)), constraints)
        problem.solve()

        # For centering the vertex distribution at y = 0.0
        min_vertex_y = None
        max_vertex_y = None

        # If optmized vertex y values were found, register them in the
        # drawing instructions
        if vertex_ys.value is not None:
            for vertex_id, index in vertexid_to_var.items():
                y = vertex_ys.value[index]
                if min_vertex_y is None or y < min_vertex_y:
                    min_vertex_y = y
                if max_vertex_y is None or y > max_vertex_y:
                    max_vertex_y = y
                self.vertices[vertex_id].y = y

        # Center the vertex distribution at y = 0.0
        if min_vertex_y is not None and max_vertex_y is not None:
            y_shift = (min_vertex_y + max_vertex_y) / 2
            for vertex_draw_info in self.vertices.values():
                vertex_draw_info.y -= y_shift
        else:
            y_shift = 0.0

        # Register the optimized edge y values in the drawing
        # instructions, accounting for the centering of the
        # vertex distribution
        if edge_ys.value is not None:
            for edge_id, index in edgeid_to_var.items():
                self.edges[edge_id].y = edge_ys.value[index] - y_shift
                targets = decomposed.edges[edge_id].targets
                # Align target non-boundary vertices nicely with the edge
                for i, vertex_id in enumerate(targets):
                    if vertex_id not in decomposed.inputs + decomposed.outputs:
                        target_y_shift = (0 if len(targets) <= 1
                                          else (i / (len(targets) - 1)) - 0.5)
                        self.vertices[vertex_id].y = (self.edges[edge_id].y
                                                      + target_y_shift)

        # If successful, update self.graph to account for new identities added
        # in layer_decomp
        self.graph = decomposed
