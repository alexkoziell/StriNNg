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
"""Functionality for drawing hypergraphs with matplotlib."""

from __future__ import annotations
from typing import Any, Dict, List, Sequence, Set, Tuple

from matplotlib.backend_bases import MouseEvent  # type: ignore
from matplotlib.patches import Circle, PathPatch, Rectangle  # type: ignore
from matplotlib.path import Path  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from matplotlib.text import Annotation  # type: ignore

from strinng.hypergraph import Hyperedge, Hypergraph, Vertex
from strinng.gui.drawinfo import HypergraphDrawInfo


class MplVertex(Circle):
    """A matplotlib `Patch` object for drawing vertices.

    This subclasses the matplotlib `Circle` class, inheriting
    its attributes as well as adding a `connected_wires` attribute.

    Attributes:
        connected_wires: A set of :py:class:`MplWire` instances
                         that have start or end point at this vertex.
    """

    def __init__(self, xy: Sequence[float], radius: float,
                 label: str | None = None,
                 **kwargs: Any) -> None:
        """Initialize an :py:class:`MplVertex` object."""
        self.label = label
        self.connected_wires: Set[MplWire] = set()
        self.annotation: Annotation | None = None
        super().__init__(xy, radius, **kwargs)

    def update_annotation(self) -> None:
        """Annotate the box for this vertex, if applicable."""
        if self.annotation is not None:
            self.annotation.set_position(self.get_center())
        elif self.label is not None:
            ax = self.axes
            label = self.label
            x, y = self.get_center()
            self.annotation = ax.annotate(label,
                                          (x, y - 2.5 * self.radius),
                                          ha='center', va='center')


class MplEdge(Rectangle):
    """A matplotlib `Patch` object for drawing hyperedges.

    This subclasses the matplotlib `Rectangle` class, inheriting
    its attributes as well as adding some additional attributes.

    Attributes:
        identity: Whether this hyperedge is an identity or not.
        label: A label to be drawn on the hyperedge box.
        sources: A list of source :py:class:`MplVertex` objects
                 for this hyperedge.
        targets: A list of target :py:class:`MplVertex` objects
                 for this hyperedge.
        annotation: A matplotlib `Annotation` object drawn on the
                    box.
        connected_wires: A set of :py:class:`MplWire` instances
                         that have start or end point at this edge.
    """

    def __init__(self, xy: Sequence[float],
                 width: float, height: float,
                 identity: bool, label: str | None,
                 sources: List[MplVertex],
                 targets: List[MplVertex],
                 **kwargs: Any) -> None:
        """Initialize an :py:class:`MplEdge` object."""
        self.identity = identity
        self.label = str(label)
        self.sources = sources
        self.targets = targets
        self.annotation: Annotation | None = None
        self.connected_wires: Set[MplWire] = set()
        super().__init__(xy, width, height, **kwargs)

    def update_annotation(self) -> None:
        """Annotate the box for this hyperedge, if applicable.

        If there is no label for this hyperedge or it is an identity edge,
        no annotation is made.
        """
        if self.annotation is not None:
            self.annotation.set_position(self.get_center())
        elif self.label is not None and not self.identity:
            ax = self.axes
            self.annotation = ax.annotate(self.label, self.get_center(),
                                          weight='bold',
                                          ha='center', va='center')


class MplWire(PathPatch):
    """A matplotlib `Patch` object for drawing wires.

    A wire indicates a connection between a vertex and a hyperedge.
    This subclasses the matplotlib `PathPatch` class, inheriting
    its attributes as well as adding some additional attributes.

    Attributes:
        source: The :py:class:`MplVertex` or :py:class:`MplEdge`
                this wire begins at.
        target: The :py:class:`MplVertex` or :py:class:`MplEdge`
                this wire ends at.
        x_shift: The x offset from the edge x coordinate and where
                 the wire meets the edge. Cached to save computation.
        y_shift: The y offset from the edge y coordinate and where
                 the wire can meet the edge to be as straight as possible.
                 Cached to save computation.
    """

    def __init__(self,
                 source: MplVertex | MplEdge,
                 target: MplEdge | MplVertex,
                 x_shift: float, y_shift: float,
                 **kwargs: Any) -> None:
        """Initialize an :py:class:`MplWire` object."""
        if not (
            isinstance(source, MplVertex) and isinstance(target, MplEdge)
            or isinstance(source, MplEdge) and isinstance(target, MplVertex)
        ):
            raise ValueError(
                'Wire must have a vertex at one end and an edge at the other.'
            )
        self.source = source
        self.target = target
        self.x_shift = x_shift
        self.y_shift = y_shift
        path = self.calculate_path(set_path=False)
        super().__init__(path, **kwargs)

    def calculate_path(self, set_path: bool = False) -> Path:
        """Calculate the cubic bezier curve to draw this edge."""
        vertex_to_edge = isinstance(self.source, MplVertex)
        mpl_vertex = self.source if vertex_to_edge else self.target
        mpl_edge = self.target if vertex_to_edge else self.source

        if not isinstance(mpl_edge, MplEdge):
            raise ValueError('Source/target type incorrect')

        if vertex_to_edge:
            start_x, start_y = mpl_vertex.get_center()
            end_x = mpl_edge.get_x()
            end_y = mpl_edge.get_y() + self.y_shift
            dx = abs(start_x - end_x)
        else:
            start_x = mpl_edge.get_x() + self.x_shift
            start_y = mpl_edge.get_y() + self.y_shift
            end_x, end_y = mpl_vertex.get_center()
            dx = abs(start_x - end_x)

        # Create the Path object for the cubic Bezier curve
        path = Path([(start_x, start_y),  # start point
                    (start_x + dx * 0.4, start_y),  # control point 1
                    (end_x - dx * 0.4, end_y,),  # control point 2
                    (end_x, end_y)],  # end point
                    [Path.MOVETO] + [Path.CURVE4] * 3)

        if set_path:
            self.set_path(path)

        return path


class MplArtist:
    """A class for drawing hypergraphs in matplotlib."""

    vertex_radius: float = 3e-2
    box_width: float = 0.5
    box_height: float = 1.0
    x_scale: float = 2e-1
    y_scale: float = 2e-1

    def __init__(self, hypergraph: Hypergraph,
                 layout: str = 'convex_opt',
                 annotate_vertices: bool = False) -> None:
        """Initialize an :py:class:`MplHypergraph` object."""
        graph_draw_info = HypergraphDrawInfo(hypergraph, layout)
        self.draw_info = graph_draw_info
        self.annotate_vertices = annotate_vertices
        self.vertices: Dict[int, MplVertex] = dict()
        self.edges: Dict[int, MplEdge] = dict()
        self.wires: Dict[Tuple[MplVertex | MplEdge,
                               MplEdge | MplVertex,
                               int, bool],
                         PathPatch] = dict()
        self.drag_vertices: Dict[int, MplVertex] = dict()
        self.drag_edges: Dict[int, MplEdge] = dict()
        self.drag_wires: Set[MplWire] = set()
        self.dragging: bool = False

    def init_vertices(self) -> None:
        """Calculate the :py:class:`MplVertex` objects to be drawn."""
        for vertex_id, vertex_draw_info in self.draw_info.vertices.items():
            x = vertex_draw_info.x * self.x_scale
            y = vertex_draw_info.y * self.y_scale
            radius = self.vertex_radius * (self.x_scale + self.y_scale)
            label = vertex_draw_info.label
            self.vertices[vertex_id] = MplVertex((x, y), radius, label,
                                                 color='black')

    def init_edges(self) -> None:
        """Calculate the :py:class:`MplEdge` objects to be drawn."""
        for edge_id, edge_draw_info in self.draw_info.edges.items():
            width = self.x_scale * self.box_width
            height = self.y_scale * self.box_height
            if edge_draw_info.identity:
                width /= 10
                height /= 10

            x_shift = width / 2
            y_shift = height / 2

            x = edge_draw_info.x * self.x_scale - x_shift
            y = edge_draw_info.y * self.y_scale - y_shift

            edge = self.draw_info.graph.edges[edge_id]
            sources = [self.vertices[vertex_id] for vertex_id in edge.sources]
            targets = [self.vertices[vertex_id] for vertex_id in edge.targets]

            self.edges[edge_id] = MplEdge((x, y), width, height,
                                          edge_draw_info.identity,
                                          edge_draw_info.label,
                                          sources, targets,
                                          linewidth=1, edgecolor='blue',
                                          facecolor='blue', alpha=0.3)

    def init_wires(self) -> None:
        """Calculate the :py:class:`MplWire` objects to be drawn."""
        # Given that every wire has an edge at one of its end points, we are
        # guaranteed to find all the wires by iterating over all the edges
        for edge_id in self.draw_info.edges:
            self.init_wires_for_edge(edge_id)

    def init_wires_for_edge(self, edge_id: int) -> None:
        """Compute :py:class:`MplVertex` objects for a :py:class:`MplEdge`."""
        mpl_edge = self.edges[edge_id]

        x_shift = 0 if mpl_edge.identity else self.box_width * self.x_scale

        for vertex_circles, vertex_to_edge in ((mpl_edge.sources, True),
                                               (mpl_edge.targets, False)):
            num_ports = len(vertex_circles)
            for i, vertex_circle in enumerate(vertex_circles):
                if num_ports == 1:
                    y_shift = mpl_edge.get_height() / 2
                else:
                    # a bit of playing around with scaling
                    # and numbers to get lines straight
                    y_shift = mpl_edge.get_height() * (1 / 1.6) * (
                        i / (num_ports - 1) + 0.3
                    )

                source: MplVertex | MplEdge = (vertex_circle if vertex_to_edge
                                               else mpl_edge)
                target: MplVertex | MplEdge = (mpl_edge if vertex_to_edge
                                               else vertex_circle)
                wire = MplWire(source, target, x_shift, y_shift,
                               facecolor='none', edgecolor='blue')
                vertex_circle.connected_wires.add(wire)
                mpl_edge.connected_wires.add(wire)
                self.wires[(source, target, i, vertex_to_edge)] = wire

    def draw(self, figsize: Tuple[int, int] = (10, 8)) -> None:
        """Draw the hypergraph associated with `self` in matplotlib."""
        # Create a figure and axis
        fig, ax = plt.subplots(figsize=figsize)

        # Calculate the matplotlib patches for the vertices, edges and wires
        self.init_vertices()
        self.init_edges()
        # Wires must be initialized after vertices and edges
        self.init_wires()

        for edge in self.edges.values():
            ax.add_patch(edge)
            # Must be called after adding edge to axes
            edge.update_annotation()

        for wire in self.wires.values():
            ax.add_patch(wire)

        # Plot vertices over wires
        for vertex in self.vertices.values():
            ax.add_patch(vertex)
            # Must be called after adding vertex to axes
            if self.annotate_vertices:
                vertex.update_annotation()

        # Set the aspect ratio and auto-adjust limits of the plot
        ax.set_aspect('auto', 'box')
        ax.autoscale_view()

        # Hide the axis ticks and labels, but keep the bounding box
        plt.tick_params(axis='both', which='both',
                        bottom=False, left=False,
                        labelbottom=False, labelleft=False)

        # Invert the y axis
        ax.invert_yaxis()

        # Use all available space in plot window
        fig.tight_layout()

        # Add interactivity
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: self.on_press(event))
        fig.canvas.mpl_connect('button_release_event',
                               lambda event: self.on_release(event))
        fig.canvas.mpl_connect('motion_notify_event',
                               lambda event: self.on_motion(event))

    def on_press(self, event: MouseEvent) -> None:
        """Actions to be taken when a mouse button is pressed."""
        self.dragging = True

        for idx, vertex in self.vertices.items():
            if vertex.contains(event)[0]:
                self.drag_vertices[idx] = vertex
                self.drag_wires.update(vertex.connected_wires)

        for idx, edge in self.edges.items():
            if edge.contains(event)[0]:
                self.drag_edges[idx] = edge
                self.drag_wires.update(edge.connected_wires)

        self.prev_drag_x = event.xdata
        self.prev_drag_y = event.ydata

    def on_motion(self, event: MouseEvent) -> None:
        """Actions to be taken when the mouse is moved."""
        if (self.dragging and event.xdata and event.ydata
           and self.prev_drag_x and self.prev_drag_y):
            # Calculate the distance dragged
            dx = event.xdata - self.prev_drag_x
            dy = event.ydata - self.prev_drag_y

            # uncomment and adjust accordingly if
            # interactivity requires too much compute
            # if abs(dx) < 5e-2 and abs(dy) < 5e-2:
            #     return

            if len(self.drag_vertices) > 0:
                # Update the position of vertices being dragged
                for vertex in self.drag_vertices.values():
                    prev_x, prev_y = vertex.get_center()
                    vertex.set_center((prev_x + dx, prev_y + dy))
                    if self.annotate_vertices:
                        vertex.update_annotation()

            if len(self.drag_edges) > 0:
                # Update the position of edges being dragged
                for edge in self.drag_edges.values():
                    prev_x = edge.get_x()
                    prev_y = edge.get_y()
                    edge.set_xy((prev_x + dx, prev_y + dy))
                    edge.update_annotation()

            # Update the wires
            for wire in self.drag_wires:
                wire.calculate_path(set_path=True)

            self.prev_drag_x = event.xdata
            self.prev_drag_y = event.ydata

            # Redraw the plot
            plt.draw()

    def on_release(self, _: MouseEvent) -> None:
        """Actions to be taken when a mouse button is released."""
        self.dragging = False
        self.drag_vertices.clear()
        self.drag_edges.clear()
        self.drag_wires.clear()
