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
"""Hypergraph Matching."""
from __future__ import annotations
from copy import deepcopy
from typing import Iterable, Iterator

from strinng.hypergraph import Hypergraph
from strinng.rule import Rule


class Match:

    def __init__(self,
                 domain: Hypergraph,
                 codomain: Hypergraph) -> None:
        self.domain = domain
        self.codomain = codomain
        self.vertex_map: dict[int, int] = {}
        self.vertex_image: set[int] = set()
        self.edge_map: dict[int, int] = {}
        self.edge_image: set[int] = set()

    def match_vertex(self, domain_vertex: int,
                     codomain_vertex: int) -> bool:
        """Try to match a vertex in the domain with one in the codomain.

        This map must satisfy various conditions, such as only being
        allowed to be non-injective on the boundary domain vertices.
        """
        # If the vertex is already mapped, only check the new mapping is
        # consistent with the current match.
        if domain_vertex in self.vertex_map:
            return self.vertex_map[domain_vertex] == codomain_vertex

        # Ensure the mapping preserves vertex type.
        domain_vertex_type = self.domain.vertices[domain_vertex].size
        codomain_vertex_type = self.codomain.vertices[codomain_vertex].size
        if domain_vertex_type != codomain_vertex_type:
            return False

        # Ensure non-boundary vertices in the domain are not mapped to
        # boundary vertices in the codomain.
        if (self.codomain.is_boundary(codomain_vertex)
           and not self.domain.is_boundary(domain_vertex)):
            return False

        # Matches must be injective everywhere except the boundary, so if the
        # domain vertex is already mapped to another codomain vertex, check
        # whether this non-injective mapping is permitted.
        if codomain_vertex in self.vertex_image:
            # If the domain vertex we are trying to add is not a boundary
            # vertex, it cannot be used in a non-injective mapping.
            if not self.domain.is_boundary(domain_vertex):
                return False
            # If any vertices already mapped to the codomain vertex, they must
            # also be boundary vertices for an allowed non-injective mapping.
            for mapped_vertex, image_vertex in self.vertex_map.items():
                if (image_vertex == codomain_vertex
                   and not self.domain.is_boundary(mapped_vertex)):
                    return False

        # If a new and consistent map is found,
        # add it to the vertex map of this match.
        self.vertex_map[domain_vertex] = codomain_vertex
        self.vertex_image.add(codomain_vertex)

        # Unless the domain vertex is a boundary vertex, check that the number
        # of adjacent edges of the codomain vertex is the same as the number
        # for the domain vertex.
        # Because matchings are required to be injective on edges, this will
        # guarantee that the gluing conditions are satisfied.
        if not self.domain.is_boundary(domain_vertex):
            if (len(self.domain.vertices[domain_vertex].sources)
               != len(self.codomain.vertices[codomain_vertex].sources)):
                return False
            if (len(self.domain.vertices[domain_vertex].targets)
               != len(self.codomain.vertices[codomain_vertex].targets)):
                return False

        # If a new consistent map is added that satisfies the gluing
        # conditions, we are successful.
        return True
    
    def match_edge(self, domain_edge: int, codomain_edge: int) -> bool:
        """Try to map `domain_edge` to `codomain_edge`.

        This must satisfy certain conditions, such as being injective and
        having consistency with the vertex map.

        Returns:
            `True` if a consistent match is found mapping `domain_edge` to
            `codomain_edge`, otherwise `False`.
        """


        # Check the values of the domain and codomain edges match.
        domain_module = self.domain.edges[domain_edge].module._get_name()
        codomain_module = self.codomain.edges[codomain_edge].module._get_name()
        if domain_module != codomain_module:
            return False

        # The edge map must be injective.
        if codomain_edge in self.edge_image:
            return False

        # If the modules match and the codomain edge has not already been
        # mapped to, map domain edge to codomain edge.
        self.edge_map[domain_edge] = codomain_edge
        self.edge_image.add(codomain_edge)

        # Check a vertex map consistent with this edge pairing exists.
        domain_sources = self.domain.edges[domain_edge].sources
        codomain_sources = self.codomain.edges[codomain_edge].sources
        domain_targets = self.domain.edges[domain_edge].targets
        codomain_targets = self.codomain.edges[codomain_edge].targets
        vertices_to_check = zip(domain_sources + domain_targets,
                                codomain_sources + codomain_targets)

        for domain_vertex, codomain_vertex in vertices_to_check:
            # Each vertex that is already mapped needs to be consistent.
            if (domain_vertex in self.vertex_map
               and self.vertex_map[domain_vertex] != codomain_vertex):
                return False
            # Otherwise, a consistent match must be found vertex for unmapped
            # source and target vertices.
            else:
                if not self.match_vertex(domain_vertex, codomain_vertex):
                    return False

        return True

    def domain_neighbourhood_mapped(self, vertex: int) -> bool:
        """Return whether all adjacent edges of a domain vertex are mapped."""
        return (all(e in self.edge_map
                    for e in self.domain.vertices[vertex].sources)
                and
                all(e in self.edge_map
                    for e in self.domain.vertices[vertex].targets))

    def map_scalars(self) -> bool:
        """Try to extend the match by mapping all scalars (i.e. 0 -> 0 edges).

        Note that any matchings of scalars will yield isomorphic results under
        rewriting, so we don't return a list of all the possible matchings.

        Returns:
            `True` if all scalars in the domain are mapped injectively to
            scalars in the codomain, otherwise `False`.
        """
        # Find all scalars in the codomain
        codomain_scalars = []
        for edge_id, edge in self.codomain.edges.items():
            if len(edge.sources) == 0 and len(edge.targets) == 0:
                codomain_scalars.append((edge_id, edge.module._get_name()))

        # Greedily try to map scalar edges in the domain to scalar
        # edges in the codomain with the same value.
        for edge_id, edge in self.domain.edges.items():
            if len(edge.sources) != 0 or len(edge.targets) != 0:
                continue
            found_match = False
            for i, (codomain_scalar, module) in enumerate(codomain_scalars):
                if module == edge.module._get_name():
                    # Map the domain scalar to the first codomain scalar
                    # available with the same value.
                    self.edge_map[edge_id] = codomain_scalar
                    self.edge_image.add(codomain_scalar)
                    found_match = True
                    # Since the edge map must be injective, if a scalar in the
                    # codomain is mapped to, remove it from the list of
                    # candidates for future domain scalars to be mapped to.
                    codomain_scalars.pop(i)
                    break
            if not found_match:
                return False

        return True

    def more(self) -> list[Match]:
        """Return any matches extending `self` by a single vertex or edge."""
        # A list of partial matches the same as this one,
        # but matching 1 more vertex or edge.
        extended_matches = []

        # First, try to add an edge adjacent to any domain vertices
        # that have already been matched.
        for domain_vertex in self.vertex_map:
            # If all the edges adjacent to the current vertex have
            # already been matched, continue.
            if self.domain_neighbourhood_mapped(domain_vertex):
                continue

            # Otherwise, try to extend the match by mapping an edge
            # adjacent to the current vertex into the codomain graph.
            codomain_vertex = self.vertex_map[domain_vertex]

            # Try to extend the match by mapping an adjacent source edge.
            for edge in self.domain.vertices[domain_vertex].sources:
                # If the edge has already been matched, continue.
                if edge in self.edge_map:
                    continue
                # Otherwise, try to map this edge into the codomain graph.
                for codomain_edge in self.codomain.vertices[codomain_vertex].sources:
                    potential_new_match = deepcopy(self)
                    # If the edge is successfully mapped to an edge in the
                    # codomain graph, extend the match with this mapping.
                    if potential_new_match.match_edge(edge, codomain_edge):
                        extended_matches.append(potential_new_match)
                return extended_matches

            # If there are no unmapped source edges, try to
            # extend the match by mapping an adjacent target edge.
            for edge in self.domain.vertices[domain_vertex].targets:
                # If the edge has already been matched, continue.
                if edge in self.edge_map:
                    continue
                # Otherwise, try to map this edge into the codomain graph.
                for codomain_edge in self.codomain.vertices[codomain_vertex].targets:
                    potential_new_match = deepcopy(self)
                    # If the edge is successfully mapped to an edge in the
                    # codomain graph, extend the match with this mapping.
                    if potential_new_match.match_edge(edge, codomain_edge):
                        extended_matches.append(potential_new_match)
                return extended_matches

        # If all domain edges adjacent to matched domain vertices have already
        # been matched, try to match an unmatched domain vertex.
        for domain_vertex in self.domain.vertices.keys():
            # If the vertex has already been matched into the codomain graph,
            # continue. (Note we have looked at the edge-neighbourhood of
            # these vertices above)
            if domain_vertex in self.vertex_map:
                continue

            # Try to map the current domain vertex to any of the codomain
            # vertices, extending the current match with this map when
            # successful.
            for codomain_vertex in self.codomain.vertices.keys():
                potential_new_match = deepcopy(self)
                if potential_new_match.match_vertex(domain_vertex,
                                                    codomain_vertex):
                    extended_matches.append(potential_new_match)
            return extended_matches

        # If no extended matches were found, return an empty list.
        return []

    def is_total(self) -> bool:
        """Return whether all domain vertices and edges have been mapped."""
        return (len(self.vertex_map) == len(self.domain.vertices)
                and len(self.edge_map) == len(self.domain.edges))

    def is_surjective(self) -> bool:
        """Return whether the vertex and edge maps are surjective."""
        return (len(self.vertex_image) == len(self.codomain.vertices)
                and len(self.edge_image) == len(self.codomain.edges))

    def is_injective(self) -> bool:
        """Return whether the vertex and edge maps are injective."""
        # Since the edge map is always injective, we only need to check
        # the vertex map is injective.
        return len(self.vertex_map) == len(self.vertex_image)

    def is_convex(self) -> bool:
        """Return whether this match is convex.

        A match is convex if:
            - It is injective.
            - Its image in the codomain hypergraph is a convex sub-hypergraph.
              This means that for any two nodes in the sub-hypergraph and any
              path between these two nodes, every hyperedge along the path is
              also in the sub-hypergraph.
        """
        # Check the match is injective.
        if not self.is_injective():
            return False

        # Check that the image sub-hypergraph is convex.
        # Get all successors of vertices in the image of the output
        # of the domain graph.
        output_image_successors = self.codomain.successors(
            [self.vertex_map[v] for v in self.domain.outputs
             if v in self.vertex_map])
        # Check there is no path from any vertices in the image of the domain
        # outputs to a vertex in the image of the domain inputs.
        for v in self.domain.inputs:
            if (v in self.vertex_map
               and self.vertex_map[v] in output_image_successors):
                return False
        return True


class Matches(Iterator):
    """An iterator over matches of one hypergraph into another.

    This class can be used to iterate over total matches, optionally
    requiring these matches to be convex.

    A class instance works by keeping a stack of partial or total matches.
    When it is iterated over, it pops a match from its match stack if it is
    non-empty, otherwise the iteration is stopped. If a match has been popped,
    the instance returns the match if it is total (and convex if required).
    Otherwise, the instance tries to extend the match and add any extended
    matches to the stack, then continues this process of popping off the match
    stack and extending if possible until a valid match is found and returned.
    """
    def __init__(self, domain: Hypergraph, codomain: Hypergraph,
                 initial_match: Match | None = None,
                 convex: bool = True) -> None:
        """Initialize a :class:`Matches` instance.

        The matches are of the domain graph into the codomain graph.
        Args:
            dom: The domain graph of the matches.
            cod: The codomain graph of the matches.
            initial_match: An optional starting match from which to build
                           further matches.
            convex: Whether to only accept convex matches.
        """
        if initial_match is None:
            initial_match = Match(domain=domain, codomain=codomain)
        self.convex = convex

        # Try to map scalars on the initial match.
        if initial_match.map_scalars():
            self.match_stack = [initial_match]
        # If the scalars could not be mapped, set the match
        # stack to be empty. This means that not suitable matches
        # will be found by this class instance.
        else:
            self.match_stack = []

    def __iter__(self) -> Iterator:
        """Return an iterator over matches of domain into codomain."""
        return self

    def __next__(self) -> Match:
        """Return the next suitable match found.

        A 'suitable' match is one that is total and, if `self.convex == True`,
        convex.
        """
        while len(self.match_stack) > 0:
            # Pop the match at the top of the match stack
            match = self.match_stack.pop()
            # If the match is total (and convex if required), return it.
            if match.is_total():
                if self.convex:
                    if match.is_convex():
                        return match
                    else:
                        pass
                else:
                    return match
            # If the match at the top of the match stack was not total
            # (and convex if required), try to extend the match at the
            # top of the stack and add the results to the match stack.
            else:
                self.match_stack += match.more()
        # If a suitable match was not found, stop the iteration.
        raise StopIteration


def match_rule(rule: Rule,
               graph:Hypergraph,
               convex: bool = True) -> Iterable[Match]:
    """Return matches of the left side of `rule` into `graph`."""
    return Matches(rule.lhs, graph, convex=convex)


def find_iso(domain_graph: Hypergraph,
             codomain_graph: Hypergraph) -> Match | None:
    """Return an isomorphism between graphs g and h if found, otherwise `None`.

    If found, the isomorphism is returned as a :py:class:`Matches` instance,
    whose vertex and edge maps are bijections.
    """
    domain_inputs = domain_graph.inputs
    domain_outputs = domain_graph.outputs
    codomain_inputs = codomain_graph.inputs
    codomain_outputs = codomain_graph.outputs
    # First, check the domains and codomains are equal between the two graphs.
    if (domain_graph.domain() != codomain_graph.domain()
       or domain_graph.codomain() != codomain_graph.codomain()):
        return None

    # Try to find an initial match mapping one of the boundary vertices of the
    # domain graph to the corresponding boundary vertex (the vertex in the same
    # input/output location) of the codomain graph.
    # If no initial match is found, return `None`.
    initial_match = Match(domain=domain_graph, codomain=codomain_graph)
    for i in range(len(domain_inputs)):
        if not initial_match.match_vertex(domain_inputs[i],
                                          codomain_inputs[i]):
            return None
    for i in range(len(domain_outputs)):
        if not initial_match.match_vertex(domain_outputs[i],
                                          codomain_outputs[i]):
            return None

    # If an initial match is found, try to find a total and surjective match
    # of the domain graph into the codomain graph.
    for match in Matches(domain=domain_graph, codomain=codomain_graph,
                         initial_match=initial_match, convex=False):
        if match.is_surjective():
            return match

    # If a total surjective match is not found, return `None`.
    return None
